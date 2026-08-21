"""
preprocess_dataset.py
────────────────────────────────────────────────────────────────
원본 JSONL → token_weights 포함 훈련용 JSONL 변환 스크립트.

입력 JSONL 샘플 형식:
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...future state html..."}
  ],
  "current_html": "<xml>...현재 상태...</xml>",
  "future_html":  "<xml>...미래 상태...</xml>"   ← assistant content와 동일
}

출력 JSONL 샘플 형식:
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...future state html..."}
  ],
  "token_weights": [1.0, 2.0, 3.0, ...]   ← assistant 부분만 (prefix 제외)
  "_diff_counts":  {"ADDED": 3, "MODIFIED": 2, "UNCHANGED": 10}   ← 디버그용
}

★ token_weights는 assistant 토큰에 대한 가중치만 저장한다.
  prefix(system+user) 부분은 훈련 시 이미지 토큰 확장 등으로 길이가 달라지므로,
  LlamaFactory collator에서 labels(-100)를 기반으로 경계를 판단해 복원한다.

사용법:
  python preprocess_dataset.py \
    --input  data/raw_train.jsonl \
    --output data/weighted_train.jsonl \
    --model  Qwen/Qwen2.5-7B-Instruct \
    --w-added 3.0 --w-modified 2.0 --w-unchanged 1.0
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import token_weight_builder_v2
from transformers import AutoTokenizer

# 메트릭 모듈은 CLI 옵션에 따라 런타임에 로드
# (v1: hungarian_diff / v2: hungarian_diff_v2 / v3: hungarian_diff_v3)
_hd = None  # hungarian_diff 모듈 참조 (classify_diff, summarize_diff)
# 가중치 빌더도 함께 갈린다 — v3 는 (diff_type × derivability) 2축 weight_map 이라
# v2 빌더에 넘기면 키 형태가 안 맞는다. 기본값은 v2 (기존 동작 유지).
_tw = token_weight_builder_v2


class SampleFailure(Exception):
    """diff / weight 생성 실패. --on-error fail 일 때 실행을 중단시킨다."""

    def __init__(self, stage: str, line_no: int, cause: Exception):
        super().__init__(f"line {line_no}: {stage} 실패 ({cause})")
        self.stage = stage
        self.line_no = line_no
        self.cause = cause


def _load_metric(version: str) -> None:
    """metric 버전에 맞는 diff 모듈 + 가중치 빌더를 함께 바인딩한다.

    v3 는 diff(ADDED/MODIFIED/UNCHANGED) 위에 **액션 유도성** 축을 얹는다 — 모델이
    원리적으로 알 수 없는 콘텐츠(서버가 내려준 상품명·가격 등)를 full weight 로
    외우게 하면 hallucination 을 학습시키기 때문이다. 축이 늘면 weight_map 의 키
    형태가 바뀌므로 빌더도 같이 갈아야 한다.
    """
    global _hd, _tw
    if version == "v3":
        _hd = importlib.import_module("hungarian_diff_v3")
        _tw = importlib.import_module("token_weight_builder_v3")
    elif version == "v2":
        _hd = importlib.import_module("hungarian_diff_v2")
        _tw = token_weight_builder_v2
    else:
        _hd = importlib.import_module("hungarian_diff")
        _tw = token_weight_builder_v2


# ── 메시지 포맷 정규화 ──────────────────────────────────────────────────
# 데이터셋에 따라 from/value 또는 role/content 형식이 섞여 있을 수 있음.
# 내부적으로 role/content 형식으로 통일하여 처리한다.

_ROLE_MAP = {"human": "user", "gpt": "assistant"}


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """from/value 형식을 role/content 형식으로 변환. 이미 role/content이면 그대로 반환."""
    if messages and "role" in messages[0]:
        return messages

    return [
        {
            "role": _ROLE_MAP.get(m["from"], m["from"]),
            "content": m["value"],
        }
        for m in messages
    ]


def _extract_current_html(user_content: str) -> str:
    """user 메시지에서 current HTML 섹션을 추출.

    지원 포맷:
      1. '## Current State' 헤더 (마크다운 스타일)
      2. 'Current UI State:' 헤더 ~ '[Screenshot]' 앞까지
    """
    lines = user_content.split("\n")

    # 포맷 1: ## Current State ~ 다음 ## 헤더 앞까지
    in_current = False
    current_lines: list[str] = []
    for line in lines:
        if "## Current State" in line:
            in_current = True
            continue
        if line.startswith("## "):
            in_current = False
        if in_current:
            current_lines.append(line)
    result = "\n".join(current_lines).strip()
    if result:
        return result

    # 포맷 2: Current UI State: ~ [Screenshot] 또는 다음 헤더 앞까지
    in_current = False
    current_lines = []
    for line in lines:
        if "Current UI State:" in line:
            in_current = True
            continue
        if "[Screenshot]" in line or line.startswith("## "):
            in_current = False
        if in_current:
            current_lines.append(line)
    return "\n".join(current_lines).strip()


# ── 모델별 chat template prefix 생성 함수 ────────────────────────────────
# 실제 LlamaFactory가 적용하는 템플릿과 완전히 일치해야 token 위치가 정확함.
# 지원하지 않는 모델은 이 딕셔너리에 추가하거나 --template 인자로 확장 가능.


def _make_prefix_qwen(system: str, user: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _make_prefix_llama3(system: str, user: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _make_prefix_default(system: str, user: str) -> str:
    """알 수 없는 모델 fallback — 실제 템플릿 확인 후 교체 권장."""
    return f"<|system|>\n{system}<|user|>\n{user}<|assistant|>\n"


TEMPLATE_MAP = {
    "qwen": _make_prefix_qwen,
    "llama3": _make_prefix_llama3,
    "default": _make_prefix_default,
}


def detect_template(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "qwen"
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    return "default"


# ── 단일 샘플 처리 ────────────────────────────────────────────────────────


def process_sample(
    sample: dict,
    tokenizer,
    prefix_fn,
    weight_map: dict[str, float],
    line_no: int,
    on_error: str = "fail",
) -> tuple[dict, str]:
    """
    샘플 하나를 받아 token_weights를 계산하여 추가한 뒤 (sample, status) 반환.
    current_html / future_html 필드는 제거됨.

    status: "action"        — action 샘플 (설계상 uniform 1.0, 오류 아님)
            "ok"            — diff 기반 가중치 정상 산출
            "diff_fail"     — classify_diff 실패 → uniform fallback
            "weight_fail"   — build_token_weights 실패 → uniform fallback

    on_error: "fail"    — 실패 시 SampleFailure 를 올려 실행 중단 (기본, fail-closed)
              "uniform" — 실패 시 균일 1.0 로 폴백하되 **성공으로 집계하지 않는다**
              "skip"    — 실패 레코드를 출력에서 제외
    """
    # ── 메시지 포맷 정규화 (내부 처리용만, 원본 messages는 유지) ──────────
    normalized = _normalize_messages(sample["messages"])
    # ★ sample["messages"]는 건드리지 않음 — LlamaFactory는 from/value 형식을 기대

    current_html = sample.pop("current_html", "")
    future_html = sample.pop("future_html", "")

    system = next((m["content"] for m in normalized if m["role"] == "system"), "")
    user = next((m["content"] for m in normalized if m["role"] == "user"), "")
    asst = next((m["content"] for m in normalized if m["role"] == "assistant"), "")

    # ── action 샘플 분기 ──────────────────────────────────────────────────
    # images 1개 = state_pred, 그 외(EXP05 action=2장 / EXP07 이미지 제거 action=0장)
    # = action_pred (assistant가 <action>{...}</action>).
    # action 출력에는 diff 개념이 없으므로 균일 1.0을 부여한다.
    # diff 경로로 흘려보내면 "diff 없음 → 전부 baseline(0.25)"으로 잘못 감쇠된다.
    is_action = len(sample.get("images", [])) != 1
    starts_with_action = asst.lstrip().startswith("<action>")
    if is_action != starts_with_action:
        print(
            f"[WARN] line {line_no}: images={len(sample.get('images', []))}인데 "
            f"assistant는 {'<action>' if starts_with_action else 'state HTML'}로 시작 "
            f"→ action(uniform 1.0)으로 처리"
        )
        is_action = True

    if is_action:
        asst_ids = tokenizer(asst, add_special_tokens=False)["input_ids"]
        sample["token_weights"] = [1.0] * len(asst_ids)
        sample["_diff_counts"] = {}
        return sample, "action"

    # current_html 필드가 없으면 user 메시지에서 추출
    if not current_html:
        current_html = _extract_current_html(user)

    # future_html이 없으면 assistant content로 대체
    if not future_html:
        future_html = asst

    def _uniform() -> list[float]:
        asst_ids = tokenizer(asst, add_special_tokens=False)["input_ids"]
        return [1.0] * len(asst_ids)

    # ── diff 분류 ─────────────────────────────────────────────────────────
    try:
        diff_result = _hd.classify_diff(current_html, future_html)
    except Exception as e:
        if on_error == "fail":
            raise SampleFailure("diff", line_no, e) from e
        print(f"[WARN] line {line_no}: diff 실패 ({e}) → uniform fallback")
        sample["token_weights"] = _uniform()
        sample["_diff_counts"] = {}
        return sample, "diff_fail"

    diff_counts = _hd.summarize_diff(diff_result)

    # ── 유도성 분류 (v3 전용) ────────────────────────────────────────────
    # v1/v2 모듈에는 classify_derivability 가 없다 → deriv_result=None 으로 1축 거동.
    # 실패해도 학습 데이터 생성을 멈추지 않는다: 유도성은 **가중치를 깎는** 축이라
    # 없으면 v2 와 같은 full weight 로 되돌아갈 뿐이고, 그건 안전한 방향이다.
    deriv_result = None
    deriv_counts: dict[str, int] = {}
    if hasattr(_hd, "classify_derivability"):
        try:
            action = _hd.extract_action(user)
            deriv_result = _hd.classify_derivability(current_html, future_html, action)
            deriv_counts = _hd.summarize_derivability(deriv_result)
        except Exception as e:
            if on_error == "fail":
                raise SampleFailure("derivability", line_no, e) from e
            print(f"[WARN] line {line_no}: 유도성 분류 실패 ({e}) → 유도성 축 없이 진행")
            deriv_result = None

    # ── token weight 생성 ─────────────────────────────────────────────────
    # ★ assistant 부분의 weight만 저장한다.
    #    prefix 길이는 훈련 시 이미지 토큰 확장 등으로 달라지므로,
    #    collator에서 labels(-100 마스크)를 기반으로 prefix/assistant 경계를 판단한다.
    prefix_text = prefix_fn(system, user)
    try:
        kwargs = {}
        if deriv_result is not None:
            kwargs["deriv_result"] = deriv_result
        weights = _tw.build_token_weights(
            tokenizer=tokenizer,
            system=system,
            user=user,
            future_html=future_html,
            diff_result=diff_result,
            prefix_text=prefix_text,
            weight_map=weight_map,
            **kwargs,
        )
        # prefix 부분(0.0)을 제거하고 assistant 부분만 저장
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        n_prefix = len(prefix_ids)
        weights = weights[n_prefix:]
    except Exception as e:
        if on_error == "fail":
            raise SampleFailure("weight", line_no, e) from e
        print(f"[WARN] line {line_no}: weight 생성 실패 ({e}) → uniform fallback")
        sample["token_weights"] = _uniform()
        sample["_diff_counts"] = diff_counts
        if deriv_counts:
            sample["_deriv_counts"] = deriv_counts
        return sample, "weight_fail"

    sample["token_weights"] = weights
    sample["_diff_counts"] = diff_counts  # 훈련 전 제거하거나 그대로 둬도 무방
    if deriv_counts:
        sample["_deriv_counts"] = deriv_counts  # 유도성 분포 진단용 (v3 만 채워진다)

    return sample, "ok"


# ── 메인 파이프라인 ───────────────────────────────────────────────────────


def _resolve_revision(model_name: str, revision: str | None) -> str:
    """HF 캐시에서 실제 사용된 commit SHA 를 best-effort 로 해석 (실패 시 'unknown')."""
    try:
        from huggingface_hub import HfApi  # noqa: F401  (존재 확인용)
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_dir = Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}"
        ref = repo_dir / "refs" / (revision or "main")
        if ref.is_file():
            return ref.read_text().strip()
        # revision 이 이미 commit SHA 인 경우
        if revision and (repo_dir / "snapshots" / revision).is_dir():
            return revision
    except Exception:  # pragma: no cover — 메타데이터용이라 실패해도 진행
        pass
    return "unknown"


def preprocess(
    input_jsonl: str,
    output_jsonl: str,
    model_name: str,
    template_key: str | None = None,
    w_added: float = 1.0,
    w_modified: float = 1.0,
    w_unchanged: float = 0.25,
    metric_version: str = "v2",
    revision: str | None = None,
    on_error: str = "fail",
    w_non_derivable: float = 0.5,
) -> dict:
    """
    input_jsonl 전체를 순회하며 token_weights를 계산하고 output_jsonl에 저장.

    출력은 sibling temp 파일에 쓴 뒤 **원자 교체**한다 (부분 산출물 방지).
    `<output>.meta.json` sidecar 에 재현에 필요한 메타데이터를 기록한다.

    Args:
        input_jsonl  : 원본 데이터 경로
        output_jsonl : 출력 데이터 경로 (input 과 같으면 거부)
        model_name   : HuggingFace 모델명 (tokenizer 로드에 사용)
        template_key : "qwen" | "llama3" | "default" | None(자동 감지)
        w_added      : ADDED element 가중치
        w_modified   : MODIFIED element 가중치
        w_unchanged  : UNCHANGED element 가중치
        revision     : tokenizer commit SHA / 태그 고정 (None 이면 캐시 기본)
        on_error     : "fail" | "uniform" | "skip"
        w_non_derivable : (v3 전용) 유도 불가능한 ADDED/MODIFIED 의 감쇠 가중치.
                          v1/v2 에서는 무시된다 (유도성 축이 없다).

    Returns:
        집계 dict (sidecar 에 기록되는 것과 동일)
    """
    in_p = Path(input_jsonl).resolve()
    out_p = Path(output_jsonl).resolve()
    if in_p == out_p:
        raise ValueError(
            f"--input 과 --output 이 같은 경로입니다 ({in_p}). "
            "in-place 쓰기는 입력을 truncate 하므로 금지한다."
        )

    _load_metric(metric_version)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    resolved_rev = _resolve_revision(model_name, revision)

    tkey = template_key or detect_template(model_name)
    prefix_fn = TEMPLATE_MAP.get(tkey, TEMPLATE_MAP["default"])
    # v3 빌더는 (diff_type × derivability) 2축 표를 요구한다. 표 생성은 빌더가 갖고
    # 있으므로 여기서 조합을 하드코딩하지 않는다 (하드 제약 10 — baseline 은 표에서 유도).
    if hasattr(_tw, "make_weight_map"):
        weight_map = _tw.make_weight_map(
            w_added, w_modified, w_unchanged, w_non_derivable
        )
        wdesc = (
            f"ADDED={w_added} MODIFIED={w_modified} UNCHANGED={w_unchanged} "
            f"NON_DERIVABLE(add/mod)={w_non_derivable}"
        )
    else:
        weight_map = {
            "ADDED": w_added,
            "MODIFIED": w_modified,
            "UNCHANGED": w_unchanged,
        }
        wdesc = f"ADDED={w_added} MODIFIED={w_modified} UNCHANGED={w_unchanged}"

    print(
        f"[INFO] metric={metric_version} | 템플릿: {tkey} | 가중치 {wdesc} | "
        f"on-error={on_error} | tokenizer={model_name}@{resolved_rev[:12]}"
    )

    counts = {
        "total": 0,
        "ok": 0,
        "action": 0,
        "diff_fail": 0,
        "weight_fail": 0,
        "json_error": 0,
        "skipped": 0,
        "written": 0,
    }
    agg_diff = {"ADDED": 0, "MODIFIED": 0, "UNCHANGED": 0}
    agg_deriv: dict[str, int] = {}

    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_name(out_p.name + ".tmp")

    try:
        with (
            in_p.open(encoding="utf-8") as fin,
            tmp_p.open("w", encoding="utf-8") as fout,
        ):
            for line_no, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                counts["total"] += 1

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    if on_error == "fail":
                        raise SampleFailure("json", line_no, e) from e
                    print(f"[ERROR] line {line_no}: JSON 파싱 실패 ({e}) → 스킵")
                    counts["json_error"] += 1
                    counts["skipped"] += 1
                    continue

                sample, status = process_sample(
                    sample, tokenizer, prefix_fn, weight_map, line_no, on_error
                )
                counts[status] += 1

                if on_error == "skip" and status in ("diff_fail", "weight_fail"):
                    counts["skipped"] += 1
                    continue

                for k, v in sample["_diff_counts"].items():
                    agg_diff[k] += v
                for k, v in sample.get("_deriv_counts", {}).items():
                    agg_deriv[k] = agg_deriv.get(k, 0) + v

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts["written"] += 1

                if line_no % 500 == 0:
                    dc = sample["_diff_counts"]
                    wl = len(sample["token_weights"])
                    print(f"  [{line_no:>6}] diff={dc}  weights_len={wl}")

        # ── 모든 처리가 성공한 뒤에만 원자 교체 ───────────────────────────
        os.replace(tmp_p, out_p)
    except BaseException:
        tmp_p.unlink(missing_ok=True)  # 부분 산출물 남기지 않는다
        raise

    meta = {
        "input": str(in_p),
        "output": str(out_p),
        "model": model_name,
        "revision_arg": revision,
        "revision_resolved": resolved_rev,
        "tokenizer_class": tokenizer.__class__.__name__,
        "template": tkey,
        "metric_version": metric_version,
        # 2축 표의 키는 튜플이라 JSON 으로 직렬화되지 않는다 → "DIFF|DERIV" 로 편다.
        "weight_map": {
            (f"{k[0]}|{k[1]}" if isinstance(k, tuple) else k): v
            for k, v in weight_map.items()
        },
        "on_error": on_error,
        "counts": counts,
        "diff_totals": agg_diff,
        "derivability_totals": agg_deriv,
    }
    Path(str(out_p) + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    fallback = counts["diff_fail"] + counts["weight_fail"]
    print(
        f"\n[완료] 총 {counts['total']}건 | 정상(diff) {counts['ok']} | "
        f"action(uniform) {counts['action']} | fallback {fallback} "
        f"(diff {counts['diff_fail']} / weight {counts['weight_fail']}) | "
        f"json오류 {counts['json_error']} | 스킵 {counts['skipped']} | 출력 {counts['written']}"
    )
    print(f"[diff 집계] {agg_diff}")
    if agg_deriv:
        print(f"[유도성 집계] {agg_deriv}")
    if fallback:
        print(
            f"[주의] {fallback}건이 uniform fallback 이다 — diff 강조가 적용되지 않았다. "
            "성공으로 집계하지 않았다."
        )
    return meta


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT 훈련용 token_weights 전처리")
    parser.add_argument("--input", required=True, help="원본 JSONL 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument("--model", required=True, help="HuggingFace 모델명")
    parser.add_argument("--template", default=None, help="qwen | llama3 | default")
    parser.add_argument("--w-added", type=float, default=1.0)
    parser.add_argument("--w-modified", type=float, default=1.0)
    parser.add_argument("--w-unchanged", type=float, default=0.25)
    parser.add_argument(
        "--w-non-derivable",
        type=float,
        default=0.5,
        help=(
            "v3 전용: 유도 불가능한(현재 state + action 으로 내용이 결정되지 않는) "
            "ADDED/MODIFIED 요소의 감쇠 가중치. v1/v2 에서는 무시된다."
        ),
    )
    parser.add_argument(
        "--metric-version",
        choices=["v1", "v2", "v3"],
        default="v2",
        help=(
            "v1: 원본 로직 / v2: _collect_texts·_match_cost 개선판 / "
            "v3: element 화이트리스트 제거 + 액션 유도성 2축 가중치"
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="tokenizer commit SHA/태그 고정 (미지정 시 캐시 기본). 산출물 sidecar 에 기록된다.",
    )
    parser.add_argument(
        "--on-error",
        choices=["fail", "uniform", "skip"],
        default="fail",
        help=(
            "diff/weight 생성 실패 처리. fail(기본)=중단, "
            "uniform=균일 1.0 폴백(성공으로 집계 안 함), skip=해당 레코드 제외"
        ),
    )
    args = parser.parse_args()

    preprocess(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model_name=args.model,
        template_key=args.template,
        w_added=args.w_added,
        w_modified=args.w_modified,
        w_unchanged=args.w_unchanged,
        metric_version=args.metric_version,
        revision=args.revision,
        on_error=args.on_error,
        w_non_derivable=args.w_non_derivable,
    )
