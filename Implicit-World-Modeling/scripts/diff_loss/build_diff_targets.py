"""
build_diff_targets.py
────────────────────────────────────────────────────────────────
build_wm_formats.py 가 만든 두 파일(*_split_raw.jsonl / *_split_applied.jsonl)을
`sample_id` 로 조인해서 diff-loss 학습용 `token_weights` 필드를 붙인 최종
학습 JSONL 을 만든다.

핵심 아이디어
──────────────
- diff 계산은 raw 의 원본 XML 로 해야 정확하다. masked/dropped 의 current XML
  로 계산하면 요소들이 사라진 채로 매칭되어 UNCHANGED 가 ADDED 로 오분류된다.
- 하지만 학습은 applied (masked/dropped 포함) 로 해야 한다 — 그게 anti-copy
  훈련의 요점이다.
- 다행히 assistant 턴(next XML) 은 두 파일에서 바이트 동일하므로(C2 불변식),
  `token_weight_builder_v2.build_token_weights()` 가 `future_html` 을 standalone
  으로 토크나이즈해 만든 assistant 토큰 배열은 raw/applied 어느 쪽에서 계산해도
  동일하다. 그래서 raw 로 계산한 토큰 가중치를 applied 메시지에 그대로 붙일 수
  있다.
- LlamaFactory 의 diff-loss 패치(collator/converter/supervised/trainer_utils)가
  이미 샘플의 `token_weights: list[float]` 필드를 읽어 labels(-100 마스크) 경계로
  배치 텐서에 정렬한다. 이 스크립트는 그 필드를 채워 넣는 역할이다.

가중치 정책 (2단 v2)
────────────────────
    ADDED / MODIFIED → 1.0     (학습 신호 집중)
    UNCHANGED        → 0.25    (baseline, span 밖 토큰도 동일)

사용법:
  python build_diff_targets.py \
      --raw     out/data_split_raw.jsonl \
      --applied out/data_split_applied.jsonl \
      --output  out/data_split_weighted.jsonl \
      --model   Qwen/Qwen2.5-VL-7B-Instruct \
      --w-added 1.0 --w-modified 1.0 --w-unchanged 0.25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# 같은 디렉터리의 v2 모듈을 import (cwd 와 무관하게 동작하도록)
sys.path.insert(0, str(Path(__file__).parent))

import hungarian_diff_v2c as hungarian_diff_v2  # noqa: E402
import token_weight_builder_v2c as token_weight_builder_v2  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


# ── 상수 ─────────────────────────────────────────────────────────────────────
HDR_UI = "Current UI State:"
HDR_SHOT = "[Screenshot]"

# sharegpt(from/value) → openai(role/content) 통일용
_ROLE_MAP = {"human": "user", "gpt": "assistant"}


class SampleFailure(Exception):
    """diff / weight 생성 실패. --on-error fail 일 때 실행을 중단시킨다."""

    def __init__(self, stage: str, sample_id: int, cause: Exception):
        super().__init__(f"sample_id {sample_id}: {stage} 실패 ({cause})")
        self.stage = stage
        self.sample_id = sample_id
        self.cause = cause


# ── 유틸 ─────────────────────────────────────────────────────────────────────


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """sharegpt(from/value) → openai(role/content). 이미 role/content 이면 그대로."""
    if messages and "role" in messages[0]:
        return messages
    return [
        {"role": _ROLE_MAP.get(m["from"], m["from"]), "content": m["value"]}
        for m in messages
    ]


def _extract_current_xml_from_raw_user(user_content: str) -> str:
    """raw 파일의 user 메시지에서 원본 current XML 을 뽑는다.

    build_wm_formats.py 의 raw 는 다음 레이아웃을 가진다:
      "Current UI State:\n<xml>\n\n[Screenshot]\n<image>\n\nAction:\n..."
    """
    i = user_content.find(HDR_UI)
    j = user_content.find(HDR_SHOT)
    if i < 0 or j < 0:
        raise ValueError(
            "raw user 레이아웃 이상: 'Current UI State:' / '[Screenshot]' 헤더 없음"
        )
    return user_content[i + len(HDR_UI) : j].strip()


# ── 모델별 chat template prefix (preprocess_dataset_v2 와 동일 규약) ─────────
# assistant 부분 가중치는 prefix 내용에 의존하지 않지만, build_token_weights 가
# 요구하는 인자라서 정확한 chat template 을 만들어 준다.


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


# ── 단일 샘플 처리 ────────────────────────────────────────────────────────────


def process_pair(
    raw_sample: dict,
    applied_sample: dict,
    tokenizer,
    prefix_fn,
    weight_map: dict[str, float],
    on_error: str,
) -> tuple[dict, str]:
    """raw_sample 로 diff 를 계산하고 applied_sample 에 token_weights 를 붙인다.

    Returns:
      (output_sample, status)
      status: "ok" | "action" | "diff_fail" | "weight_fail"

    on_error:
      "fail"    — SampleFailure 를 올려 실행 중단
      "uniform" — 균일 1.0 로 폴백 (성공으로 집계하지 않음)
      "skip"    — 실패 레코드는 호출자가 출력에서 제외
    """
    sid = raw_sample["sample_id"]

    # ── 정합성 검증 ───────────────────────────────────────
    if raw_sample["sample_id"] != applied_sample["sample_id"]:
        raise SampleFailure(
            "join", sid, ValueError("sample_id mismatch between raw and applied")
        )
    if raw_sample["content_hash"] != applied_sample["content_hash"]:
        raise SampleFailure(
            "join",
            sid,
            ValueError("content_hash mismatch (assistant 가 raw/applied 에서 다름)"),
        )

    # ── raw 메시지 정규화 ─────────────────────────────────
    raw_msgs = _normalize_messages(raw_sample["messages"])
    system = next((m["content"] for m in raw_msgs if m["role"] == "system"), "")
    user = next((m["content"] for m in raw_msgs if m["role"] == "user"), "")
    asst = next((m["content"] for m in raw_msgs if m["role"] == "assistant"), "")

    # applied 를 얕은 복사해 output 으로 (messages 는 applied 것을 유지)
    output = dict(applied_sample)

    # ── action 샘플 분기 ─────────────────────────────────
    # NEXT_STATE_PREDICTION 데이터는 이미지 1장 규약. 그 외는 action_pred 로 취급하고
    # 균일 1.0. build_wm_formats 는 state_pred 전용이지만 안전망으로 둔다.
    is_action = len(raw_sample.get("images", [])) != 1
    if not is_action and asst.lstrip().startswith("<action>"):
        is_action = True

    if is_action:
        asst_ids = tokenizer(asst, add_special_tokens=False)["input_ids"]
        output["token_weights"] = [1.0] * len(asst_ids)
        output["_diff_counts"] = {}
        return output, "action"

    # ── raw 에서 원본 current XML 뽑기 ───────────────────
    try:
        current_html = _extract_current_xml_from_raw_user(user)
    except ValueError as e:
        if on_error == "fail":
            raise SampleFailure("extract", sid, e) from e
        current_html = ""

    future_html = asst

    def _uniform() -> list[float]:
        asst_ids = tokenizer(asst, add_special_tokens=False)["input_ids"]
        return [1.0] * len(asst_ids)

    # ── Hungarian diff (raw 로) ──────────────────────────
    try:
        diff_result = hungarian_diff_v2.classify_diff(current_html, future_html)
    except Exception as e:
        if on_error == "fail":
            raise SampleFailure("diff", sid, e) from e
        print(f"[WARN] sample_id {sid}: diff 실패 ({e}) → uniform fallback")
        output["token_weights"] = _uniform()
        output["_diff_counts"] = {}
        return output, "diff_fail"

    diff_counts = hungarian_diff_v2.summarize_diff(diff_result)

    # ── token_weights 생성 (assistant 부분만) ────────────
    # future_html 은 raw/applied 에서 바이트 동일 (C2) → 이 배열은 그대로 applied 에
    # 이식 가능하다. prefix_text 는 build_token_weights 가 prefix 길이를 재기 위해
    # 필요한 것이라 raw 것을 준다 (결과의 assistant 부분에는 영향 없음).
    prefix_text = prefix_fn(system, user)
    try:
        weights = token_weight_builder_v2.build_token_weights(
            tokenizer=tokenizer,
            system=system,
            user=user,
            future_html=future_html,
            diff_result=diff_result,
            prefix_text=prefix_text,
            weight_map=weight_map,
        )
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        weights = weights[len(prefix_ids):]
    except Exception as e:
        if on_error == "fail":
            raise SampleFailure("weight", sid, e) from e
        print(f"[WARN] sample_id {sid}: weight 실패 ({e}) → uniform fallback")
        output["token_weights"] = _uniform()
        output["_diff_counts"] = diff_counts
        return output, "weight_fail"

    output["token_weights"] = weights
    output["_diff_counts"] = diff_counts
    return output, "ok"


# ── 메인 파이프라인 ──────────────────────────────────────────────────────────


def build(
    raw_path: str,
    applied_path: str,
    output_path: str,
    model_name: str,
    template_key: str | None,
    w_added: float,
    w_modified: float,
    w_unchanged: float,
    on_error: str,
    revision: str | None = None,
) -> dict:
    raw = _load_jsonl(raw_path)
    applied = _load_jsonl(applied_path)

    # ── 1:1 대응 사전 검증 (per-sample content_hash 는 process_pair 에서) ─
    if len(raw) != len(applied):
        raise ValueError(
            f"raw ({len(raw)}) 와 applied ({len(applied)}) 의 샘플 수가 다르다"
        )
    raw_ids = {s["sample_id"] for s in raw}
    app_ids = {s["sample_id"] for s in applied}
    if raw_ids != app_ids:
        missing_in_app = raw_ids - app_ids
        missing_in_raw = app_ids - raw_ids
        raise ValueError(
            f"sample_id 집합이 다르다: "
            f"applied 에 없음 {sorted(missing_in_app)[:5]}, "
            f"raw 에 없음 {sorted(missing_in_raw)[:5]}"
        )

    raw_by_sid = {s["sample_id"]: s for s in raw}

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tkey = template_key or detect_template(model_name)
    prefix_fn = TEMPLATE_MAP.get(tkey, TEMPLATE_MAP["default"])
    weight_map = {
        "ADDED": w_added,
        "MODIFIED": w_modified,
        "UNCHANGED": w_unchanged,
    }

    print(
        f"[INFO] model={model_name} template={tkey} "
        f"weights={weight_map} on_error={on_error}"
    )
    print(f"[INFO] raw={raw_path} applied={applied_path} → 샘플 {len(raw)}건")

    counts: Counter = Counter()
    agg_diff: Counter = Counter()
    fmt_count: Counter = Counter()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as fout:
            for i, applied_sample in enumerate(applied):
                sid = applied_sample["sample_id"]
                raw_sample = raw_by_sid[sid]

                counts["total"] += 1
                fmt_count[applied_sample.get("fmt", "unknown")] += 1

                output, status = process_pair(
                    raw_sample,
                    applied_sample,
                    tokenizer,
                    prefix_fn,
                    weight_map,
                    on_error,
                )
                counts[status] += 1

                if on_error == "skip" and status in ("diff_fail", "weight_fail"):
                    counts["skipped"] += 1
                    continue

                for k, v in output.get("_diff_counts", {}).items():
                    agg_diff[k] += v

                fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                counts["written"] += 1

                if i % 500 == 0:
                    dc = output.get("_diff_counts", {})
                    wl = len(output["token_weights"])
                    print(
                        f"  [{i:>6}] sid={sid} fmt={applied_sample.get('fmt','?'):>7} "
                        f"diff={dict(dc)} weights_len={wl}"
                    )

        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    meta = {
        "raw_input": str(raw_path),
        "applied_input": str(applied_path),
        "output": str(out_path),
        "model": model_name,
        "template": tkey,
        "metric_version": "v2",
        "weight_map": weight_map,
        "on_error": on_error,
        "counts": dict(counts),
        "fmt_counts": dict(fmt_count),
        "diff_totals": dict(agg_diff),
    }
    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )

    fallback = counts.get("diff_fail", 0) + counts.get("weight_fail", 0)
    print(
        f"\n[완료] 총 {counts['total']}건 | ok {counts.get('ok', 0)} "
        f"| action {counts.get('action', 0)} "
        f"| fallback {fallback} "
        f"(diff {counts.get('diff_fail', 0)} / weight {counts.get('weight_fail', 0)}) "
        f"| 출력 {counts['written']}"
    )
    print(f"[포맷 분포] {dict(fmt_count)}")
    print(f"[diff 집계] {dict(agg_diff)}")
    if fallback:
        print(
            f"[주의] {fallback}건이 uniform fallback 이다 — diff 강조가 적용되지 않았다."
        )
    return meta


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--raw", required=True, help="build_wm_formats 의 *_split_raw.jsonl")
    ap.add_argument(
        "--applied", required=True, help="build_wm_formats 의 *_split_applied.jsonl"
    )
    ap.add_argument(
        "--output", required=True, help="출력 JSONL (applied + token_weights)"
    )
    ap.add_argument("--model", required=True, help="HuggingFace 모델명 (토크나이저용)")
    ap.add_argument(
        "--template",
        default=None,
        help="qwen | llama3 | default (미지정 시 모델명으로 자동 감지)",
    )
    ap.add_argument("--w-added", type=float, default=1.0)
    ap.add_argument("--w-modified", type=float, default=1.0)
    ap.add_argument("--w-unchanged", type=float, default=0.25)
    ap.add_argument(
        "--revision",
        default=None,
        help="tokenizer commit SHA / 태그 고정 (미지정 시 캐시 기본)",
    )
    ap.add_argument(
        "--on-error",
        choices=["fail", "uniform", "skip"],
        default="fail",
        help="diff / weight 실패 처리 (기본 fail=중단)",
    )
    args = ap.parse_args()

    for a, b, label in [
        (args.raw, args.output, "--raw"),
        (args.applied, args.output, "--applied"),
    ]:
        if Path(a).resolve() == Path(b).resolve():
            ap.error(f"{label} 와 --output 이 같은 경로다 (덮어쓰기 방지)")

    build(
        raw_path=args.raw,
        applied_path=args.applied,
        output_path=args.output,
        model_name=args.model,
        template_key=args.template,
        w_added=args.w_added,
        w_modified=args.w_modified,
        w_unchanged=args.w_unchanged,
        on_error=args.on_error,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
