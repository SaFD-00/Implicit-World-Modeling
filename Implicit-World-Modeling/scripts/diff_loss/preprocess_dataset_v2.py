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
from pathlib import Path

from token_weight_builder_v2 import build_token_weights
from transformers import AutoTokenizer

# 메트릭 모듈은 CLI 옵션에 따라 런타임에 로드 (v1: hungarian_diff / v2: hungarian_diff_v2)
_hd = None  # hungarian_diff 모듈 참조 (classify_diff, summarize_diff)


def _load_metric(version: str) -> None:
    global _hd
    if version == "v2":
        _hd = importlib.import_module("hungarian_diff_v2")
    else:
        _hd = importlib.import_module("hungarian_diff")


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
) -> dict:
    """
    샘플 하나를 받아 token_weights를 계산하여 추가한 뒤 반환.
    current_html / future_html 필드는 제거됨.
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
    # images 2개 = action_pred (assistant가 <action>{...}</action>), 1개 = state_pred.
    # action 출력에는 diff 개념이 없으므로 균일 1.0을 부여한다.
    # diff 경로로 흘려보내면 "diff 없음 → 전부 baseline(0.25)"으로 잘못 감쇠된다.
    is_action = len(sample.get("images", [])) == 2
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
        return sample

    # current_html 필드가 없으면 user 메시지에서 추출
    if not current_html:
        current_html = _extract_current_html(user)

    # future_html이 없으면 assistant content로 대체
    if not future_html:
        future_html = asst

    # ── diff 분류 ─────────────────────────────────────────────────────────
    try:
        diff_result = _hd.classify_diff(current_html, future_html)
    except Exception as e:
        print(f"[WARN] line {line_no}: diff 실패 ({e}), 균일 가중치 사용")
        diff_result = []

    diff_counts = _hd.summarize_diff(diff_result)

    # ── token weight 생성 ─────────────────────────────────────────────────
    # ★ assistant 부분의 weight만 저장한다.
    #    prefix 길이는 훈련 시 이미지 토큰 확장 등으로 달라지므로,
    #    collator에서 labels(-100 마스크)를 기반으로 prefix/assistant 경계를 판단한다.
    prefix_text = prefix_fn(system, user)
    try:
        weights = build_token_weights(
            tokenizer=tokenizer,
            system=system,
            user=user,
            future_html=future_html,
            diff_result=diff_result,
            prefix_text=prefix_text,
            weight_map=weight_map,
        )
        # prefix 부분(0.0)을 제거하고 assistant 부분만 저장
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        n_prefix = len(prefix_ids)
        weights = weights[n_prefix:]
    except Exception as e:
        print(f"[WARN] line {line_no}: weight 생성 실패 ({e}), 균일 가중치 사용")
        asst_ids = tokenizer(asst, add_special_tokens=False)["input_ids"]
        weights = [1.0] * len(asst_ids)

    sample["token_weights"] = weights
    sample["_diff_counts"] = diff_counts  # 훈련 전 제거하거나 그대로 둬도 무방

    return sample


# ── 메인 파이프라인 ───────────────────────────────────────────────────────


def preprocess(
    input_jsonl: str,
    output_jsonl: str,
    model_name: str,
    template_key: str | None = None,
    w_added: float = 1.0,
    w_modified: float = 1.0,
    w_unchanged: float = 0.25,
    metric_version: str = "v2",
) -> None:
    """
    input_jsonl 전체를 순회하며 token_weights를 계산하고 output_jsonl에 저장.

    Args:
        input_jsonl  : 원본 데이터 경로
        output_jsonl : 출력 데이터 경로
        model_name   : HuggingFace 모델명 (tokenizer 로드에 사용)
        template_key : "qwen" | "llama3" | "default" | None(자동 감지)
        w_added      : ADDED element 가중치
        w_modified   : MODIFIED element 가중치
        w_unchanged  : UNCHANGED element 가중치 (보통 1.0)
    """
    _load_metric(metric_version)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tkey = template_key or detect_template(model_name)
    prefix_fn = TEMPLATE_MAP.get(tkey, TEMPLATE_MAP["default"])
    print(
        f"[INFO] metric={metric_version} | 템플릿: {tkey} | 가중치 ADDED={w_added} MODIFIED={w_modified} UNCHANGED={w_unchanged}"
    )

    weight_map = {
        "ADDED": w_added,
        "MODIFIED": w_modified,
        "UNCHANGED": w_unchanged,
    }

    total = ok = warn = 0
    agg_diff = {"ADDED": 0, "MODIFIED": 0, "UNCHANGED": 0}

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_jsonl, encoding="utf-8") as fin,
        open(output_jsonl, "w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] line {line_no}: JSON 파싱 실패 ({e}), 스킵")
                warn += 1
                continue

            sample = process_sample(sample, tokenizer, prefix_fn, weight_map, line_no)

            # 집계
            for k, v in sample["_diff_counts"].items():
                agg_diff[k] += v
            ok += 1

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if line_no % 500 == 0:
                dc = sample["_diff_counts"]
                wl = len(sample["token_weights"])
                print(f"  [{line_no:>6}] diff={dc}  weights_len={wl}")

    print(f"\n[완료] 총 {total}건 처리 | 성공 {ok}건 | 실패 {warn}건")
    print(f"[diff 집계] {agg_diff}")


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
        "--metric-version",
        choices=["v1", "v2"],
        default="v2",
        help="v1: 원본 로직 / v2: _collect_texts·_match_cost 개선판",
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
    )
