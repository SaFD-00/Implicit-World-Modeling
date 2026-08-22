"""build_diff_targets 산출물 검증 스크립트.

경로는 CLI 로 받는다 (원본 번들은 스모크 경로가 하드코딩돼 있었다).

  python scripts/diff_loss/inspect_weighted.py \
      --weighted out/x_split_weighted.jsonl \
      --applied  out/x_split_applied.jsonl \
      --raw      out/x_split_raw.jsonl \
      --model    Qwen/Qwen2.5-VL-3B-Instruct

`--weighted` 만 주면 나머지 두 경로는 `_split_weighted` 를 치환해서 추론한다.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--weighted", required=True, type=Path)
_ap.add_argument("--applied", type=Path, default=None)
_ap.add_argument("--raw", type=Path, default=None)
_ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
_args = _ap.parse_args()

WEIGHTED = _args.weighted
APPLIED = _args.applied or Path(str(WEIGHTED).replace("_split_weighted", "_split_applied"))
RAW = _args.raw or Path(str(WEIGHTED).replace("_split_weighted", "_split_raw"))

tok = AutoTokenizer.from_pretrained(_args.model)

# ── 1. 배열 길이 정합성 (전체) ────────────────────────────────
print("=" * 62)
print("[1] token_weights 길이 정합성 검사 (전수)")
print("-" * 62)
mismatch = 0
value_counter = Counter()
sid_by_fmt = {"full": [], "masked": [], "dropped": []}
diff_by_fmt = {"full": Counter(), "masked": Counter(), "dropped": Counter()}
weight_dist_by_fmt = {"full": Counter(), "masked": Counter(), "dropped": Counter()}

with WEIGHTED.open() as f:
    for line in f:
        rec = json.loads(line)
        asst = next(m["value"] for m in rec["messages"] if m["from"] == "gpt")
        expected = len(tok(asst, add_special_tokens=False)["input_ids"])
        actual = len(rec["token_weights"])
        if expected != actual:
            mismatch += 1
            if mismatch <= 3:
                print(f"  MISMATCH sid={rec['sample_id']} fmt={rec['fmt']}: expected {expected}, got {actual}")

        for w in rec["token_weights"]:
            value_counter[round(w, 3)] += 1

        fmt = rec["fmt"]
        sid_by_fmt[fmt].append(rec["sample_id"])
        for k, v in rec.get("_diff_counts", {}).items():
            diff_by_fmt[fmt][k] += v
        for w in rec["token_weights"]:
            weight_dist_by_fmt[fmt][round(w, 3)] += 1

print(f"  총 2000건 중 길이 불일치: {mismatch}건")
print(f"  가중치 값 분포: {dict(value_counter.most_common(10))}")

print()
print("=" * 62)
print("[2] fmt 별 diff/가중치 분포")
print("-" * 62)
for fmt in ("full", "masked", "dropped"):
    total = sum(weight_dist_by_fmt[fmt].values())
    dist = weight_dist_by_fmt[fmt]
    frac_10 = dist.get(1.0, 0) / total * 100 if total else 0
    frac_025 = dist.get(0.25, 0) / total * 100 if total else 0
    print(f"  {fmt:>7} ({len(sid_by_fmt[fmt])}건): diff={dict(diff_by_fmt[fmt])}")
    print(f"          가중치 분포 → 1.0: {frac_10:.1f}%   0.25: {frac_025:.1f}%   기타: {100-frac_10-frac_025:.2f}%")

print()
print("=" * 62)
print("[3] 실제 샘플 3개(fmt별) 자세히 보기")
print("-" * 62)

# fmt별 대표 샘플 1건씩
applied_by_sid = {}
with APPLIED.open() as f:
    for line in f:
        r = json.loads(line)
        applied_by_sid[r["sample_id"]] = r

raw_by_sid = {}
with RAW.open() as f:
    for line in f:
        r = json.loads(line)
        raw_by_sid[r["sample_id"]] = r

picked = {"full": None, "masked": None, "dropped": None}
with WEIGHTED.open() as f:
    for line in f:
        rec = json.loads(line)
        if picked[rec["fmt"]] is None:
            picked[rec["fmt"]] = rec
        if all(picked.values()):
            break

for fmt, rec in picked.items():
    sid = rec["sample_id"]
    print(f"\n  ────── fmt={fmt} sid={sid} ──────")
    print(f"  _diff_counts: {rec.get('_diff_counts')}")
    print(f"  token_weights: 길이={len(rec['token_weights'])}, "
          f"1.0={sum(1 for w in rec['token_weights'] if w == 1.0)}건, "
          f"0.25={sum(1 for w in rec['token_weights'] if w == 0.25)}건")

    # applied 의 user XML 첫 200자 (마스킹 확인용)
    hum = next(m["value"] for m in rec["messages"] if m["from"] == "human")
    ui_marker = "Current UI State"
    idx = hum.find(ui_marker)
    if idx >= 0:
        excerpt = hum[idx:idx + 300].replace("\n", " ")
        print(f"  applied user XML 발췌: {excerpt}")

    # raw 의 assistant 앞부분 + 해당 위치의 토큰별 가중치
    asst = next(m["value"] for m in rec["messages"] if m["from"] == "gpt")
    enc = tok(asst, add_special_tokens=False, return_offsets_mapping=True)
    weights = rec["token_weights"]
    print(f"  assistant 앞 15 토큰:")
    for i in range(min(15, len(enc["input_ids"]))):
        tid = enc["input_ids"][i]
        piece = tok.decode([tid])
        w = weights[i]
        print(f"    [{i:>3}] w={w:.2f}  '{piece}'")

    # 가중치 1.0 (=ADDED/MODIFIED) 토큰이 처음 나타나는 위치의 문맥
    for i, w in enumerate(weights):
        if w == 1.0:
            start = enc["offset_mapping"][i][0]
            end = enc["offset_mapping"][min(i+20, len(enc["offset_mapping"])-1)][1]
            print(f"  첫 가중치=1.0 위치 (토큰 idx {i}, char offset {start}~{end}):")
            print(f"    ...{asst[max(0,start-40):end+40]}...")
            break
    else:
        print("  ⚠️ 가중치 1.0 이 없음 (샘플이 모두 UNCHANGED)")
