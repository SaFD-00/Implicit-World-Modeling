#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_wm_formats.py
======================
build_wm_formats.py 가 만든 두 파일(_split_raw.jsonl / _split_applied.jsonl)이
불변식을 지키는지 검사한다. 학습을 돌리기 전에 반드시 통과시킬 것.

검사 항목
---------
C1  두 파일의 길이·순서·sample_id·content_hash 가 1:1 대응
C2  타깃(assistant 턴)이 두 파일에서 바이트 동일        <- diff 조인의 전제
C3  포맷 비율이 설정값과 일치
C4  human 턴 레이아웃: image_first 설정대로 배치, 인라인 마커 존재
C5  system 턴: # Observability 가 # Output Format 앞에 존재,
    # Output Format 이후 구간은 세 포맷에서 완전히 동일
C6  dropped 포맷에 XML 잔존 없음
C7  masked 포맷의 XML 이 well-formed
C8  masked_elements 의 src_span 이 raw 원본 XML 기준으로 정확
C9  같은 요소의 inner text 만 가리고 aria-label 을 남긴 누수가 없음
C10 타깃에 <MASK/> / [MASK] 유출 없음
C11 마스킹 목표 비율 대비 실제 달성률

사용법:
  python validate_wm_formats.py --raw out/x_split_raw.jsonl --applied out/x_split_applied.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import Counter
from typing import List, Tuple

MASK_ELEM = "<MASK/>"
MASK_ATTR = "[MASK]"
HDR_UI = "Current UI State"
HDR_SHOT = "[Screenshot]"
HDR_ACTION = "Action:"
ANCHOR = "# Output Format (STRICT)"

TAG_RE = re.compile(r'<\s*(/?)\s*([A-Za-z][\w:-]*)((?:"[^"]*"|[^<>"])*?)(/?)\s*>')


def load(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def raw_xml(human: str) -> str:
    i = human.find("Current UI State:")
    j = human.find(HDR_SHOT)
    return human[i + len("Current UI State:"):j].strip()


def applied_xml(human: str) -> str:
    """applied 쪽 human 턴에서 XML 본문만 추출."""
    m = re.search(r"^Current UI State [^\n]*:\n", human, re.M)
    if not m:
        return ""
    start = m.end()
    nxt = human.find("\n\n", start)
    body = human[start:nxt if nxt > 0 else len(human)]
    return body.strip()


def wellformed(xml: str) -> bool:
    stack: List[str] = []
    for m in TAG_RE.finditer(xml):
        closing, tag, _, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)
        if closing:
            if not stack or stack.pop() != tag:
                return False
        elif not selfclose:
            stack.append(tag)
    return not stack


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--applied", required=True)
    ap.add_argument("--expect-full", type=float, default=0.25)
    ap.add_argument("--expect-masked", type=float, default=0.55)
    ap.add_argument("--expect-dropped", type=float, default=0.20)
    ap.add_argument("--tol", type=float, default=0.02, help="포맷 비율 허용 오차")
    a = ap.parse_args()

    raw, app = load(a.raw), load(a.applied)
    fails: List[str] = []
    warns: List[str] = []

    def check(cond: bool, code: str, msg: str) -> None:
        (fails if not cond else warns if False else fails if not cond else warns)
    # 위 헬퍼는 쓰지 않고 명시적으로 기록한다.

    # ---- C1 ----
    ok = (len(raw) == len(app)
          and all(r["sample_id"] == p["sample_id"] and r["content_hash"] == p["content_hash"]
                  for r, p in zip(raw, app)))
    (warns if ok else fails).append(f"C1  1:1 대응 ({len(raw)} vs {len(app)}): {'OK' if ok else 'FAIL'}")

    # ---- C2 ----
    bad = [r["sample_id"] for r, p in zip(raw, app)
           if r["messages"][2]["value"] != p["messages"][2]["value"]]
    (warns if not bad else fails).append(
        f"C2  타깃 동일성: {'OK' if not bad else f'FAIL ({len(bad)}건, 예: {bad[:5]})'}")

    # ---- C3 ----
    c = Counter(p["fmt"] for p in app)
    n = len(app)
    exp = {"full": a.expect_full, "masked": a.expect_masked, "dropped": a.expect_dropped}
    off = {k: abs(c[k] / n - v) for k, v in exp.items()}
    ok = all(v <= a.tol for v in off.values())
    (warns if ok else fails).append(
        f"C3  포맷 비율 {{'full':{c['full']/n:.3f},'masked':{c['masked']/n:.3f},"
        f"'dropped':{c['dropped']/n:.3f}}}: {'OK' if ok else 'FAIL'}")

    # ---- C4 ----
    image_first = app[0].get("meta", {}).get("image_first", True)
    bad4 = []
    for p in app:
        h = p["messages"][1]["value"]
        first_is_shot = h.startswith(HDR_SHOT)
        if image_first != first_is_shot:
            bad4.append(p["sample_id"])
        if not re.search(r"^Current UI State [^\n]*:$", h, re.M):
            bad4.append(p["sample_id"])
        if HDR_ACTION not in h:
            bad4.append(p["sample_id"])
    bad4 = sorted(set(bad4))
    (warns if not bad4 else fails).append(
        f"C4  human 레이아웃 (image_first={image_first}): "
        f"{'OK' if not bad4 else f'FAIL ({len(bad4)}건)'}")

    # ---- C5 ----
    tails = set()
    bad5 = []
    for p in app:
        s = p["messages"][0]["value"]
        if "# Observability" not in s or s.index("# Observability") > s.index(ANCHOR):
            bad5.append(p["sample_id"])
        tails.add(s.split(ANCHOR, 1)[1])
    ok = (not bad5) and len(tails) == 1
    (warns if ok else fails).append(
        f"C5  system 구조 (Output Format 이후 동일: {len(tails)}종): {'OK' if ok else 'FAIL'}")

    # ---- C6 ----
    bad6 = [p["sample_id"] for p in app
            if p["fmt"] == "dropped" and "data-bbox" in p["messages"][1]["value"]]
    (warns if not bad6 else fails).append(
        f"C6  dropped XML 제거: {'OK' if not bad6 else f'FAIL ({len(bad6)}건)'}")

    # ---- C7 ----
    bad7 = [p["sample_id"] for p in app
            if p["fmt"] == "masked" and not wellformed(applied_xml(p["messages"][1]["value"]))]
    (warns if not bad7 else fails).append(
        f"C7  masked XML well-formed: {'OK' if not bad7 else f'FAIL ({len(bad7)}건: {bad7[:5]})'}")

    # ---- C8 ----
    checked = bad8 = 0
    for r, p in zip(raw, app):
        if p["fmt"] != "masked":
            continue
        x = raw_xml(r["messages"][1]["value"])
        for rec in p["meta"]["masked_elements"]:
            s, e = rec["src_span"]
            frag = x[s:e]
            checked += 1
            if rec["kind"] == "whole" and not frag.startswith("<" + rec["tag"]):
                bad8 += 1
            elif rec["kind"] == "aria" and frag != rec["key"][:len(frag)]:
                bad8 += 1
    (warns if not bad8 else fails).append(
        f"C8  src_span 정합 ({checked}건 중 불일치 {bad8}): {'OK' if not bad8 else 'FAIL'}")

    # ---- C9 ----
    leak = 0
    for r, p in zip(raw, app):
        if p["fmt"] != "masked":
            continue
        recs = p["meta"]["masked_elements"]
        groups: dict = {}
        for rec in recs:
            groups.setdefault(rec["group"], []).append(rec["kind"])
        x = raw_xml(r["messages"][1]["value"])
        for rec in recs:
            if rec["kind"] != "text":
                continue
            s = rec["src_span"][0]
            open_tag = x[x.rfind("<", 0, s):s]
            if "aria-label=" in open_tag and "aria" not in groups[rec["group"]]:
                leak += 1
    (warns if not leak else fails).append(
        f"C9  text/aria 누수: {'OK' if not leak else f'FAIL ({leak}건)'}")

    # ---- C10 ----
    bad10 = [p["sample_id"] for p in app
             if MASK_ELEM in p["messages"][2]["value"] or MASK_ATTR in p["messages"][2]["value"]]
    (warns if not bad10 else fails).append(
        f"C10 타깃 마스크 유출: {'OK' if not bad10 else f'FAIL ({len(bad10)}건)'}")

    # ---- C11 ----
    ach = [p["meta"]["mask_ratio_actual"] / p["meta"]["mask_ratio_target"]
           for p in app if p["fmt"] == "masked" and p["meta"].get("mask_ratio_target")]
    if ach:
        med = st.median(ach)
        p10 = sorted(ach)[len(ach) // 10]
        ok = med >= 0.90
        (warns if ok else fails).append(
            f"C11 마스킹 달성률 median {med:.3f} / p10 {p10:.3f}: {'OK' if ok else 'LOW'}")

    print("=" * 62)
    for line in warns:
        print("  " + line)
    hard = [l for l in fails if "FAIL" in l or "LOW" in l]
    for line in hard:
        print("  !! " + line)
    print("-" * 62)
    print("  RESULT:", "ALL PASS" if not hard else f"{len(hard)} CHECK(S) FAILED")
    print("=" * 62)
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
