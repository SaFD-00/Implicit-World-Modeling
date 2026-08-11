#!/usr/bin/env python3
"""산출된 copy_baseline_metrics.json 53개를 사후 검증하고 복사기 기준선 표를 만든다.

검증 1 — 복사기 항등식 (advisor 가 지적한 닫힌 형태)
    복사기는 `pred_els == cur_els` 이므로 `hit_gt` 를 만드는 매칭
    (`_hungarian_match(pred, gt)`) 과 `diff_gt` 분류를 만드는 매칭
    (`_hungarian_match(cur, gt)`) 이 **같은 호출**이다. 따라서 행마다 항등적으로
        modified_recall = 1.0 · unchanged_recall = 1.0 · added_recall = 0.0
    이고 addmod_recall = |MODIFIED| / (|MODIFIED| + |ADDED|) — 즉 **성능이 아니라
    test set 구성 통계**다. 어긋나면 solver 비결정성이나 추출 불일치다.

검증 2 — 그룹 상수성
    복사기 값은 (test jsonl, 필터) 조합의 함수다. 같은 조합의 leaf 끼리
    `copy_baseline` 섹션이 키별로 완전히 같아야 Notion 에 EXP 당 1행으로 접을 수
    있다. woa 는 행 부분집합이므로 **다른 그룹**이다 (값이 달라야 정상).
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["overall", "in_domain", "out_of_domain"]


def group_key(leaf: str) -> str:
    """(test jsonl, 필터) 를 결정하는 키. woa 는 별도 그룹이다."""
    p = Path(leaf)
    if "probe_forget" in p.parts:
        return "AC_EXP07_v1"  # probe_forget 은 EXP07 v1 test 를 쓴다 (woa 없음)
    base = p.name  # on-<DS>-state[-without-open_app]
    woa = base.endswith("-without-open_app")
    ds = base[len("on-"):].removesuffix("-without-open_app").removesuffix("-state")
    return f"{ds}{'+woa' if woa else ''}"


out = subprocess.run(
    ["bash", "-c", f"cd {ROOT} && find outputs -name copy_baseline_metrics.json -not -path '*_backup*' | sort"],
    capture_output=True, text=True, check=True)
files = [f for f in out.stdout.split() if f]
print(f"산출물 {len(files)} 개\n")

groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
viol = 0
for f in files:
    leaf = str(Path(f).parent)
    d = json.loads((ROOT / f).read_text())
    groups[group_key(leaf)].append((leaf, d))
    for sec in SECTIONS:
        c = d[sec]["copy_baseline"]
        for key, want in (("avg_modified_recall", 1.0), ("avg_unchanged_recall", 1.0),
                          ("avg_added_recall", 0.0)):
            n = c[f"n_{key[4:]}"]
            if n and c[key] != want:
                viol += 1
                print(f"[항등식 위반] {leaf} {sec} {key}={c[key]} (want {want}, n={n})")

print(f"검증 1 — 복사기 항등식: 위반 {viol} 건"
      f" ({len(files)} 파일 × 3 섹션 × 3 키 = {len(files) * 9} 검사)\n")

print("검증 2 — 그룹 상수성")
rows = []
for g, items in sorted(groups.items()):
    ref_leaf, ref = items[0]
    mismatch = []
    for leaf, d in items[1:]:
        for sec in SECTIONS:
            a, b = ref[sec]["copy_baseline"], d[sec]["copy_baseline"]
            bad = [k for k in a if a[k] != b.get(k)]
            if bad:
                mismatch.append((leaf, sec, bad[:4]))
    status = "동일" if not mismatch else f"불일치 {len(mismatch)}"
    print(f"  {g:24s} leaf {len(items):2d} → {status}   (기준 {ref_leaf})")
    for m in mismatch[:3]:
        print(f"      {m}")
    rows.append((g, ref_leaf, len(items), ref))

print("\n=== 복사기 기준선 (overall / ID / OOD) ===")
KEYS = [
    ("addmod_recall", "avg_addmod_recall"), ("added_recall", "avg_added_recall"),
    ("modified_recall", "avg_modified_recall"), ("unchanged_recall", "avg_unchanged_recall"),
    ("change_f1_strict", "avg_change_f1_strict"), ("change_f1_loose", "avg_change_f1_loose"),
    ("change_f1_floor", "avg_change_f1_floor"), ("no_change_acc", "avg_no_change_acc"),
    ("copy_excess", "avg_copy_excess"), ("parse_fail_rate", "parse_fail_rate"),
    ("hung_f1", "avg_hungarian_f1"), ("hung_prec", "avg_hungarian_prec"),
    ("hung_rec", "avg_hungarian_rec"), ("bleu", "avg_bleu"), ("rouge_l", "avg_rouge_l"),
    ("exact_match", "exact_match_rate"),
]
table = {}
for g, ref_leaf, n, ref in rows:
    print(f"\n[{g}]  (leaf {n} 개 · 기준 {ref_leaf})")
    table[g] = {"ref_leaf": ref_leaf, "n_leaf": n, "values": {}}
    for name, key in KEYS:
        vals = [ref[s]["copy_baseline"][key] for s in SECTIONS]
        table[g]["values"][name] = vals
        print(f"   {name:18s} " + " / ".join(f"{v:+.4f}" if name == "copy_excess" else f"{v:.4f}" for v in vals))

Path(sys.argv[1] if len(sys.argv) > 1 else "copy_baseline_table.json").write_text(
    json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n=> {sys.argv[1] if len(sys.argv) > 1 else 'copy_baseline_table.json'}")
