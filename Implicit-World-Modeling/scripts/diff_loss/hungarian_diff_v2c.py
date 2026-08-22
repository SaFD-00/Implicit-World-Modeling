"""
hungarian_diff_v2c.py
────────────────────────────────────────────────────────────────
hungarian_diff.py의 v2 판. 로직은 동일하고 hungarian_metric_v2에서 임포트.
v1 대비 실질적 차이는 하위 metric의 개선(_collect_texts, _match_cost)에서 발생.

[v2c = v2 + Cerebra 스키마]  이 파일은 `hungarian_diff_v2.py` 의 **복제본**이다.
`hungarian_diff_v2.py` 자체는 EXP02/05/06/07 학습 데이터의 생성기라 재현성을 위해 불가침이므로
(AGENTS 하드 제약 9, 2026-08-21 결정 #2), Cerebra(`data-bbox`/`aria-label`) 확장은
이 `_v2c` 파일에만 넣는다. AC_EXP08 파이프라인(`build_diff_targets.py`)만 이걸 import 하고,
기존 `preprocess_dataset_v2.py` 경로는 그대로 `_v2` 를 쓴다.
변경 내역: `docs/CHANGES_v2_cerebra.md`
"""

from __future__ import annotations

from hungarian_metric_v2c import (
    _hungarian_match,
    _text_sim,
    extract_elements,
)

UNCHANGED_COST_THRESHOLD = 0.05


def classify_diff(current_html: str, future_html: str) -> list[dict]:
    current_els = extract_elements(current_html)
    future_els = extract_elements(future_html)

    if not future_els:
        return []

    if not current_els:
        return [
            {
                "element": el,
                "future_seq_idx": i,
                "diff_type": "ADDED",
                "change_detail": {"text_sim": 0.0, "match_cost": -1.0},
            }
            for i, el in enumerate(future_els)
        ]

    pairs, _ = _hungarian_match(current_els, future_els)

    future_to_match: dict[int, tuple[int, float]] = {
        j: (i, cost) for i, j, cost in pairs
    }

    result: list[dict] = []
    for fut_seq_idx, fut_el in enumerate(future_els):
        if fut_seq_idx not in future_to_match:
            entry = {
                "element": fut_el,
                "future_seq_idx": fut_seq_idx,
                "diff_type": "ADDED",
                "change_detail": {"text_sim": 0.0, "match_cost": -1.0},
            }
        else:
            cur_idx, cost = future_to_match[fut_seq_idx]
            cur_el = current_els[cur_idx]
            text_sim = _text_sim(cur_el["text"], fut_el["text"])

            if cost <= UNCHANGED_COST_THRESHOLD:
                diff_type = "UNCHANGED"
            else:
                diff_type = "MODIFIED"

            entry = {
                "element": fut_el,
                "future_seq_idx": fut_seq_idx,
                "diff_type": diff_type,
                "change_detail": {
                    "text_sim": round(text_sim, 4),
                    "match_cost": round(cost, 5),
                },
            }

        result.append(entry)

    return result


def summarize_diff(diff_result: list[dict]) -> dict[str, int]:
    counts = {"ADDED": 0, "MODIFIED": 0, "UNCHANGED": 0}
    for d in diff_result:
        counts[d["diff_type"]] += 1
    return counts
