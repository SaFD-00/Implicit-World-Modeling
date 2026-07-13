"""
hungarian_diff.py
────────────────────────────────────────────────────────────────
current HTML → future HTML 사이의 element 단위 diff를 분류하는 모듈.

헝가리안 매칭 결과를 기반으로 각 future element에 대해:
  - UNCHANGED : current에 매칭됨 + cost가 거의 0 (변경 없음)
  - MODIFIED  : current에 매칭됨 + cost > 0 (내용/위치 변경)
  - ADDED     : current에 매칭되는 element가 없음 (새로 추가됨)

DELETED(current에만 있는 element)는 future 학습 대상이 아니므로 제외.

외부에서 호출하는 함수:
  classify_diff(current_html, future_html) -> list[dict]
"""

from __future__ import annotations

from hungarian_metric import (
    _hungarian_match,
    _text_sim,
    extract_elements,
)

# UNCHANGED / MODIFIED 경계 비용
# 이 값 이하면 "사실상 동일한 element"로 간주
UNCHANGED_COST_THRESHOLD = 0.05


def classify_diff(current_html: str, future_html: str) -> list[dict]:
    """
    current → future 사이의 diff를 element 단위로 분류.

    Args:
        current_html : 현재 상태 HTML 문자열
        future_html  : 미래 상태 HTML 문자열 (= assistant 답변)

    Returns:
        future element 기준 리스트. 각 항목:
        {
            "element"      : {"tag": str, "text": str, "index": int},
            "future_seq_idx": int,   # future_els 내 순서 인덱스 (0-based)
            "diff_type"    : "UNCHANGED" | "MODIFIED" | "ADDED",
            "change_detail": {
                "text_sim"  : float,   # 매칭된 current element와의 텍스트 유사도
                "match_cost": float,   # 헝가리안 매칭 비용 (ADDED이면 -1.0)
            }
        }

    Notes:
        - future_els가 비어 있으면 빈 리스트 반환
        - current_els가 비어 있으면 전부 ADDED로 분류
    """
    current_els = extract_elements(current_html)
    future_els = extract_elements(future_html)

    if not future_els:
        return []

    # current가 없으면 전부 ADDED
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

    # ── 헝가리안 매칭: current=pred, future=gt ─────────────────────────────
    pairs, _ = _hungarian_match(current_els, future_els)

    # future 인덱스 → (current 인덱스, 매칭 비용) 딕셔너리
    future_to_match: dict[int, tuple[int, float]] = {
        j: (i, cost) for i, j, cost in pairs
    }

    # ── future element 별 분류 ─────────────────────────────────────────────
    result: list[dict] = []
    for fut_seq_idx, fut_el in enumerate(future_els):
        if fut_seq_idx not in future_to_match:
            # current에 매칭되는 element 없음 → ADDED
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
    """diff_result 리스트에서 type 별 카운트를 반환 (로깅용)."""
    counts = {"ADDED": 0, "MODIFIED": 0, "UNCHANGED": 0}
    for d in diff_result:
        counts[d["diff_type"]] += 1
    return counts
