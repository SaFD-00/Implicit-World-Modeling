#!/usr/bin/env python3
"""
Standalone Stage 2 Action Prediction evaluator.

본 스크립트는 Stage 2 Action Prediction 채점의 **정본** 이며, 노트북
``implicit-world-modeling.ipynb`` Section 8 의 정본 셀이 이 파일과 글자 단위로 동일하게
유지된다. 메트릭은 AndroidControl 데이터셋의 실제 스키마
(``bounds`` 필드 영구 부재, element-index 기반 grounding) 에 맞춘
**Step Accuracy (SA)** 단일 1차 지표를 사용한다.

Subcommand
----------
score   : prediction.jsonl 대해 Action 메트릭 → action_metrics.json 저장
          ID/OOD 파일이 주어지면 overall/in_domain/out_of_domain 3-섹션 출력.

Examples
--------
  # 1. 기본 (단일 pair) — overall 만 기록
  python scripts/_action_eval.py score \\
      --test   data/AndroidControl/implicit-world-modeling_stage2_test_id.jsonl \\
      --pred   .../epoch-1/generated_predictions_id.jsonl \\
      --output .../epoch-1/action_metrics.json

  # 2. ID + OOD 동시 입력 — overall/in_domain/out_of_domain 3 섹션
  python scripts/_action_eval.py score \\
      --test-id   data/AndroidControl/implicit-world-modeling_stage2_test_id.jsonl \\
      --pred-id   .../epoch-1/generated_predictions_id.jsonl \\
      --test-ood  data/AndroidControl/implicit-world-modeling_stage2_test_ood.jsonl \\
      --pred-ood  .../epoch-1/generated_predictions_ood.jsonl \\
      --output    .../epoch-1/action_metrics.json

Step Accuracy 정의 (요약)
-------------------------
SA = (1/N) · Σ correct_i,  correct_i = 1 iff (parse_ok ∧ type==gt ∧ field_match(type))

  type            field_match
  ─────────────── ──────────────────────────────────────────────
  navigate_back   (필드 없음) → 항상 통과
  finish          (status 단일값) → 항상 통과
  click           str(pred.index) == str(gt.index)
  long_click      str(pred.index) == str(gt.index)
  scroll          norm(direction) 일치
  open_app        norm(params.app) 일치
  input           norm(params.text) 일치 (gt.index=null 무시)

  norm(s) = str(s or '').strip().lower()

Coord mode (--coord-mode)
-------------------------
index (기본) : 위 표 그대로. EXP01~EXP04 용, 바이트 단위 불변.
xy           : EXP05 의 xy 통일 액션 스페이스용. GT 스키마가 다르다
               (키 ``action``, ``<action>{...}</action>`` 래핑, click 은 ``coordinate``,
               방향 액션은 ``swipe`` 의 ``coordinate1``→``coordinate2``).

  type                     field_match
  ──────────────────────── ────────────────────────────────────────────────
  wait / navigate_back     (필드 없음) → 타입만 일치하면 통과
  navigate_home
  click / long_press       test 샘플의 UI State XML 에서 GT 좌표를 포함하는
                           최소 면적 element 의 bounds 를 GT bbox 로 잡고,
                           pred 좌표가 그 bbox 안(경계 포함)이면 정답.
                           포함 element 가 없으면 오답 + no_bbox_n 집계.
  swipe                    coordinate1→coordinate2 벡터의 주 성분 방향 일치
                           (|dx| >= |dy| → left/right, else up/down)
  open                     norm(app_name) 일치
  type                     norm(text) 일치 (좌표 무관)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Action Parsing ───────────────────────────────────────────────────────
def parse_action(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ── Field 추출 헬퍼 (top-level + nested params 모두 지원) ────────────────
def _pval(action, key):
    if action is None:
        return None
    if key in action:
        return action[key]
    return (action.get("params") or {}).get(key)


def _norm(s):
    return str(s if s is not None else "").strip().lower()


# ── Step Accuracy 채점 ───────────────────────────────────────────────────
_FIELD_MATCH_TYPES = {
    "navigate_back",
    "navigate_home",
    "wait",
    "finish",
    "click",
    "long_press",
    "scroll",
    "open_app",
    "input_text",
}


def _atype(action):
    """AndroidControl 스키마는 'action_type' 키 사용 (구 'type' fallback)."""
    if action is None:
        return ""
    return str(action.get("action_type", action.get("type", ""))).lower()


def evaluate_single(gt_action, pred_action):
    result = {
        "parsed": pred_action is not None,
        "type_correct": False,
        "step_correct": False,
        "has_index_check": False,  # click / long_click
        "has_dir_check": False,  # scroll
        "has_app_check": False,  # open_app
        "has_text_check": False,  # input
    }
    if pred_action is None:
        return result

    gt_type = _atype(gt_action)
    pred_type = _atype(pred_action)
    result["type_correct"] = gt_type == pred_type
    if not result["type_correct"]:
        return result

    if gt_type in ("navigate_back", "navigate_home", "wait", "finish"):
        result["step_correct"] = True
        return result

    if gt_type in ("click", "long_press"):
        result["has_index_check"] = True
        result["step_correct"] = str(gt_action.get("index")) == str(
            pred_action.get("index")
        )
        return result

    if gt_type == "scroll":
        result["has_dir_check"] = True
        result["step_correct"] = _norm(_pval(gt_action, "direction")) == _norm(
            _pval(pred_action, "direction")
        )
        return result

    if gt_type == "open_app":
        result["has_app_check"] = True
        result["step_correct"] = _norm(_pval(gt_action, "app_name")) == _norm(
            _pval(pred_action, "app_name")
        )
        return result

    if gt_type == "input_text":
        result["has_text_check"] = True
        result["step_correct"] = _norm(_pval(gt_action, "text")) == _norm(
            _pval(pred_action, "text")
        )
        return result

    return result


# ── XY coord-mode (EXP05 전용) ───────────────────────────────────────────
# EXP05 는 액션 스페이스가 xy 로 통일돼 index grounding 이 없다. GT 스키마도 다르다:
#   <action>{"action":"click","coordinate":[622,620]}</action>
#   <action>{"action":"swipe","coordinate1":[420,1407],"coordinate2":[420,469]}</action>
# index 모드 경로(_atype / evaluate_single)는 건드리지 않고 이 블록 안에서만 처리한다.
_UI_STATE_MARKER = "Current UI State:"
_SCREENSHOT_MARKER = "[Screenshot]"
_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

_XY_NO_FIELD_TYPES = {"wait", "navigate_back", "navigate_home", "finish"}

# bbox 채점은 pred 좌표가 GT 와 같은 절대 픽셀 공간이라고 가정한다. 다른 좌표계
# (예: 0~1 정규화) 로 학습된 체크포인트는 bbox 정확도가 0 으로 나오는데, 그것이
# "모델이 못 배웠다" 인지 "좌표계가 다르다" 인지 메트릭만으로는 구분되지 않는다.
# 채점 결과는 바꾸지 않고 경고만 남겨 조용한 오진단을 막는다.
XY_SCREEN_W = 840
XY_SCREEN_H = 1876
_XY_UNIT_RATIO = 0.9  # pred 좌표가 이 비율 이상 0~1 구간이면 정규화 좌표로 의심
_XY_OOR_RATIO = 0.5  # pred 좌표가 이 비율 이상 화면 범위를 벗어나면 경고


def _xy_atype(action):
    """EXP05 스키마는 'action' 키 사용 (구 action_type/type fallback)."""
    if action is None:
        return ""
    return str(
        action.get("action", action.get("action_type", action.get("type", "")))
    ).lower()


def _extract_ui_xml(entry):
    """test 샘플의 user content 에서 'Current UI State:' ~ '[Screenshot]' 구간 XML 추출."""
    for msg in entry.get("messages", []):
        text = msg.get("value", "")
        start = text.find(_UI_STATE_MARKER)
        if start < 0:
            continue
        start += len(_UI_STATE_MARKER)
        end = text.find(_SCREENSHOT_MARKER, start)
        return text[start : end if end >= 0 else len(text)].strip()
    return ""


def _bbox_elements(xml_str):
    """bounds 속성을 가진 element 의 (x1, y1, x2, y2) 목록."""
    if not xml_str:
        return []
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(xml_str, "html.parser")
    boxes = []
    for el in soup.find_all(True):
        m = _BOUNDS_RE.search(el.get("bounds", "") or "")
        if m:
            boxes.append(tuple(int(v) for v in m.groups()))
    return boxes


def _gt_bbox(boxes, point):
    """GT 좌표를 포함하는 element 중 최소 면적인 것의 bbox. 없으면 None."""
    x, y = point
    containing = [b for b in boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if not containing:
        return None
    return min(containing, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _coord(action, key):
    """[x, y] 형태의 좌표를 (x, y) float 튜플로. 실패 시 None."""
    v = _pval(action, key)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return float(v[0]), float(v[1])
        except (TypeError, ValueError):
            return None
    return None


def _pred_coords(pred_action):
    """pred action 에 담긴 모든 좌표. 좌표계 sanity 검사용 (채점에는 미사용)."""
    if pred_action is None:
        return []
    pts = []
    for key in ("coordinate", "coordinate1", "coordinate2"):
        pt = _coord(pred_action, key)
        if pt is not None:
            pts.append(pt)
    return pts


def _warn_coord_space(points):
    """pred 좌표가 절대 픽셀 공간(840x1876)을 크게 벗어나면 경고 1줄. 결과는 불변."""
    n = len(points)
    if not n:
        return
    n_unit = sum(1 for x, y in points if 0 <= x <= 1 and 0 <= y <= 1)
    n_oor = sum(
        1 for x, y in points if not (0 <= x <= XY_SCREEN_W and 0 <= y <= XY_SCREEN_H)
    )
    if n_unit / n >= _XY_UNIT_RATIO:
        print(
            f"[warn] pred 좌표 {n_unit}/{n} 이 0~1 구간에 몰려 있습니다 — 정규화 좌표로 보입니다. "
            f"xy 채점은 절대 픽셀({XY_SCREEN_W}x{XY_SCREEN_H})을 가정하므로 "
            f"click/long_press 가 전부 오답 처리됩니다.",
            file=sys.stderr,
        )
    elif n_oor / n >= _XY_OOR_RATIO:
        print(
            f"[warn] pred 좌표 {n_oor}/{n} 이 화면 범위({XY_SCREEN_W}x{XY_SCREEN_H})를 "
            f"벗어납니다 — 좌표계 불일치 가능성이 있습니다.",
            file=sys.stderr,
        )


def _primary_direction(start, end):
    """start→end 벡터의 주 성분 방향. |dx| >= |dy| → left/right, else up/down."""
    if start is None or end is None:
        return None
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def evaluate_single_xy(gt_action, pred_action, ui_xml):
    result = {
        "parsed": pred_action is not None,
        "type_correct": False,
        "step_correct": False,
        "has_bbox_check": False,  # click / long_press
        "has_dir_check": False,  # swipe
        "has_app_check": False,  # open
        "has_text_check": False,  # type
        "no_bbox": False,  # GT 좌표를 포함하는 element 부재
    }
    if pred_action is None:
        return result

    gt_type = _xy_atype(gt_action)
    pred_type = _xy_atype(pred_action)
    result["type_correct"] = gt_type == pred_type
    if not result["type_correct"]:
        return result

    if gt_type in _XY_NO_FIELD_TYPES:
        result["step_correct"] = True
        return result

    if gt_type in ("click", "long_press"):
        result["has_bbox_check"] = True
        gt_pt = _coord(gt_action, "coordinate")
        pred_pt = _coord(pred_action, "coordinate")
        if gt_pt is None:
            return result
        bbox = _gt_bbox(_bbox_elements(ui_xml), gt_pt)
        if bbox is None:
            result["no_bbox"] = True
            return result
        if pred_pt is None:
            return result
        x1, y1, x2, y2 = bbox
        result["step_correct"] = x1 <= pred_pt[0] <= x2 and y1 <= pred_pt[1] <= y2
        return result

    if gt_type == "swipe":
        result["has_dir_check"] = True
        gt_dir = _primary_direction(
            _coord(gt_action, "coordinate1"), _coord(gt_action, "coordinate2")
        )
        pred_dir = _primary_direction(
            _coord(pred_action, "coordinate1"), _coord(pred_action, "coordinate2")
        )
        result["step_correct"] = gt_dir is not None and gt_dir == pred_dir
        return result

    if gt_type == "open":
        result["has_app_check"] = True
        result["step_correct"] = _norm(_pval(gt_action, "app_name")) == _norm(
            _pval(pred_action, "app_name")
        )
        return result

    if gt_type == "type":
        result["has_text_check"] = True
        result["step_correct"] = _norm(_pval(gt_action, "text")) == _norm(
            _pval(pred_action, "text")
        )
        return result

    return result


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_pairs(gt_entries, pred_entries, coord_mode="index"):
    """Compute metrics for a pre-loaded list of (gt, pred) pairs."""
    if coord_mode == "xy":
        return _evaluate_pairs_xy(gt_entries, pred_entries)
    if len(gt_entries) != len(pred_entries):
        print(
            f"[warn] length mismatch: gt={len(gt_entries)} pred={len(pred_entries)}"
            f" → truncating to {min(len(gt_entries), len(pred_entries))}",
            file=sys.stderr,
        )

    per_type = defaultdict(
        lambda: {
            "count": 0,
            "type_correct": 0,
            "step_correct": 0,
        }
    )
    cond = {
        "index": {"n": 0, "k": 0},
        "dir": {"n": 0, "k": 0},
        "app": {"n": 0, "k": 0},
        "text": {"n": 0, "k": 0},
    }

    total = 0
    parsed = 0
    type_correct = 0
    step_correct = 0

    for gt_entry, pred_entry in zip(gt_entries, pred_entries):
        gt_action = parse_action(gt_entry["messages"][-1]["value"])
        pred_text = pred_entry.get("predict", pred_entry.get("output", ""))
        pred_action = parse_action(pred_text)

        r = evaluate_single(gt_action, pred_action)
        gt_type = _atype(gt_action) or "unknown"

        total += 1
        parsed += int(r["parsed"])
        type_correct += int(r["type_correct"])
        step_correct += int(r["step_correct"])

        per_type[gt_type]["count"] += 1
        per_type[gt_type]["type_correct"] += int(r["type_correct"])
        per_type[gt_type]["step_correct"] += int(r["step_correct"])

        if r["has_index_check"]:
            cond["index"]["n"] += 1
            cond["index"]["k"] += int(r["step_correct"])
        if r["has_dir_check"]:
            cond["dir"]["n"] += 1
            cond["dir"]["k"] += int(r["step_correct"])
        if r["has_app_check"]:
            cond["app"]["n"] += 1
            cond["app"]["k"] += int(r["step_correct"])
        if r["has_text_check"]:
            cond["text"]["n"] += 1
            cond["text"]["k"] += int(r["step_correct"])

    parse_rate = parsed / total if total else 0
    type_acc = type_correct / total if total else 0
    step_acc = step_correct / total if total else 0

    per_type_summary = {}
    for t, d in per_type.items():
        per_type_summary[t] = {
            "count": d["count"],
            "type_acc": round(d["type_correct"] / d["count"] if d["count"] else 0, 4),
            "step_acc": round(d["step_correct"] / d["count"] if d["count"] else 0, 4),
        }

    macro_step = (
        (sum(v["step_acc"] for v in per_type_summary.values()) / len(per_type_summary))
        if per_type_summary
        else 0
    )

    def _ratio(c):
        return c["k"] / c["n"] if c["n"] else 0

    return {
        "total": total,
        "parse_rate": round(parse_rate, 4),
        "type_accuracy": round(type_acc, 4),
        "step_accuracy": round(step_acc, 4),
        "macro_step_accuracy": round(macro_step, 4),
        "cond_index_acc": round(_ratio(cond["index"]), 4),
        "cond_dir_acc": round(_ratio(cond["dir"]), 4),
        "cond_app_acc": round(_ratio(cond["app"]), 4),
        "cond_text_acc": round(_ratio(cond["text"]), 4),
        "per_type": per_type_summary,
    }


def _evaluate_pairs_xy(gt_entries, pred_entries):
    """EXP05 xy 액션 스페이스 채점. bbox 포함 / 주 성분 방향 / 텍스트 매칭."""
    if len(gt_entries) != len(pred_entries):
        print(
            f"[warn] length mismatch: gt={len(gt_entries)} pred={len(pred_entries)}"
            f" → truncating to {min(len(gt_entries), len(pred_entries))}",
            file=sys.stderr,
        )

    per_type = defaultdict(
        lambda: {
            "count": 0,
            "type_correct": 0,
            "step_correct": 0,
        }
    )
    cond = {
        "bbox": {"n": 0, "k": 0},
        "dir": {"n": 0, "k": 0},
        "app": {"n": 0, "k": 0},
        "text": {"n": 0, "k": 0},
    }

    total = parsed = type_correct = step_correct = no_bbox_n = 0
    coord_samples = []

    for gt_entry, pred_entry in zip(gt_entries, pred_entries):
        gt_action = parse_action(gt_entry["messages"][-1]["value"])
        pred_text = pred_entry.get("predict", pred_entry.get("output", ""))
        pred_action = parse_action(pred_text)

        r = evaluate_single_xy(gt_action, pred_action, _extract_ui_xml(gt_entry))
        gt_type = _xy_atype(gt_action) or "unknown"
        coord_samples.extend(_pred_coords(pred_action))

        total += 1
        parsed += int(r["parsed"])
        type_correct += int(r["type_correct"])
        step_correct += int(r["step_correct"])
        no_bbox_n += int(r["no_bbox"])

        per_type[gt_type]["count"] += 1
        per_type[gt_type]["type_correct"] += int(r["type_correct"])
        per_type[gt_type]["step_correct"] += int(r["step_correct"])

        for flag, key in (
            ("has_bbox_check", "bbox"),
            ("has_dir_check", "dir"),
            ("has_app_check", "app"),
            ("has_text_check", "text"),
        ):
            if r[flag]:
                cond[key]["n"] += 1
                cond[key]["k"] += int(r["step_correct"])

    _warn_coord_space(coord_samples)

    per_type_summary = {}
    for t, d in per_type.items():
        per_type_summary[t] = {
            "count": d["count"],
            "type_acc": round(d["type_correct"] / d["count"] if d["count"] else 0, 4),
            "step_acc": round(d["step_correct"] / d["count"] if d["count"] else 0, 4),
        }

    macro_step = (
        (sum(v["step_acc"] for v in per_type_summary.values()) / len(per_type_summary))
        if per_type_summary
        else 0
    )

    def _ratio(c):
        return c["k"] / c["n"] if c["n"] else 0

    return {
        "total": total,
        "parse_rate": round(parsed / total if total else 0, 4),
        "type_accuracy": round(type_correct / total if total else 0, 4),
        "step_accuracy": round(step_correct / total if total else 0, 4),
        "macro_step_accuracy": round(macro_step, 4),
        "cond_bbox_acc": round(_ratio(cond["bbox"]), 4),
        "cond_dir_acc": round(_ratio(cond["dir"]), 4),
        "cond_app_acc": round(_ratio(cond["app"]), 4),
        "cond_text_acc": round(_ratio(cond["text"]), 4),
        "no_bbox_n": no_bbox_n,
        "per_type": per_type_summary,
    }


def evaluate_predictions(test_path, pred_path, coord_mode="index"):
    """Backward-compatible file-based entry point."""
    return evaluate_pairs(_load_jsonl(test_path), _load_jsonl(pred_path), coord_mode)


# ── CLI ──────────────────────────────────────────────────────────────────
def _print_metrics_row(label, metrics):
    if "cond_bbox_acc" in metrics:  # xy 모드
        print(
            f"[score:{label}] total={metrics['total']}  "
            f"parse={metrics['parse_rate']:.2%}  "
            f"type={metrics['type_accuracy']:.4f}  "
            f"step={metrics['step_accuracy']:.4f}  "
            f"macro={metrics['macro_step_accuracy']:.4f}  "
            f"bbox={metrics['cond_bbox_acc']:.4f}  "
            f"dir={metrics['cond_dir_acc']:.4f}  "
            f"app={metrics['cond_app_acc']:.4f}  "
            f"text={metrics['cond_text_acc']:.4f}  "
            f"no_bbox={metrics['no_bbox_n']}"
        )
        return
    print(
        f"[score:{label}] total={metrics['total']}  "
        f"parse={metrics['parse_rate']:.2%}  "
        f"type={metrics['type_accuracy']:.4f}  "
        f"step={metrics['step_accuracy']:.4f}  "
        f"macro={metrics['macro_step_accuracy']:.4f}  "
        f"index={metrics['cond_index_acc']:.4f}  "
        f"dir={metrics['cond_dir_acc']:.4f}  "
        f"app={metrics['cond_app_acc']:.4f}  "
        f"text={metrics['cond_text_acc']:.4f}"
    )


def _cmd_score(args):
    split_mode = bool(args.test_id or args.pred_id or args.test_ood or args.pred_ood)
    coord_mode = getattr(args, "coord_mode", "index")

    if split_mode:
        # Require both ID and OOD paths if any split flag is set.
        missing = [
            name
            for name, val in [
                ("--test-id", args.test_id),
                ("--pred-id", args.pred_id),
                ("--test-ood", args.test_ood),
                ("--pred-ood", args.pred_ood),
            ]
            if not val
        ]
        if missing:
            print(f"[score] ERROR: split mode needs {missing}", file=sys.stderr)
            return 2

        gt_id = _load_jsonl(args.test_id)
        pr_id = _load_jsonl(args.pred_id)
        gt_ood = _load_jsonl(args.test_ood)
        pr_ood = _load_jsonl(args.pred_ood)

        m_id = evaluate_pairs(gt_id, pr_id, coord_mode)
        m_ood = evaluate_pairs(gt_ood, pr_ood, coord_mode)
        m_overall = evaluate_pairs(gt_id + gt_ood, pr_id + pr_ood, coord_mode)

        metrics = {
            "overall": m_overall,
            "in_domain": m_id,
            "out_of_domain": m_ood,
        }
        _print_metrics_row("overall", m_overall)
        _print_metrics_row("in_domain", m_id)
        _print_metrics_row("out_of_domain", m_ood)
    else:
        if not (args.test and args.pred):
            print(
                "[score] ERROR: --test and --pred required in single-pair mode",
                file=sys.stderr,
            )
            return 2
        metrics = evaluate_predictions(args.test, args.pred, coord_mode)
        _print_metrics_row("all", metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[score] saved: {args.output}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Stage 2 Action Prediction evaluator")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser(
        "score",
        help="Compute action metrics. Single-pair (--test/--pred) or "
        "ID/OOD split (--test-id/--pred-id/--test-ood/--pred-ood).",
    )
    p_s.add_argument("--test", default=None)
    p_s.add_argument("--pred", default=None)
    p_s.add_argument("--test-id", default=None, dest="test_id")
    p_s.add_argument("--pred-id", default=None, dest="pred_id")
    p_s.add_argument("--test-ood", default=None, dest="test_ood")
    p_s.add_argument("--pred-ood", default=None, dest="pred_ood")
    p_s.add_argument("--output", required=True)
    p_s.add_argument(
        "--coord-mode",
        default="index",
        choices=["index", "xy"],
        dest="coord_mode",
        help="index (기본, EXP01~04): click/long_click 을 element index 로 채점. "
        "xy (EXP05): click/long_press 는 GT 좌표가 속한 element 의 bounds 안에 "
        "pred 좌표가 들어가면 정답, swipe 는 주 성분 방향 일치, type 은 텍스트만 검사.",
    )
    p_s.set_defaults(func=_cmd_score)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
