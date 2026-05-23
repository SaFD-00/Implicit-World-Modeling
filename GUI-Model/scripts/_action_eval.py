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
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r'\{[^{}]*\}', text)
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
    return (action.get('params') or {}).get(key)


def _norm(s):
    return str(s if s is not None else '').strip().lower()


# ── Step Accuracy 채점 ───────────────────────────────────────────────────
_FIELD_MATCH_TYPES = {
    'navigate_back', 'navigate_home', 'wait', 'finish',
    'click', 'long_press', 'scroll', 'open_app', 'input_text',
}


def _atype(action):
    """AndroidControl 스키마는 'action_type' 키 사용 (구 'type' fallback)."""
    if action is None:
        return ''
    return str(action.get('action_type', action.get('type', ''))).lower()


def evaluate_single(gt_action, pred_action):
    result = {
        'parsed': pred_action is not None,
        'type_correct': False,
        'step_correct': False,
        'has_index_check': False,    # click / long_click
        'has_dir_check': False,      # scroll
        'has_app_check': False,      # open_app
        'has_text_check': False,     # input
    }
    if pred_action is None:
        return result

    gt_type = _atype(gt_action)
    pred_type = _atype(pred_action)
    result['type_correct'] = (gt_type == pred_type)
    if not result['type_correct']:
        return result

    if gt_type in ('navigate_back', 'navigate_home', 'wait', 'finish'):
        result['step_correct'] = True
        return result

    if gt_type in ('click', 'long_press'):
        result['has_index_check'] = True
        result['step_correct'] = (
            str(gt_action.get('index')) == str(pred_action.get('index'))
        )
        return result

    if gt_type == 'scroll':
        result['has_dir_check'] = True
        result['step_correct'] = (
            _norm(_pval(gt_action, 'direction')) == _norm(_pval(pred_action, 'direction'))
        )
        return result

    if gt_type == 'open_app':
        result['has_app_check'] = True
        result['step_correct'] = (
            _norm(_pval(gt_action, 'app_name')) == _norm(_pval(pred_action, 'app_name'))
        )
        return result

    if gt_type == 'input_text':
        result['has_text_check'] = True
        result['step_correct'] = (
            _norm(_pval(gt_action, 'text')) == _norm(_pval(pred_action, 'text'))
        )
        return result

    return result


def _load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_pairs(gt_entries, pred_entries):
    """Compute metrics for a pre-loaded list of (gt, pred) pairs."""
    if len(gt_entries) != len(pred_entries):
        print(
            f"[warn] length mismatch: gt={len(gt_entries)} pred={len(pred_entries)}"
            f" → truncating to {min(len(gt_entries), len(pred_entries))}",
            file=sys.stderr,
        )

    per_type = defaultdict(lambda: {
        'count': 0, 'type_correct': 0, 'step_correct': 0,
    })
    cond = {
        'index': {'n': 0, 'k': 0},
        'dir':   {'n': 0, 'k': 0},
        'app':   {'n': 0, 'k': 0},
        'text':  {'n': 0, 'k': 0},
    }

    total = 0
    parsed = 0
    type_correct = 0
    step_correct = 0

    for gt_entry, pred_entry in zip(gt_entries, pred_entries):
        gt_action = parse_action(gt_entry['messages'][-1]['value'])
        pred_text = pred_entry.get('predict', pred_entry.get('output', ''))
        pred_action = parse_action(pred_text)

        r = evaluate_single(gt_action, pred_action)
        gt_type = _atype(gt_action) or 'unknown'

        total += 1
        parsed += int(r['parsed'])
        type_correct += int(r['type_correct'])
        step_correct += int(r['step_correct'])

        per_type[gt_type]['count'] += 1
        per_type[gt_type]['type_correct'] += int(r['type_correct'])
        per_type[gt_type]['step_correct'] += int(r['step_correct'])

        if r['has_index_check']:
            cond['index']['n'] += 1
            cond['index']['k'] += int(r['step_correct'])
        if r['has_dir_check']:
            cond['dir']['n'] += 1
            cond['dir']['k'] += int(r['step_correct'])
        if r['has_app_check']:
            cond['app']['n'] += 1
            cond['app']['k'] += int(r['step_correct'])
        if r['has_text_check']:
            cond['text']['n'] += 1
            cond['text']['k'] += int(r['step_correct'])

    parse_rate = parsed / total if total else 0
    type_acc = type_correct / total if total else 0
    step_acc = step_correct / total if total else 0

    per_type_summary = {}
    for t, d in per_type.items():
        per_type_summary[t] = {
            'count':    d['count'],
            'type_acc': round(d['type_correct'] / d['count'] if d['count'] else 0, 4),
            'step_acc': round(d['step_correct'] / d['count'] if d['count'] else 0, 4),
        }

    macro_step = (sum(v['step_acc'] for v in per_type_summary.values()) /
                  len(per_type_summary)) if per_type_summary else 0

    def _ratio(c):
        return c['k'] / c['n'] if c['n'] else 0

    return {
        'total':                total,
        'parse_rate':           round(parse_rate, 4),
        'type_accuracy':        round(type_acc, 4),
        'step_accuracy':        round(step_acc, 4),
        'macro_step_accuracy':  round(macro_step, 4),
        'cond_index_acc':       round(_ratio(cond['index']), 4),
        'cond_dir_acc':         round(_ratio(cond['dir']),   4),
        'cond_app_acc':         round(_ratio(cond['app']),   4),
        'cond_text_acc':        round(_ratio(cond['text']),  4),
        'per_type':             per_type_summary,
    }


def evaluate_predictions(test_path, pred_path):
    """Backward-compatible file-based entry point."""
    return evaluate_pairs(_load_jsonl(test_path), _load_jsonl(pred_path))


# ── CLI ──────────────────────────────────────────────────────────────────
def _print_metrics_row(label, metrics):
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

    if split_mode:
        # Require both ID and OOD paths if any split flag is set.
        missing = [
            name for name, val in [
                ("--test-id", args.test_id), ("--pred-id", args.pred_id),
                ("--test-ood", args.test_ood), ("--pred-ood", args.pred_ood),
            ] if not val
        ]
        if missing:
            print(f"[score] ERROR: split mode needs {missing}", file=sys.stderr)
            return 2

        gt_id = _load_jsonl(args.test_id)
        pr_id = _load_jsonl(args.pred_id)
        gt_ood = _load_jsonl(args.test_ood)
        pr_ood = _load_jsonl(args.pred_ood)

        m_id = evaluate_pairs(gt_id, pr_id)
        m_ood = evaluate_pairs(gt_ood, pr_ood)
        m_overall = evaluate_pairs(gt_id + gt_ood, pr_id + pr_ood)

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
            print("[score] ERROR: --test and --pred required in single-pair mode",
                  file=sys.stderr)
            return 2
        metrics = evaluate_predictions(args.test, args.pred)
        _print_metrics_row("all", metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
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
    p_s.set_defaults(func=_cmd_score)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
