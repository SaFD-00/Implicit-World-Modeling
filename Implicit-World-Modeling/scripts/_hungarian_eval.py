#!/usr/bin/env python3
"""
Standalone Hungarian/BLEU/ROUGE evaluator for Stage 1 World-Modeling predictions.

Ported from the Stage 1 evaluation section of the project notebook
(implicit-world-modeling.ipynb, Section 5). Used by scripts/stage1_eval.sh.

Subcommand
----------
score   : prediction.jsonl 의 평균 메트릭 계산 → hungarian_metrics.json 저장.
          ID/OOD 파일이 주어지면 overall/in_domain/out_of_domain 3-섹션 출력.
          --exclude-action ACTION 으로 GT action.type==ACTION 행을 양쪽에서 동시 drop
          후 메트릭 계산. 정규 eval 의 generated_predictions*.jsonl 을 그대로 입력으로
          받아 추론 재실행 없이 필터 산출만 만든다.

Examples
--------
  # 1. Single-pair (MC / MB) — overall 만 기록
  python scripts/_hungarian_eval.py score \\
      --test  data/MonkeyCollection/implicit-world-modeling_stage1_test.jsonl \\
      --pred  .../generated_predictions.jsonl \\
      --output .../hungarian_metrics.json

  # 2. ID + OOD 동시 입력 (AC) — overall/in_domain/out_of_domain 3 섹션
  python scripts/_hungarian_eval.py score \\
      --test-id   data/AndroidControl/implicit-world-modeling_stage1_test_id.jsonl \\
      --pred-id   .../generated_predictions_id.jsonl \\
      --test-ood  data/AndroidControl/implicit-world-modeling_stage1_test_ood.jsonl \\
      --pred-ood  .../generated_predictions_ood.jsonl \\
      --output    .../hungarian_metrics.json

  # 3. 필터 산출 (open_app 행 제외) — 정규 eval 산출물을 재활용해 sibling 디렉토리에
  #    필터 jsonl + hungarian_metrics + predict_results 를 idempotent 저장
  python scripts/_hungarian_eval.py score \\
      --test  data/MobiBench/implicit-world-modeling_stage1.jsonl \\
      --pred  on-MB/generated_predictions.jsonl \\
      --exclude-action open_app \\
      --filtered-test-dir data/MobiBench \\
      --filtered-pred-dir on-MB-without-open_app \\
      --output            on-MB-without-open_app/hungarian_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

# bs4 / munkres 는 score 서브커맨드에서만 사용. 지연 로딩.
BeautifulSoup = None  # type: ignore
Munkres = None  # type: ignore

def _lazy_deps():
    """bs4 / munkres 를 지연 로드. score 서브커맨드 진입 시 한 번 호출."""
    global BeautifulSoup, Munkres
    if BeautifulSoup is None:
        from bs4 import BeautifulSoup as _BS
        BeautifulSoup = _BS
    if Munkres is None:
        from munkres import Munkres as _M
        Munkres = _M


# ── Hungarian Metric 상수 (Cell 25 상수 복제) ──────────────────────────────
INTERACTIVE_TAGS = {"button", "input", "a", "select", "textarea"}
CONTENT_TAGS     = {"p", "img", "span"}
CLICKABLE_ATTRS  = {"clickable", "long-clickable"}

W_TAG   = 3.0
W_TEXT  = 1.5
W_INDEX = 0.2

MATCH_THRESHOLD = 1.5
INDEX_TAU       = 2


# ── 요소 추출 ────────────────────────────────────────────────────────────
def _collect_texts(el):
    tokens = set()
    def add(v):
        if v:
            tokens.add(v.strip())
    add(el.get("description"))
    add(el.get("id"))
    for child in el.find_all(True):
        add(child.get("description"))
        add(child.get("id"))
        t = child.get_text(strip=True)
        if t:
            tokens.add(t)
    t = el.get_text(strip=True)
    if t:
        tokens.add(t)
    return " | ".join(sorted(tokens)) if tokens else ""


def _safe_int(v, default=-1):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def extract_elements(xml_str):
    try:
        soup = BeautifulSoup(xml_str, "xml")
    except Exception:
        soup = BeautifulSoup(xml_str, "html.parser")
    elements = []
    for el in soup.find_all(True):
        tag  = el.name
        idx  = _safe_int(el.get("index", -1))
        text = _collect_texts(el)
        is_interactive = tag in INTERACTIVE_TAGS
        is_content     = (tag in CONTENT_TAGS) and bool(text)
        is_clickable   = any(el.get(a) for a in CLICKABLE_ATTRS)
        if is_interactive or is_content or is_clickable:
            elements.append({"tag": tag, "text": text, "index": idx})
    return elements


# ── 매칭 비용 & Hungarian ───────────────────────────────────────────────
def _text_sim(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = set(a.lower().replace("|", "").split())
    sb = set(b.lower().replace("|", "").split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _match_cost(e1, e2, max_idx):
    if e1["tag"] != e2["tag"]:
        return W_TAG
    tc = W_TEXT  * (1.0 - _text_sim(e1["text"], e2["text"]))
    ic = W_INDEX * (abs(e1["index"] - e2["index"]) / max(max_idx, 1))
    return round(tc + ic, 5)


def _hungarian_match(pred, gt):
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return [], []
    max_idx = max(
        (e["index"] for e in pred + gt if e["index"] >= 0),
        default=1,
    )
    matrix = [[_match_cost(p, g, max_idx) for g in gt] for p in pred]
    size = max(n, m)
    padded = [row + [MATCH_THRESHOLD * 2] * (size - len(row)) for row in matrix]
    while len(padded) < size:
        padded.append([MATCH_THRESHOLD * 2] * size)
    indexes = Munkres().compute(padded)
    pairs = []
    for i, j in indexes:
        if i < n and j < m and matrix[i][j] < MATCH_THRESHOLD:
            pairs.append((i, j, matrix[i][j]))
    return pairs, matrix


def compute_hungarian_acc(pred_str, gt_str):
    _zero = {
        "hungarian_ea": 0.0, "hungarian_f1": 0.0,
        "hungarian_prec": 0.0, "hungarian_rec": 0.0,
        "hungarian_text": 0.0, "hungarian_idx": 0.0,
    }
    try:
        pred_els = extract_elements(pred_str)
        gt_els   = extract_elements(gt_str)
    except Exception:
        return _zero
    if not gt_els:
        return _zero

    pairs, _ = _hungarian_match(pred_els, gt_els)
    n_pred, n_gt, n_matched = len(pred_els), len(gt_els), len(pairs)

    ea   = n_matched / max(n_pred, n_gt) if max(n_pred, n_gt) > 0 else 0.0
    prec = n_matched / n_pred             if n_pred  > 0           else 0.0
    rec  = n_matched / n_gt               if n_gt    > 0           else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0    else 0.0

    if pairs:
        text_sims = [_text_sim(pred_els[i]["text"], gt_els[j]["text"]) for i, j, _ in pairs]
        idx_diffs = [abs(pred_els[i]["index"] - gt_els[j]["index"]) for i, j, _ in pairs]
        text_avg  = sum(text_sims) / len(text_sims)
        idx_acc   = sum(1 for d in idx_diffs if d <= INDEX_TAU) / len(idx_diffs)
    else:
        text_avg = 0.0
        idx_acc  = 0.0

    return {
        "hungarian_ea":   round(ea, 4),
        "hungarian_f1":   round(f1, 4),
        "hungarian_prec": round(prec, 4),
        "hungarian_rec":  round(rec, 4),
        "hungarian_text": round(text_avg, 4),
        "hungarian_idx":  round(idx_acc, 4),
    }


# ── BLEU / ROUGE-L ──────────────────────────────────────────────────────
def calc_bleu(reference, hypothesis, max_n=4):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens)))
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))
        clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        precisions.append(0 if total == 0 else clipped / total)
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


def calc_rouge_n(reference, hypothesis, n):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if len(ref_tokens) < n or len(hyp_tokens) < n:
        return 0.0
    ref_ng = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
    hyp_ng = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
    overlap = sum((ref_ng & hyp_ng).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp_ng.values())
    recall    = overlap / sum(ref_ng.values())
    return 2 * precision * recall / (precision + recall)


def calc_rouge_l(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    precision = lcs_len / n
    recall    = lcs_len / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── 전체 평가 (Cell 26 evaluate_stage1_predictions 포팅) ───────────────
def _load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_pairs(gt_entries, pred_entries):
    """Pair-level Hungarian/BLEU/ROUGE 집계. ID/OOD 합산용으로 entries 리스트를 직접 받음."""
    results = []
    for gt_entry, pred_entry in zip(gt_entries, pred_entries):
        gt_text = gt_entry['messages'][-1]['value']
        pred_text = pred_entry.get('predict', pred_entry.get('output', ''))
        results.append({
            'bleu':        calc_bleu(gt_text, pred_text),
            'rouge_1':     calc_rouge_n(gt_text, pred_text, 1),
            'rouge_2':     calc_rouge_n(gt_text, pred_text, 2),
            'rouge_l':     calc_rouge_l(gt_text, pred_text),
            'exact_match': 1.0 if gt_text.strip() == pred_text.strip() else 0.0,
            'hungarian':   compute_hungarian_acc(pred_text, gt_text),
        })

    total = len(results)
    avg = lambda key: sum(r[key] for r in results) / total if total else 0.0
    hung_avg = lambda key: sum(r['hungarian'][key] for r in results) / total if total else 0.0
    return {
        'total': total,
        'avg_bleu':           round(avg('bleu'), 4),
        'avg_rouge_1':        round(avg('rouge_1'), 4),
        'avg_rouge_2':        round(avg('rouge_2'), 4),
        'avg_rouge_l':        round(avg('rouge_l'), 4),
        'exact_match_rate':   round(avg('exact_match'), 4),
        'avg_hungarian_ea':   round(hung_avg('hungarian_ea'), 4),
        'avg_hungarian_f1':   round(hung_avg('hungarian_f1'), 4),
        'avg_hungarian_prec': round(hung_avg('hungarian_prec'), 4),
        'avg_hungarian_rec':  round(hung_avg('hungarian_rec'), 4),
        'avg_hungarian_text': round(hung_avg('hungarian_text'), 4),
        'avg_hungarian_idx':  round(hung_avg('hungarian_idx'), 4),
    }


# ── open_app 등 GT action.type 기준 행 필터링 ────────────────────────────
ACTION_MARKER = "## Action\n"


def _gt_action_type(rec):
    """GT entry 의 user 메시지에서 ## Action 블록 type 을 추출."""
    text = rec["messages"][1]["value"]
    idx = text.find(ACTION_MARKER)
    if idx < 0:
        return None
    raw = text[idx + len(ACTION_MARKER):].strip()
    try:
        return json.loads(raw).get("type")
    except json.JSONDecodeError:
        return None


def _filter_pairs(gts, preds, exclude_action):
    """exclude_action 과 일치하는 GT 행을 양쪽에서 동시 drop."""
    if not exclude_action:
        return list(gts), list(preds)
    keep = [i for i, gt in enumerate(gts) if _gt_action_type(gt) != exclude_action]
    return [gts[i] for i in keep], [preds[i] for i in keep]


def _write_jsonl_idempotent(records, path):
    """이미 존재하면 no-op. 없으면 atomic 하게 jsonl 저장."""
    p = Path(path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _filtered_test_name(src_path, exclude_action):
    """data/MobiBench/implicit-world-modeling_stage1.jsonl + open_app
       → implicit-world-modeling_stage1_without_open_app.jsonl"""
    p = Path(src_path)
    return f"{p.stem}_without_{exclude_action}{p.suffix}"


def _predict_results_dict(metrics):
    """vllm_infer.py 가 만들어주는 predict_results.json 과 동일 schema."""
    return {
        "predict_bleu-4":  round(metrics["avg_bleu"]    * 100, 4),
        "predict_rouge-1": round(metrics["avg_rouge_1"] * 100, 4),
        "predict_rouge-2": round(metrics["avg_rouge_2"] * 100, 4),
        "predict_rouge-l": round(metrics["avg_rouge_l"] * 100, 4),
        "predict_total":   metrics["total"],
    }


def evaluate_stage1_predictions(test_path, pred_path):
    """Backward-compatible file-based entry point."""
    return evaluate_pairs(_load_jsonl(test_path), _load_jsonl(pred_path))


# ── CLI ──────────────────────────────────────────────────────────────────
def _print_metrics_row(label, metrics):
    print(
        f"[score:{label}] total={metrics['total']}  "
        f"f1={metrics['avg_hungarian_f1']:.4f}  "
        f"bleu={metrics['avg_bleu']:.4f}  "
        f"rouge-l={metrics['avg_rouge_l']:.4f}  "
        f"em={metrics['exact_match_rate']:.4f}"
    )


def _cmd_score(args):
    _lazy_deps()

    split_mode = bool(args.test_id or args.pred_id or args.test_ood or args.pred_ood)
    exclude = args.exclude_action or None

    # 필터된 jsonl 산출용 디렉토리 (exclude 가 set 일 때만 사용)
    test_out_dir = Path(args.filtered_test_dir) if args.filtered_test_dir else None
    pred_out_dir = Path(args.filtered_pred_dir) if args.filtered_pred_dir else None

    if split_mode:
        missing = [
            name for name, val in [
                ("--test-id", args.test_id), ("--pred-id", args.pred_id),
                ("--test-ood", args.test_ood), ("--pred-ood", args.pred_ood),
            ] if not val
        ]
        if missing:
            print(f"[score] ERROR: split mode needs {missing}", file=sys.stderr)
            return 2

        gt_id  = _load_jsonl(args.test_id)
        pr_id  = _load_jsonl(args.pred_id)
        gt_ood = _load_jsonl(args.test_ood)
        pr_ood = _load_jsonl(args.pred_ood)

        if exclude:
            gt_id,  pr_id  = _filter_pairs(gt_id,  pr_id,  exclude)
            gt_ood, pr_ood = _filter_pairs(gt_ood, pr_ood, exclude)
            if test_out_dir is not None:
                _write_jsonl_idempotent(gt_id,  test_out_dir / _filtered_test_name(args.test_id,  exclude))
                _write_jsonl_idempotent(gt_ood, test_out_dir / _filtered_test_name(args.test_ood, exclude))
            if pred_out_dir is not None:
                _write_jsonl_idempotent(pr_id,  pred_out_dir / "generated_predictions_id.jsonl")
                _write_jsonl_idempotent(pr_ood, pred_out_dir / "generated_predictions_ood.jsonl")

        m_id      = evaluate_pairs(gt_id, pr_id)
        m_ood     = evaluate_pairs(gt_ood, pr_ood)
        m_overall = evaluate_pairs(gt_id + gt_ood, pr_id + pr_ood)

        metrics = {
            "overall": m_overall,
            "in_domain": m_id,
            "out_of_domain": m_ood,
        }
        _print_metrics_row("overall", m_overall)
        _print_metrics_row("in_domain", m_id)
        _print_metrics_row("out_of_domain", m_ood)
        predict_results = _predict_results_dict(m_overall)
    else:
        if not (args.test and args.pred):
            print("[score] ERROR: --test and --pred required in single-pair mode",
                  file=sys.stderr)
            return 2
        gts   = _load_jsonl(args.test)
        preds = _load_jsonl(args.pred)
        if exclude:
            gts, preds = _filter_pairs(gts, preds, exclude)
            if test_out_dir is not None:
                _write_jsonl_idempotent(gts, test_out_dir / _filtered_test_name(args.test, exclude))
            if pred_out_dir is not None:
                _write_jsonl_idempotent(preds, pred_out_dir / "generated_predictions.jsonl")
        metrics = evaluate_pairs(gts, preds)
        _print_metrics_row("all", metrics)
        predict_results = _predict_results_dict(metrics)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[score] saved: {out_path}")

    # vllm_infer.py 가 정규 eval 산출에서 만들어주는 predict_results.json 과 동일 schema
    # 를 sibling 으로 함께 저장. 정규 eval 산출에 이미 있으면 덮어쓰지 않는다.
    pr_path = out_path.parent / "predict_results.json"
    if not pr_path.exists():
        with pr_path.open('w', encoding='utf-8') as f:
            json.dump(predict_results, f, ensure_ascii=False, indent=4)
        print(f"[score] saved: {pr_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Hungarian/BLEU/ROUGE evaluator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser(
        "score",
        help="Compute metrics. Single-pair (--test/--pred) or "
             "ID/OOD split (--test-id/--pred-id/--test-ood/--pred-ood).",
    )
    p_score.add_argument("--test", default=None, help="Single-pair: GT test jsonl")
    p_score.add_argument("--pred", default=None, help="Single-pair: prediction jsonl")
    p_score.add_argument("--test-id",  default=None, dest="test_id",  help="ID/OOD: in-domain GT")
    p_score.add_argument("--pred-id",  default=None, dest="pred_id",  help="ID/OOD: in-domain prediction")
    p_score.add_argument("--test-ood", default=None, dest="test_ood", help="ID/OOD: out-of-domain GT")
    p_score.add_argument("--pred-ood", default=None, dest="pred_ood", help="ID/OOD: out-of-domain prediction")
    p_score.add_argument("--output", required=True, help="Output metrics.json path")
    p_score.add_argument(
        "--exclude-action", default=None, dest="exclude_action",
        help="GT messages 의 ## Action 블록 type 이 이 값과 일치하는 행을 양쪽에서 동시 drop 후 채점 "
             "(예: open_app). 정규 eval 의 generated_predictions*.jsonl 을 그대로 입력으로 받아 "
             "추론 재실행 없이 필터 산출을 만든다.",
    )
    p_score.add_argument(
        "--filtered-test-dir", default=None, dest="filtered_test_dir",
        help="--exclude-action 과 함께. 필터된 GT jsonl 을 이 디렉토리에 "
             "{원본 stem}_without_{ACTION}.jsonl 로 idempotent 저장.",
    )
    p_score.add_argument(
        "--filtered-pred-dir", default=None, dest="filtered_pred_dir",
        help="--exclude-action 과 함께. 필터된 prediction jsonl 을 이 디렉토리에 "
             "generated_predictions{,_id,_ood}.jsonl 로 idempotent 저장.",
    )
    p_score.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
