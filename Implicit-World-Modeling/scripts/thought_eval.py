#!/usr/bin/env python3
"""
Stage 2 <thought> 유사도 평가기 (EXP07 스펙 "<thought> 부분 GT 와 유사도 비교" 구현).

``_action_eval.py`` 가 채점하는 것과 같은 ``generated_predictions*.jsonl`` 을 입력으로
받는다. 별도 test jsonl 은 필요 없다 — LlamaFactory ``vllm_infer.py`` 가 쓰는 각 행의
``label`` 필드가 이미 GT (배치 labels 를 IGNORE_INDEX 필터링 후 디코딩한 것) 이므로,
GT/예측이 같은 파일의 같은 행에 있어 cross-file row misalignment 위험이 없다.

지표
----
정본  : 임베딩 코사인 (``sentence-transformers/all-MiniLM-L6-v2`` 기본, ``--embed-model``
        로 교체 가능). 인코더 로드 실패(미설치/오프라인) 시 이 지표만 null 로 비고,
        나머지 지표는 정상 계산하며 exit code 는 그대로 0 이다 (graceful degradation).
보조1 : ROUGE-L F1 — LCS 직접 구현 (whitespace 토큰, β=1). ``rouge_score`` 비의존.
보조2 : BLEU — ``sacrebleu.sentence_bleu`` (exp smoothing), 0~100 원점수를 100 으로
        나눠 0~1 스케일로 통일해 코사인/ROUGE-L 과 나란히 비교 가능하게 한다. sacrebleu
        모듈도 인코더와 같은 패턴으로 1회 로드 시도 후 가드한다 (미설치 시 mean_bleu 를
        null 로 기록, 행 단위 재시도·크래시 없음).

예측에서 ``<thought>`` 를 못 찾으면 세 지표 모두 0 으로 처리하고 missing_thought 로
집계한다. GT(label) 쪽에 ``<thought>`` 가 없는 행은 비교 대상이 성립하지 않으므로 n 에서
제외한다 (thought 가 없는 실험군에서 조용히 0 점 행이 쌓이는 것을 막기 위함).

No-op 게이트
------------
입력 파일(들) 전체에서 GT(label) 쪽 ``<thought>`` 가 단 하나도 없으면 (EXP01~04 처럼
thought 필드 자체가 없는 데이터셋) — sentence_transformers/sacrebleu 같은 무거운
import 조차 하지 않고 thought_metrics.json 을 만들지 않은 채 exit 0 으로 끝낸다.
stage2_eval.sh 의 1-hook 이 EXP 종류를 가리지 않고 무조건 호출되므로, 이 게이트가
thought 없는 실험군에서 완전히 무해하게 만든다.

Examples
--------
  # 단일 파일 (MB 등 overall-only 데이터셋)
  python scripts/thought_eval.py \\
      --pred   .../generated_predictions.jsonl \\
      --output .../thought_metrics.json

  # ID/OOD split (EXP05/06/07 등) — overall/in_domain/out_of_domain 3-섹션
  python scripts/thought_eval.py \\
      --pred-id  .../generated_predictions_id.jsonl \\
      --pred-ood .../generated_predictions_ood.jsonl \\
      --output   .../thought_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── I/O ──────────────────────────────────────────────────────────────────
def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_thought(text):
    """첫 <thought>...</thought> 매치의 내용을 strip 해 반환. 없으면 None."""
    match = _THOUGHT_RE.search(text or "")
    if match is None:
        return None
    return match.group(1).strip()


def _pred_text(entry):
    return entry.get("predict", entry.get("output", ""))


def _gt_text(entry):
    """vllm_infer.py 산출물의 GT (batch labels 디코딩 결과)."""
    return entry.get("label", "")


def _has_any_gt_thought(entries_list):
    """입력 행들(복수 파일 합산) 중 GT(label) 에 <thought> 가 하나라도 있는지."""
    for entries in entries_list:
        for entry in entries:
            if _extract_thought(_gt_text(entry)) is not None:
                return True
    return False


# ── ROUGE-L (LCS 직접 구현, β=1) ─────────────────────────────────────────
def _lcs_length(a_tokens, b_tokens):
    n, m = len(a_tokens), len(b_tokens)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = dp[j] if dp[j] >= dp[j - 1] else dp[j - 1]
            prev = tmp
    return dp[m]


def rouge_l_f1(hyp, ref):
    """whitespace 토큰 기준 LCS 기반 ROUGE-L F1 (β=1). hyp/ref 둘 다 비면 0.0."""
    hyp_tokens = hyp.split()
    ref_tokens = ref.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ── BLEU (sacrebleu sentence-level, 0~1 로 정규화) ───────────────────────
def load_bleu_module():
    """sacrebleu 모듈 로드. 실패(미설치 등) 시 None + stderr 경고 (인코더와 대칭 패턴).
    호출부는 행 단위로 재시도하지 않고 이 1회 판정 결과를 그대로 사용한다."""
    try:
        import sacrebleu

        return sacrebleu
    except Exception as exc:  # noqa: BLE001 - BLEU 는 보조 실패해도 파이프라인은 계속
        print(
            f"[warn] sacrebleu 로드 실패: {exc}. "
            "mean_bleu 는 null 로 기록하고 나머지 지표는 정상 계산합니다.",
            file=sys.stderr,
        )
        return None


def sentence_bleu01(hyp, ref, bleu_module):
    """bleu_module.sentence_bleu (exp smoothing) 의 0~100 점수를 0~1 로 정규화.
    bleu_module 은 load_bleu_module() 이 반환한 sacrebleu 모듈 (호출 전 가드 필수)."""
    if not hyp.strip() or not ref.strip():
        return 0.0
    score = bleu_module.sentence_bleu(hyp, [ref], smooth_method="exp")
    return score.score / 100.0


# ── 임베딩 코사인 (정본) ──────────────────────────────────────────────────
def load_encoder(model_name):
    """SentenceTransformer 인코더 로드. 실패(미설치/오프라인 등) 시 None + stderr 경고."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 - 임베딩은 보조 실패해도 파이프라인은 계속
        print(
            f"[warn] embedding encoder 로드 실패 ({model_name}): {exc}. "
            "mean_cosine 은 null 로 기록하고 ROUGE-L/BLEU 는 정상 계산합니다.",
            file=sys.stderr,
        )
        return None


def cosine_scores(encoder, hyps, refs):
    """encoder 로 hyps/refs 를 배치 인코딩해 행별 코사인 유사도 리스트를 반환."""
    if not hyps:
        return []
    import numpy as np

    embs = encoder.encode(
        hyps + refs, normalize_embeddings=True, show_progress_bar=False
    )
    embs = np.asarray(embs)
    n = len(hyps)
    hyp_embs, ref_embs = embs[:n], embs[n:]
    return [float(np.dot(h, r)) for h, r in zip(hyp_embs, ref_embs, strict=False)]


# ── 행 단위 채점 + 집계 ───────────────────────────────────────────────────
def _score_rows(entries, bleu_module):
    """GT(label) 에 thought 가 있는 행만 남겨 (missing, hyp, ref, rouge_l[, bleu]) 로 채점.
    cosine 은 아직 계산하지 않는다 (배치 처리를 위해 evaluate_thoughts 에서 별도 수행).
    bleu_module 이 None (로드 실패/미설치) 이면 bleu 키 자체를 채우지 않는다 — 행 단위
    재시도 없이 evaluate_thoughts 가 최종 집계에서 null 처리한다."""
    scored = []
    for entry in entries:
        gt_thought = _extract_thought(_gt_text(entry))
        if gt_thought is None:
            continue  # GT 자체에 thought 없음 → 비교 불성립, n 제외
        pred_thought = _extract_thought(_pred_text(entry))
        missing = pred_thought is None
        row = {
            "missing": missing,
            "hyp": pred_thought or "",
            "ref": gt_thought,
            "rouge_l": 0.0 if missing else rouge_l_f1(pred_thought, gt_thought),
        }
        if bleu_module is not None:
            row["bleu"] = (
                0.0
                if missing
                else sentence_bleu01(pred_thought, gt_thought, bleu_module)
            )
        scored.append(row)
    return scored


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return round(mean, 4), round(std, 4)


def evaluate_thoughts(entries, encoder, bleu_module):
    """단일 (split 안 된) entries 리스트에 대해 thought 유사도 지표 집계.

    반환 dict 는 n / missing_thought_n / missing_thought_ratio /
    mean_cosine·std_cosine (encoder 없으면 둘 다 None) /
    mean_rouge_l·std_rouge_l / mean_bleu·std_bleu (bleu_module 없으면 둘 다 None).
    """
    scored = _score_rows(entries, bleu_module)
    n = len(scored)
    missing_n = sum(1 for s in scored if s["missing"])

    non_missing = [s for s in scored if not s["missing"]]
    if encoder is not None:
        cos = cosine_scores(
            encoder, [s["hyp"] for s in non_missing], [s["ref"] for s in non_missing]
        )
        for s, c in zip(non_missing, cos, strict=False):
            s["cosine"] = c
        for s in scored:
            if s["missing"]:
                s["cosine"] = 0.0
        mean_cos, std_cos = _mean_std([s["cosine"] for s in scored])
    else:
        mean_cos, std_cos = None, None

    mean_rouge, std_rouge = _mean_std([s["rouge_l"] for s in scored])

    if bleu_module is not None:
        mean_bleu, std_bleu = _mean_std([s["bleu"] for s in scored])
    else:
        mean_bleu, std_bleu = None, None

    return {
        "n": n,
        "missing_thought_n": missing_n,
        "missing_thought_ratio": round(missing_n / n, 4) if n else 0.0,
        "mean_cosine": mean_cos,
        "std_cosine": std_cos,
        "mean_rouge_l": mean_rouge,
        "std_rouge_l": std_rouge,
        "mean_bleu": mean_bleu,
        "std_bleu": std_bleu,
    }


# ── CLI ──────────────────────────────────────────────────────────────────
def _print_row(label, metrics):
    cos = "n/a" if metrics["mean_cosine"] is None else f"{metrics['mean_cosine']:.4f}"
    bleu = "n/a" if metrics["mean_bleu"] is None else f"{metrics['mean_bleu']:.4f}"
    print(
        f"[thought:{label}] n={metrics['n']}  "
        f"missing={metrics['missing_thought_ratio']:.2%}  "
        f"cosine={cos}  "
        f"rouge_l={metrics['mean_rouge_l']:.4f}  "
        f"bleu={bleu}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Stage 2 <thought> 유사도 평가기 (임베딩 코사인 정본 + ROUGE-L/BLEU 보조)"
    )
    p.add_argument(
        "--pred", default=None, help="단일 pair 모드 (예: MB, overall only)."
    )
    p.add_argument("--pred-id", default=None, dest="pred_id", help="ID/OOD split 모드.")
    p.add_argument(
        "--pred-ood", default=None, dest="pred_ood", help="ID/OOD split 모드."
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        dest="embed_model",
        help=f"코사인 정본 지표에 쓸 sentence-transformers 모델 (기본 {DEFAULT_EMBED_MODEL}).",
    )
    args = p.parse_args()

    split_mode = bool(args.pred_id or args.pred_ood)
    if split_mode:
        missing = [
            name
            for name, val in [
                ("--pred-id", args.pred_id),
                ("--pred-ood", args.pred_ood),
            ]
            if not val
        ]
        if missing:
            print(f"[thought] ERROR: split mode needs {missing}", file=sys.stderr)
            return 2
        entries_id = _load_jsonl(args.pred_id)
        entries_ood = _load_jsonl(args.pred_ood)
        all_entries = [entries_id, entries_ood]
    else:
        if not args.pred:
            print(
                "[thought] ERROR: --pred required in single-pair mode", file=sys.stderr
            )
            return 2
        entries_single = _load_jsonl(args.pred)
        all_entries = [entries_single]

    # No-op 게이트: GT(label) 어디에도 <thought> 가 없으면 (EXP01~04 등) 무거운 import
    # 조차 하지 않고 파일을 만들지 않은 채 무해하게 종료한다.
    if not _has_any_gt_thought(all_entries):
        print(
            "[thought] no-op: 입력에 <thought> GT 가 없습니다 (thought 미사용 데이터셋으로 "
            "판단) — thought_metrics.json 을 생성하지 않고 종료합니다.",
            file=sys.stderr,
        )
        return 0

    encoder = load_encoder(args.embed_model)
    bleu_module = load_bleu_module()

    if split_mode:
        m_id = evaluate_thoughts(entries_id, encoder, bleu_module)
        m_ood = evaluate_thoughts(entries_ood, encoder, bleu_module)
        m_overall = evaluate_thoughts(entries_id + entries_ood, encoder, bleu_module)
        metrics = {"overall": m_overall, "in_domain": m_id, "out_of_domain": m_ood}
        _print_row("overall", m_overall)
        _print_row("in_domain", m_id)
        _print_row("out_of_domain", m_ood)
    else:
        metrics = evaluate_thoughts(entries_single, encoder, bleu_module)
        _print_row("all", metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[thought] saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
