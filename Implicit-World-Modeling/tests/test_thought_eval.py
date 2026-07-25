"""Regression tests for scripts/thought_eval.py — Stage 2 <thought> 유사도 평가기.

GT 는 generated_predictions*.jsonl 자체의 ``label`` 필드(LlamaFactory vllm_infer.py 의
배치 labels 디코딩 결과)에서 읽는다 — 별도 test jsonl 을 threading 하지 않는다.

임베딩(코사인) 은 무거운 모델 다운로드 없이 인터페이스만 모킹해 검증한다
(``_FakeEncoder`` — 결정론적 벡터를 반환하는 fake encoder). ROUGE-L(LCS 직접 구현)과
BLEU(sacrebleu) 는 알려진 쌍의 수기 계산값과 직접 대조하는 실구현 테스트다.

Run:
    pytest tests/test_thought_eval.py -v
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import sacrebleu as _sacrebleu

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

thought_eval = importlib.import_module("thought_eval")


# ── fixtures ────────────────────────────────────────────────────────────
def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class _FakeEncoder:
    """실 임베딩 모델 대체용 결정론적 인코더. 텍스트 → 3차원 벡터(정규화)."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            v = np.array([len(t), t.count("a"), t.count("b")], dtype=float)
            if normalize_embeddings and np.linalg.norm(v) > 0:
                v = v / np.linalg.norm(v)
            vecs.append(v)
        return np.array(vecs)


# ── <thought> 파싱 ──────────────────────────────────────────────────────
class TestExtractThought:
    def test_normal(self):
        text = "<thought>hello world</thought>\n<action>{}</action>"
        assert thought_eval._extract_thought(text) == "hello world"

    def test_missing(self):
        text = '<action>{"action":"click"}</action>'
        assert thought_eval._extract_thought(text) is None

    def test_multiline_dotall(self):
        text = "<thought>line one\nline two\nline three</thought>\n<action>{}</action>"
        assert thought_eval._extract_thought(text) == "line one\nline two\nline three"

    def test_empty_input(self):
        assert thought_eval._extract_thought("") is None
        assert thought_eval._extract_thought(None) is None

    def test_strips_whitespace(self):
        text = "<thought>  padded  </thought>"
        assert thought_eval._extract_thought(text) == "padded"


# ── ROUGE-L (LCS 직접 구현) — 수기 계산값 대조 ────────────────────────────
class TestRougeL:
    def test_exact_match(self):
        # LCS("the cat sat", "the cat sat") = 3, P=R=1.0 -> F1=1.0
        assert thought_eval.rouge_l_f1("the cat sat", "the cat sat") == pytest.approx(
            1.0
        )

    def test_no_overlap(self):
        assert thought_eval.rouge_l_f1("x y z", "a b c") == 0.0

    def test_hand_calc_subsequence(self):
        # hyp="the cat sat" (3 tok), ref="the cat sat on the mat" (6 tok)
        # LCS = "the cat sat" = 3 -> P=3/3=1.0, R=3/6=0.5
        # F1 = 2*1.0*0.5/(1.0+0.5) = 1/1.5 = 0.666666...
        got = thought_eval.rouge_l_f1("the cat sat", "the cat sat on the mat")
        assert got == pytest.approx(2 / 3, abs=1e-6)

    def test_hand_calc_reordered(self):
        # hyp="a b c" ref="c b a": 순서 보존 LCS 는 단일 토큰 하나뿐(길이 2 subsequence
        # 없음 -- a는 ref 뒤쪽, b는 ref 중간, c는 ref 앞쪽이라 어느 두 토큰도 순서가
        # hyp/ref 양쪽에서 동시에 유지되지 않는다). LCS=1 -> P=R=1/3
        # F1 = 2*(1/3)*(1/3)/(2/3) = (2/9)/(2/3) = 1/3
        got = thought_eval.rouge_l_f1("a b c", "c b a")
        assert got == pytest.approx(1 / 3, abs=1e-6)

    def test_empty_hyp_or_ref(self):
        assert thought_eval.rouge_l_f1("", "the cat sat") == 0.0
        assert thought_eval.rouge_l_f1("the cat sat", "") == 0.0
        assert thought_eval.rouge_l_f1("", "") == 0.0


# ── BLEU (sacrebleu, 0~1 정규화) ──────────────────────────────────────────
class TestBleu:
    def test_exact_match_is_one(self):
        # sacrebleu exact match -> 100.0 (부동소수점 오차 미세) -> /100 = 1.0
        got = thought_eval.sentence_bleu01(
            "the cat sat on the mat", "the cat sat on the mat", _sacrebleu
        )
        assert got == pytest.approx(1.0, abs=1e-6)

    def test_empty_hyp_or_ref_is_zero(self):
        assert thought_eval.sentence_bleu01("", "the cat sat", _sacrebleu) == 0.0
        assert thought_eval.sentence_bleu01("the cat sat", "", _sacrebleu) == 0.0

    def test_partial_match_in_unit_range(self):
        # 정확한 sacrebleu 부분점수는 exp-smoothing 내부 구현에 의존적이라 hand-calc
        # 하지 않고 0~1 스케일 범위 및 완전 불일치보다 우월함만 확인한다.
        got = thought_eval.sentence_bleu01(
            "the cat sat on the mat", "the cat sat on a mat today", _sacrebleu
        )
        assert 0.0 < got < 1.0

    def test_completely_different_is_low(self):
        got = thought_eval.sentence_bleu01(
            "completely unrelated text about nothing",
            "the cat sat on the mat",
            _sacrebleu,
        )
        assert 0.0 <= got < 0.3


class TestLoadBleuModuleDegradation:
    def test_load_failure_returns_none(self, monkeypatch):
        # sys.modules[name] = None 이면 그 이름의 import 는 ImportError 로 실패한다
        # (파이썬 표준 동작) -- sacrebleu 미설치 상황을 네트워크 없이 재현.
        monkeypatch.setitem(sys.modules, "sacrebleu", None)
        assert thought_eval.load_bleu_module() is None

    def test_load_success_returns_module(self):
        mod = thought_eval.load_bleu_module()
        assert mod is not None
        assert hasattr(mod, "sentence_bleu")


# ── 임베딩 코사인 (인터페이스 모킹) ────────────────────────────────────────
class TestCosineScores:
    def test_identical_text_is_one(self):
        enc = _FakeEncoder()
        scores = thought_eval.cosine_scores(enc, ["hello aab"], ["hello aab"])
        assert scores == pytest.approx([1.0], abs=1e-6)

    def test_different_text_matches_manual_dot_product(self):
        enc = _FakeEncoder()
        hyp, ref = "aab", "xyz"
        [got] = thought_eval.cosine_scores(enc, [hyp], [ref])
        hv = np.array([len(hyp), hyp.count("a"), hyp.count("b")], dtype=float)
        rv = np.array([len(ref), ref.count("a"), ref.count("b")], dtype=float)
        hv = hv / np.linalg.norm(hv)
        rv = rv / np.linalg.norm(rv)
        expected = float(np.dot(hv, rv))
        assert got == pytest.approx(expected, abs=1e-6)

    def test_empty_input_returns_empty(self):
        enc = _FakeEncoder()
        assert thought_eval.cosine_scores(enc, [], []) == []


class TestLoadEncoderDegradation:
    def test_load_failure_returns_none(self, monkeypatch):
        fake_module = types.ModuleType("sentence_transformers")

        class _Boom:
            def __init__(self, *a, **kw):
                raise RuntimeError("boom: no network")

        fake_module.SentenceTransformer = _Boom
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
        assert thought_eval.load_encoder("whatever-model") is None


# ── GT thought 존재 여부 게이트 ────────────────────────────────────────────
class TestHasAnyGtThought:
    def test_true_when_present(self):
        entries = [{"label": "<thought>hi</thought>\n<action>{}</action>"}]
        assert thought_eval._has_any_gt_thought([entries]) is True

    def test_false_when_absent(self):
        entries = [{"label": '{"action_type":"click","index":"1"}'}]
        assert thought_eval._has_any_gt_thought([entries]) is False

    def test_false_on_empty(self):
        assert thought_eval._has_any_gt_thought([[]]) is False


# ── 집계 (evaluate_thoughts) ───────────────────────────────────────────────
class TestEvaluateThoughts:
    def _entries(self):
        return [
            {
                "predict": "<thought>aab</thought>\n<action>{}</action>",
                "label": "<thought>aab</thought>\n<action>{}</action>",
            },
            {
                # 예측에 thought 부재 -> missing
                "predict": "<action>{}</action>",
                "label": "<thought>completely different words here</thought>\n<action>{}</action>",
            },
            {
                # GT 에 thought 부재 -> n 에서 제외
                "predict": "<thought>whatever</thought>",
                "label": '{"action_type":"click"}',
            },
        ]

    def test_n_excludes_gt_missing_rows(self):
        result = thought_eval.evaluate_thoughts(
            self._entries(), encoder=None, bleu_module=_sacrebleu
        )
        assert result["n"] == 2  # 3번째 행(GT thought 없음) 제외

    def test_missing_thought_counts_pred_missing_only(self):
        result = thought_eval.evaluate_thoughts(
            self._entries(), encoder=None, bleu_module=_sacrebleu
        )
        assert result["missing_thought_n"] == 1
        assert result["missing_thought_ratio"] == pytest.approx(0.5)

    def test_cosine_none_without_encoder(self):
        result = thought_eval.evaluate_thoughts(
            self._entries(), encoder=None, bleu_module=_sacrebleu
        )
        assert result["mean_cosine"] is None
        assert result["std_cosine"] is None

    def test_cosine_populated_with_fake_encoder(self):
        result = thought_eval.evaluate_thoughts(
            self._entries(), encoder=_FakeEncoder(), bleu_module=_sacrebleu
        )
        assert result["mean_cosine"] is not None
        assert result["std_cosine"] is not None
        assert 0.0 <= result["mean_cosine"] <= 1.0

    def test_missing_row_gets_zero_for_all_metrics(self):
        # 2번째 행(missing) 은 rouge/bleu/cosine 모두 0 이어야 하므로, 첫 행이
        # exact-match(1.0) 인 것과 평균이 정확히 절반이 되는지로 간접 검증.
        result = thought_eval.evaluate_thoughts(
            self._entries(), encoder=None, bleu_module=_sacrebleu
        )
        assert result["mean_rouge_l"] == pytest.approx(0.5, abs=1e-6)
        assert result["mean_bleu"] == pytest.approx(0.5, abs=1e-6)

    def test_empty_entries(self):
        result = thought_eval.evaluate_thoughts(
            [], encoder=None, bleu_module=_sacrebleu
        )
        assert result["n"] == 0
        assert result["missing_thought_ratio"] == 0.0
        assert result["mean_rouge_l"] == 0.0

    def test_bleu_none_without_bleu_module(self):
        # bleu_module=None (sacrebleu 로드 실패/미설치) -> mean_bleu/std_bleu 만 null,
        # 나머지 지표(n, missing, rouge_l)는 정상 계산 (인코더 가드와 대칭 동작).
        result = thought_eval.evaluate_thoughts(
            self._entries(), encoder=None, bleu_module=None
        )
        assert result["mean_bleu"] is None
        assert result["std_bleu"] is None
        assert result["n"] == 2
        assert result["mean_rouge_l"] == pytest.approx(0.5, abs=1e-6)


# ── CLI 통합 (main, 인코더는 fake 로 monkeypatch) ─────────────────────────
class TestMainCli:
    def test_single_mode_writes_output(self, tmp_path, monkeypatch):
        pred_path = tmp_path / "generated_predictions.jsonl"
        _write_jsonl(
            pred_path,
            [
                {
                    "predict": "<thought>hi there</thought>\n<action>{}</action>",
                    "label": "<thought>hi there</thought>\n<action>{}</action>",
                }
            ],
        )
        out_path = tmp_path / "thought_metrics.json"
        monkeypatch.setattr(thought_eval, "load_encoder", lambda model: _FakeEncoder())
        monkeypatch.setattr(
            sys,
            "argv",
            ["thought_eval.py", "--pred", str(pred_path), "--output", str(out_path)],
        )
        rc = thought_eval.main()
        assert rc == 0
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["n"] == 1
        assert "mean_cosine" in data and "mean_rouge_l" in data and "mean_bleu" in data

    def test_split_mode_has_three_sections(self, tmp_path, monkeypatch):
        id_path = tmp_path / "generated_predictions_id.jsonl"
        ood_path = tmp_path / "generated_predictions_ood.jsonl"
        row = {
            "predict": "<thought>hi there</thought>\n<action>{}</action>",
            "label": "<thought>hi there</thought>\n<action>{}</action>",
        }
        _write_jsonl(id_path, [row])
        _write_jsonl(ood_path, [row])
        out_path = tmp_path / "thought_metrics.json"
        monkeypatch.setattr(thought_eval, "load_encoder", lambda model: _FakeEncoder())
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "thought_eval.py",
                "--pred-id",
                str(id_path),
                "--pred-ood",
                str(ood_path),
                "--output",
                str(out_path),
            ],
        )
        rc = thought_eval.main()
        assert rc == 0
        data = json.loads(out_path.read_text())
        assert set(data.keys()) == {"overall", "in_domain", "out_of_domain"}
        assert data["overall"]["n"] == 2

    def test_no_op_when_no_gt_thought(self, tmp_path, monkeypatch):
        # EXP01~04 시뮬레이션: label 에 <thought> 가 전혀 없음.
        pred_path = tmp_path / "generated_predictions.jsonl"
        _write_jsonl(
            pred_path,
            [
                {
                    "predict": '{"action_type":"click","index":"1"}',
                    "label": '{"action_type":"click","index":"1"}',
                }
            ],
        )
        out_path = tmp_path / "thought_metrics.json"

        def _boom(model=None):
            raise AssertionError(
                "no-op 게이트를 지나면 안 된다 (heavy import 호출 금지)"
            )

        monkeypatch.setattr(thought_eval, "load_encoder", _boom)
        monkeypatch.setattr(thought_eval, "load_bleu_module", _boom)
        monkeypatch.setattr(
            sys,
            "argv",
            ["thought_eval.py", "--pred", str(pred_path), "--output", str(out_path)],
        )
        rc = thought_eval.main()
        assert rc == 0
        assert not out_path.exists()

    def test_bleu_module_load_failure_degrades_not_crashes(self, tmp_path, monkeypatch):
        # sacrebleu 미설치/로드 실패 시 mean_bleu 는 null 로 강등되고, rc==0 유지,
        # rouge_l 등 나머지 지표는 정상 계산된다 (U4-fix 목표: crash 방지).
        pred_path = tmp_path / "generated_predictions.jsonl"
        _write_jsonl(
            pred_path,
            [
                {
                    "predict": "<thought>hi there</thought>\n<action>{}</action>",
                    "label": "<thought>hi there</thought>\n<action>{}</action>",
                }
            ],
        )
        out_path = tmp_path / "thought_metrics.json"
        monkeypatch.setattr(thought_eval, "load_encoder", lambda model: None)
        monkeypatch.setattr(thought_eval, "load_bleu_module", lambda: None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["thought_eval.py", "--pred", str(pred_path), "--output", str(out_path)],
        )
        rc = thought_eval.main()
        assert rc == 0
        data = json.loads(out_path.read_text())
        assert data["mean_bleu"] is None
        assert data["std_bleu"] is None
        assert data["mean_rouge_l"] == pytest.approx(1.0)

    def test_split_mode_missing_pred_ood_errors(self, tmp_path, monkeypatch, capsys):
        id_path = tmp_path / "generated_predictions_id.jsonl"
        _write_jsonl(id_path, [{"predict": "x", "label": "<thought>y</thought>"}])
        out_path = tmp_path / "thought_metrics.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["thought_eval.py", "--pred-id", str(id_path), "--output", str(out_path)],
        )
        rc = thought_eval.main()
        assert rc == 2
