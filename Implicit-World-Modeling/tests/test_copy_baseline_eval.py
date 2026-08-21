"""scripts/_copy_baseline_eval.py — 복사기(copy baseline) 기준선 채점기 테스트.

복사기 점수는 `(current, gt)` 만의 함수라 **닫힌 형태로 검증된다** — 데이터가 무엇이든
`copy_exact_rate == 1.0`, `avg_change_f1_strict == 0.0` 같은 등식이 성립해야 한다.
그래서 여기서는 "값이 그럴듯한가"가 아니라 그 등식들을 직접 찌른다.

이 채점기의 조용한 실패 모드는 셋이다:
  1. `_lazy_deps()` 미호출 → `compute_hungarian_acc` 의 except 가 예외를 삼켜 전 행 0점.
  2. gain 을 두 집계의 뺄셈으로 만들기 → 정의행 집합이 다른 두 평균을 빼게 된다.
  3. `_aggregate_hungarian` 이 정본(`_he.evaluate_pairs`)에서 드리프트 → model 섹션이
     기존 `hungarian_metrics.json` 과 어긋나고, 그러면 gain 의 피감수가 딴 세계의 수다.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_he = __import__("_hungarian_eval")
_sd = __import__("_state_diff_eval")
_cb = __import__("_copy_baseline_eval")


# ── 합성 픽스처 ──────────────────────────────────────────────────────────
# cur : node + button(OK) + p("unread inbox 3") + img(gone)
# gt  : node + button(OK) + p("unread inbox 7") + p("brand new banner")
#       → UNCHANGED(button) · MODIFIED(p) · ADDED(p) · DELETED(img) 각 1
#
# ⚠️ MODIFIED 쪽 텍스트는 **일부러 부분 겹침**이다. `_state_diff_eval._PROBE` 는
# hello → world 처럼 텍스트를 통째로 바꾸는데, 그러면 `_text_sim` 이 0 → cost 가 정확히
# `W_TEXT`(1.5)가 되고 index 모드의 임계도 1.5 라 판정(`cost < threshold`)에서 **떨어진다**
# — index 모드에서는 MODIFIED 가 한 건도 안 생긴다. 복사기의 recall 층 항등식
# (아래 test_copier_recall_layers_are_closed_form)은 세 층이 다 있어야 의미가 있으므로
# 두 모드에서 같은 유형이 나오는 픽스처를 따로 잡는다.
# `maxdel`(= current 를 하나도 재현하지 않는 비어 있지 않은 예측)은 `_PROBE` 것을 쓴다 —
# `select` 태그라 cur/gt 의 button/p/img 와 tag cost W_TAG=3.0 으로 확실히 떨어진다.
CUR = {
    "index": (
        '<node index="0"><button index="1" aria-label="OK"/>'
        '<p index="2">unread inbox 3</p><img index="4">gone</img></node>'
    ),
    "pos": (
        '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<p bounds="[6,6][9,9]" point="[7,7]">unread inbox 3</p>'
        '<img bounds="[6,30][9,34]" point="[7,32]">gone</img></node>'
    ),
}
GT = {
    "index": (
        '<node index="0"><button index="1" aria-label="OK"/>'
        '<p index="2">unread inbox 7</p><p index="3">brand new banner</p></node>'
    ),
    "pos": (
        '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<p bounds="[6,6][9,9]" point="[7,7]">unread inbox 7</p>'
        '<p bounds="[6,20][9,24]" point="[7,22]">brand new banner</p></node>'
    ),
}


def _prompt(cur: str, mode: str) -> str:
    """current state 를 담은 렌더 프롬프트. 두 계열 중 모드에 맞는 쪽을 쓴다."""
    if mode == "index":
        return (
            "system\nrole\nuser\n\n## Current State\n"
            + cur
            + '\n\n## Action\n{"action_type":"click","index":"1"}\nassistant\n'
        )
    return (
        "system\nrole\nuser\nCurrent UI State:\n"
        + cur
        + '\n\nAction:\n<action>{"action": "click", "coordinate": [3, 3]}</action>\n'
        "assistant\n"
    )


def _fixture(mode: str):
    """(gt_entries, pred_entries). 6행이고 각 행의 역할이 아래 주석에 고정돼 있다.

    행 구성은 gain 교집합을 만들기 위한 것이다 — 모델과 복사기의 **정의행이 갈리는**
    행(3·5)이 반드시 들어 있어야 "두 집계의 뺄셈"과 "행 단위 교집합"이 구분된다.
    """
    cur, gt, maxdel = CUR[mode], GT[mode], _sd._PROBE[mode]["maxdel"]
    rows = [
        # (current, gt, 모델 예측, 설명)
        (cur, gt, gt),  # 0 완벽 예측
        (cur, gt, cur),  # 1 모델도 복사 — gain 이 0 이어야 하는 행
        (cur, cur, cur),  # 2 변화 없는 행 + 복사 → no_change_acc 1.0
        (cur, gt, ""),  # 3 생성 실패 — 모델 쪽만 change_* 가 0.0 으로 정의된다
        (cur, gt, maxdel),  # 4 최대 삭제 예측
        (cur, cur, gt),  # 5 변화 없는 행인데 변화를 지어냄 — 복사기는 change_* 미정의
    ]
    gts = [
        {"messages": [{"value": "sys"}, {"value": "u"}, {"value": g}]}
        for _, g, _ in rows
    ]
    preds = [{"prompt": _prompt(c, mode), "predict": pr} for c, _, pr in rows]
    return gts, preds


MODES = ("index", "pos")


class TestCopyBaselineInvariants(unittest.TestCase):
    """복사기 점수의 닫힌 형태 6종. 어긋나면 값이 아니라 배선이 깨진 것이다."""

    def test_invariants_hold_in_both_modes(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                gts, preds = _fixture(mode)
                _, copy_rows = _cb.score_rows(gts, preds, mode)
                cb = _cb.aggregate(copy_rows, mode)
                # (1) 예측 슬롯에 current 를 그대로 넣었는가
                self.assertEqual(cb["copy_exact_rate"], 1.0)
                # (2) 예측 요소가 전부 current 와 매칭되는가
                self.assertEqual(cb["avg_copy_rate_pred"], 1.0)
                # (3) copy_excess = 1 − copy_rate_gt (분모 차이·행별 round 때문에 근사)
                self.assertAlmostEqual(
                    cb["avg_copy_excess"],
                    1.0 - cb["avg_copy_rate_gt"],
                    delta=_cb.INVARIANT_TOL,
                )
                # (4) 복사기는 어떤 변화도 주장하지 않는다
                self.assertEqual(cb["avg_change_f1_strict"], 0.0)
                # (5) 변화 없는 행에서는 복사가 정답
                self.assertGreater(
                    cb["n_no_change_acc"], 0, "픽스처에 무변화 행이 있어야 한다"
                )
                self.assertEqual(cb["avg_no_change_acc"], 1.0)
                # (6) 0.0 이면 지표 해석 문제가 아니라 `_lazy_deps` 배선이 죽은 것이다
                self.assertGreater(cb["avg_hungarian_f1"], 0.0)
                # 위 6종을 채점기 스스로도 검사한다 (build_section 이 매번 부른다)
                _cb.assert_copy_baseline_invariants(cb, mode)

    def test_copier_recall_layers_are_closed_form(self):
        """복사기의 recall 3층은 닫힌 형태로 유도된다 — 불변식 6종과 **독립적인** 배선 증거.

        `pred == cur` 이면 recall 의 매칭 `_hungarian_match(pred, gt)` 와 diff 분류의
        `_hungarian_match(cur, gt)` 가 같은 함수·같은 입력이라 짝이 동일하다. 그러면
        hit 집합 == "UNCHANGED ∪ MODIFIED" 이므로 두 층은 정확히 1.0, ADDED 층은 0.0
        이어야 한다. 어느 하나라도 어긋나면 두 매칭이 갈린 것이다 (임계·element 집합
        스위치가 한쪽에만 걸리면 정확히 이렇게 깨진다).
        """
        for mode in MODES:
            with self.subTest(mode=mode):
                gts, preds = _fixture(mode)
                _, copy_rows = _cb.score_rows(gts, preds, mode)
                cb = _cb.aggregate(copy_rows, mode)
                self.assertEqual(cb["avg_unchanged_recall"], 1.0)
                self.assertEqual(cb["avg_modified_recall"], 1.0)
                self.assertEqual(cb["avg_added_recall"], 0.0)
                self.assertGreater(
                    cb["n_modified_recall"], 0, "픽스처에 MODIFIED 가 있어야 한다"
                )
                self.assertGreater(
                    cb["n_added_recall"], 0, "픽스처에 ADDED 가 있어야 한다"
                )

    def test_copier_beats_nothing_on_the_change_axis(self):
        """복사기는 `change_f1_loose` 도 0 이어야 한다 — strict 만 0 이면 τ 게이트만 본 것."""
        for mode in MODES:
            with self.subTest(mode=mode):
                gts, preds = _fixture(mode)
                _, copy_rows = _cb.score_rows(gts, preds, mode)
                cb = _cb.aggregate(copy_rows, mode)
                self.assertEqual(cb["avg_change_f1_loose"], 0.0)


class TestInvariantGuardActuallyFires(unittest.TestCase):
    """가드가 실제로 무는지 — 통과만 확인하면 `pass` 로 바꿔도 테스트가 초록이다."""

    def _base(self, mode="index"):
        gts, preds = _fixture(mode)
        _, copy_rows = _cb.score_rows(gts, preds, mode)
        return _cb.aggregate(copy_rows, mode)

    def test_broken_copy_exact_is_caught(self):
        m = {**self._base(), "copy_exact_rate": 0.5}
        with self.assertRaises(_cb.CopyBaselineError):
            _cb.assert_copy_baseline_invariants(m)

    def test_nonzero_change_f1_is_caught(self):
        m = {**self._base(), "avg_change_f1_strict": 0.1}
        with self.assertRaises(_cb.CopyBaselineError):
            _cb.assert_copy_baseline_invariants(m)

    def test_zero_hungarian_f1_blames_the_wiring_not_the_metric(self):
        """전 행 0점은 이 리포에서 실제로 물린 함정이라 메시지가 원인을 짚어야 한다."""
        m = {**self._base(), "avg_hungarian_f1": 0.0}
        with self.assertRaises(_cb.CopyBaselineError) as ctx:
            _cb.assert_copy_baseline_invariants(m)
        self.assertIn("_lazy_deps", str(ctx.exception))

    def test_excess_identity_tolerance_is_not_infinite(self):
        m = {**self._base(), "avg_copy_excess": 0.0, "avg_copy_rate_gt": 0.5}
        with self.assertRaises(_cb.CopyBaselineError):
            _cb.assert_copy_baseline_invariants(m)

    def test_empty_denominator_does_not_fire_falsely(self):
        """`_sd.aggregate` 는 정의행 0 일 때 None 이 아니라 0.0 을 낸다 — 그건 위반이 아니다."""
        m = {**self._base(), "n_no_change_acc": 0, "avg_no_change_acc": 0.0}
        _cb.assert_copy_baseline_invariants(m)  # 예외 없이 통과해야 한다

    def test_broken_recall_layer_identity_is_caught(self):
        """recall 3층 항등식이 가드에도 박혀 있는지.

        `test_copier_recall_layers_are_closed_form` 은 값이 맞는지만 본다 — 가드가
        비어 있어도 통과한다. 두 매칭이 갈리는 회귀(임계·element 집합 스위치가 한쪽에만
        걸리는 경우)는 **채점 중에** 죽어야 하므로 여기서 무는지 따로 확인한다.
        """
        for key, bad in (
            ("avg_modified_recall", 0.9),
            ("avg_unchanged_recall", 0.9),
            ("avg_added_recall", 0.1),
        ):
            with self.subTest(key=key):
                with self.assertRaises(_cb.CopyBaselineError) as ctx:
                    _cb.assert_copy_baseline_invariants({**self._base(), key: bad})
                self.assertIn(key, str(ctx.exception))

    def test_recall_layer_identity_skips_empty_layers(self):
        """그 층이 없는 test set(`n_*` == 0)에서는 물지 않는다 — 0.0 은 위반이 아니다."""
        m = {**self._base(), "n_added_recall": 0, "avg_added_recall": 0.0,
             "n_modified_recall": 0, "avg_modified_recall": 0.0}
        _cb.assert_copy_baseline_invariants(m)


class TestHungarianAggregateParity(unittest.TestCase):
    """`_aggregate_hungarian` 은 이 모듈이 유일하게 **다시 쓴** 코드다.

    드리프트하면 model 섹션이 기존 `hungarian_metrics.json` 과 어긋나고, 그러면 gain 의
    피감수가 딴 세계의 수가 된다. 정본과 직접 비교해 못박는다.
    """

    def test_matches_hungarian_evaluate_pairs(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                gts, preds = _fixture(mode)
                model_rows, _ = _cb.score_rows(gts, preds, mode)
                self.assertEqual(
                    _cb._aggregate_hungarian(model_rows, mode),
                    _he.evaluate_pairs(gts, preds, mode),
                )

    def test_state_half_matches_state_diff_evaluate_pairs(self):
        """model 섹션의 state-diff 절반도 정본과 같아야 한다 (재계산이므로 정확히)."""
        for mode in MODES:
            with self.subTest(mode=mode):
                gts, preds = _fixture(mode)
                model_rows, _ = _cb.score_rows(gts, preds, mode)
                self.assertEqual(
                    _sd.aggregate(model_rows), _sd.evaluate_pairs(gts, preds, mode)
                )

    def test_only_total_collides_on_merge(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                gts, preds = _fixture(mode)
                model_rows, _ = _cb.score_rows(gts, preds, mode)
                h = _cb._aggregate_hungarian(model_rows, mode)
                s = _sd.aggregate(model_rows)
                self.assertEqual(set(h) & set(s), {"total"})
                self.assertEqual(h["total"], s["total"])

    def test_overall_is_the_concatenation_of_splits(self):
        """overall 은 섹션 평균의 가중합이 아니라 **행을 이어붙여** 잰 값이어야 한다."""
        for mode in MODES:
            with self.subTest(mode=mode):
                gts, preds = _fixture(mode)
                m = _cb.build_metrics(gts, preds, gts, preds, mode)
                self.assertEqual(
                    m["overall"]["model"],
                    _cb.aggregate(
                        _cb.score_rows(gts + gts, preds + preds, mode)[0], mode
                    ),
                )


class TestGainIntersection(unittest.TestCase):
    """gain 은 **행 단위 교집합** 위에서만 낸다 — 두 JSON 의 뺄셈이 아니다."""

    def _gain(self, mode):
        gts, preds = _fixture(mode)
        model_rows, copy_rows = _cb.score_rows(gts, preds, mode)
        return model_rows, copy_rows, _cb.compute_gain(model_rows, copy_rows, mode)

    def test_n_gain_equals_the_intersection_size(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model_rows, copy_rows, g = self._gain(mode)
                for k in _cb.gain_keys(mode):
                    expected = sum(
                        1
                        for m, c in zip(model_rows, copy_rows, strict=True)
                        if m[k] is not None and c[k] is not None
                    )
                    self.assertEqual(g[f"n_gain_{k}"], expected, k)

    def test_change_axis_intersection_is_strictly_smaller_than_the_model_side(self):
        """교집합이 실제로 좁아지는 행을 픽스처가 갖고 있어야 이 테스트가 의미를 갖는다.

        복사기는 변화를 주장하지 않으므로 `n_change_pred == 0` 이고, 따라서
        `change_*` 는 **GT 에 변화가 있는 행에서만** 정의된다. 픽스처 6행 중 GT 가
        current 와 다른 행은 0·1·3·4 의 4행이다. 모델 쪽은 거기에 5행(무변화 GT 인데
        변화를 지어냄)이 더해져 5행에서 정의된다 — 그 한 행이 교집합에서 빠져야 한다.
        """
        for mode in MODES:
            with self.subTest(mode=mode):
                model_rows, copy_rows, g = self._gain(mode)
                n_model = sum(
                    1 for r in model_rows if r["change_f1_strict"] is not None
                )
                self.assertEqual(n_model, 5)
                self.assertEqual(g["n_gain_change_f1_strict"], 4)

    def test_addmod_prec_gain_is_none_because_the_copier_never_defines_it(self):
        """복사기는 pred-side diff 가 항상 공집합이라 `addmod_prec`/`addmod_f1` 이 전 행 None.

        교집합이 0행이면 `avg_gain` 은 **None** 이다 — 0.0 은 "차이가 없다"로 읽힌다.
        """
        for mode in MODES:
            with self.subTest(mode=mode):
                _, copy_rows, g = self._gain(mode)
                self.assertTrue(all(r["addmod_prec"] is None for r in copy_rows))
                for k in ("addmod_prec", "addmod_f1"):
                    self.assertEqual(g[f"n_gain_{k}"], 0, k)
                    self.assertIsNone(g[f"avg_gain_{k}"], k)

    def test_gain_is_the_mean_of_row_differences(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model_rows, copy_rows, g = self._gain(mode)
                for k in ("hungarian_f1", "addmod_recall", "change_f1_strict"):
                    deltas = [
                        m[k] - c[k]
                        for m, c in zip(model_rows, copy_rows, strict=True)
                        if m[k] is not None and c[k] is not None
                    ]
                    self.assertAlmostEqual(
                        g[f"avg_gain_{k}"], sum(deltas) / len(deltas), places=4, msg=k
                    )

    def test_a_copying_model_gains_nothing(self):
        """모델이 복사기와 **같은 예측**을 내면 gain 이 전 키에서 정확히 0 이어야 한다."""
        for mode in MODES:
            with self.subTest(mode=mode):
                cur, gt = CUR[mode], GT[mode]
                gts = [{"messages": [{"value": "s"}, {"value": "u"}, {"value": gt}]}]
                preds = [{"prompt": _prompt(cur, mode), "predict": cur}]
                model_rows, copy_rows = _cb.score_rows(gts, preds, mode)
                g = _cb.compute_gain(model_rows, copy_rows, mode)
                for k in _cb.gain_keys(mode):
                    v = g[f"avg_gain_{k}"]
                    if v is not None:
                        self.assertEqual(v, 0.0, k)

    def test_no_gain_keys_are_absent(self):
        """방향성이 없는 키에는 gain 을 붙이지 않는다 (`copy_excess` 등)."""
        for mode in MODES:
            with self.subTest(mode=mode):
                _, _, g = self._gain(mode)
                for k in _cb._NO_GAIN_KEYS:
                    self.assertNotIn(f"avg_gain_{k}", g, k)
                    self.assertNotIn(f"n_gain_{k}", g, k)

    def test_position_key_follows_the_match_mode(self):
        self.assertIn("hungarian_pos", _cb.gain_keys("pos"))
        self.assertNotIn("hungarian_idx", _cb.gain_keys("pos"))
        self.assertIn("hungarian_idx", _cb.gain_keys("index"))
        self.assertNotIn("hungarian_pos", _cb.gain_keys("index"))


class TestSectionStructure(unittest.TestCase):
    def test_three_sections_with_three_slots_each(self):
        """`hungarian_metrics.json` 과 **같은 섹션 구조**여야 한다 —
        eval_viewer.load_metrics 의 section 조회는 부재를 silent skip 한다."""
        gts, preds = _fixture("index")
        m = _cb.build_metrics(gts, preds, gts, preds, "index")
        self.assertEqual(set(m), {"overall", "in_domain", "out_of_domain"})
        for sec in m.values():
            self.assertEqual(set(sec), {"copy_baseline", "model", "gain"})

    def test_prompt_parse_failure_raises_instead_of_scoring_an_empty_copier(self):
        """current 를 못 읽으면 복사기가 '빈 문자열을 내는 다른 모델'이 된다 — 터져야 한다."""
        gts = [{"messages": [{"value": "s"}, {"value": "u"}, {"value": "<node/>"}]}]
        preds = [
            {"prompt": "system\nrole\nuser\n(마커 없음)\nassistant\n", "predict": "x"}
        ]
        with self.assertRaises(_cb.CopyBaselineError):
            _cb.score_rows(gts, preds, "index")


class TestTruncatedLeaf(unittest.TestCase):
    """절단 leaf 는 **건너뛰지 않는다** — 잘린 것은 예측이지 프롬프트가 아니다.

    `_state_diff_eval._cmd_score` 는 leaf 전체를 건너뛰는데, 여기서는 복사기 점수를
    정상 산출하고 `model`/`gain` 만 null 로 둔다. 그 차이가 이 채점기의 설계점이라
    회귀하면 절단 leaf 17개가 통째로 비어 버린다.
    """

    REASON = "generated_predictions_id.jsonl 이 절단(1024)됐다 — 테스트용 사유"

    def setUp(self):
        self._real = _sd.truncated_reason

    def tearDown(self):
        _sd.truncated_reason = self._real

    def _write_split(self, d: Path, mode: str):
        gts, preds = _fixture(mode)
        paths = {}
        for split in ("id", "ood"):
            t = d / f"test_{split}.jsonl"
            p = d / f"generated_predictions_{split}.jsonl"
            t.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in gts),
                encoding="utf-8",
            )
            p.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in preds),
                encoding="utf-8",
            )
            paths[split] = (str(t), str(p))
        return paths

    def _run(self, d: Path, mode: str) -> dict:
        paths = self._write_split(d, mode)
        out = d / "copy_baseline_metrics.json"
        args = argparse.Namespace(
            test=None,
            pred=None,
            test_id=paths["id"][0],
            pred_id=paths["id"][1],
            test_ood=paths["ood"][0],
            pred_ood=paths["ood"][1],
            output=str(out),
            match_mode=mode,
            strict_pos_match=False,
            include_aria=False,
            exclude_action=None,
        )
        self.assertEqual(_cb._cmd_score(args), 0)
        return json.loads(out.read_text(encoding="utf-8"))

    def test_baseline_survives_truncation_while_model_and_gain_go_null(self):
        _sd.truncated_reason = lambda *paths: self.REASON
        with tempfile.TemporaryDirectory() as tmp:
            m = self._run(Path(tmp), "index")
        self.assertEqual(m["truncated"], self.REASON)
        for sec in ("overall", "in_domain", "out_of_domain"):
            self.assertIsNone(m[sec]["model"], sec)
            self.assertIsNone(m[sec]["gain"], sec)
            cb = m[sec]["copy_baseline"]
            self.assertEqual(cb["copy_exact_rate"], 1.0, sec)
            self.assertGreater(cb["avg_hungarian_f1"], 0.0, sec)

    def test_intact_leaf_carries_model_and_gain(self):
        _sd.truncated_reason = lambda *paths: None
        with tempfile.TemporaryDirectory() as tmp:
            m = self._run(Path(tmp), "index")
        self.assertIsNone(m["truncated"])
        self.assertIsNotNone(m["overall"]["model"])
        self.assertIn("avg_gain_hungarian_f1", m["overall"]["gain"])

    def test_top_level_metadata_is_stamped(self):
        _sd.truncated_reason = lambda *paths: None
        with tempfile.TemporaryDirectory() as tmp:
            m = self._run(Path(tmp), "pos")
        self.assertEqual(m["copy_baseline_schema"], _cb.COPY_BASELINE_SCHEMA)
        self.assertEqual(m["metrics_schema"], _sd.METRICS_SCHEMA)
        self.assertEqual(m["match_mode"], "pos")
        self.assertEqual(
            set(m),
            {
                "overall",
                "in_domain",
                "out_of_domain",
                "copy_baseline_schema",
                "metrics_schema",
                "match_mode",
                "element_set",
                "truncated",
            },
        )


class TestViewerExposure(unittest.TestCase):
    """산출물이 **뷰어까지 닿는가.** 이 채점기는 2026-08-11 부터 leaf 옆에 gain 을
    쓰고 있었는데 `eval_viewer` 가 그 파일을 아예 안 읽어 34개 leaf 에서 열이 통째로
    비어 있었다. 그 구멍이 조용했던 이유가 둘이라 둘 다 여기서 찌른다:

      1. site 모드는 `rows[s][k]` bare lookup 이라 **없는 키가 빈 칸**이 된다
         (pairs 모드의 `_metric()` 폴백조차 안 탄다). 키를 잘못 쓰면 아무도 모른다.
      2. 이 파일만 섹션이 **2단**(`{section: {copy_baseline, model, gain}}`)이라
         1단 section 조회로는 numeric scalar 가 하나도 안 잡혀 조용히 빈 dict 다.
    """

    def _write_leaf(self, d: Path, metrics: dict) -> Path:
        (d / "copy_baseline_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
        )
        return d

    def test_registered_gain_keys_exist_in_real_output(self):
        """`STATE_METRIC_KEYS` 의 gain 키가 실제 산출물에서 값을 얻는가."""
        ev = __import__("eval_viewer")
        gts, preds = _fixture("index")
        m = _cb.build_metrics(gts, preds, gts, preds, "index")
        gain_keys = [k for k in ev.STATE_METRIC_KEYS if "_gain_" in k]
        self.assertTrue(gain_keys, "gain 키가 등록되지 않았다 — 노출 자체가 회귀")
        with tempfile.TemporaryDirectory() as tmp:
            leaf = self._write_leaf(Path(tmp), m)
            loaded = ev.load_metrics(leaf, [ev._copy_baseline_file("in_domain")])
        for k in gain_keys:
            self.assertIn(k, loaded, f"{k} 가 산출물에 없다 — 열이 빈 칸이 된다")

    def test_registered_gain_keys_match_the_scorer(self):
        """키 이름이 채점기 정의와 붙어 있는가. `gain_keys()` 가 개명되면 뷰어 열은
        **에러 없이** 비므로, 목록끼리 직접 대조해 두지 않으면 아무도 못 본다.
        위치 키(`hungarian_idx`/`hungarian_pos`)는 모드마다 이름이 달라 합집합으로 본다."""
        ev = __import__("eval_viewer")
        known = {
            f"{prefix}_gain_{k}"
            for prefix in ("avg", "n")
            for mode in ("index", "pos")
            for k in _cb.gain_keys(mode)
        }
        for k in ev.STATE_METRIC_KEYS:
            if "_gain_" in k:
                self.assertIn(k, known, f"{k} 는 채점기가 내지 않는 이름이다")

    def test_every_state_entry_registers_the_file(self):
        """state leaf 마다 등록돼야 한다 — 한 곳만 빠지면 그 분할만 조용히 빈다."""
        ev = __import__("eval_viewer")
        for stage, exps in ev.EVAL_DATASETS.items():
            for exp, entries in exps.items():
                for key, entry in entries.items():
                    if entry["metric_keys"] is not ev.STATE_METRIC_KEYS:
                        continue
                    files = {fn for fn, _ in entry["metric_files"]}
                    self.assertIn(
                        "copy_baseline_metrics.json",
                        files,
                        f"stage{stage}/{exp}/{key} 에 복사기 기준선이 없다",
                    )

    def test_truncated_leaf_yields_no_gain_columns_without_raising(self):
        """절단 leaf 는 `gain` 이 null 이다. 예외 없이 **열이 비어야** 한다 —
        0 으로 채워지면 "복사기와 같다"는 거짓 주장이 표에 남는다."""
        ev = __import__("eval_viewer")
        m = {"in_domain": {"copy_baseline": {"total": 3}, "model": None, "gain": None}}
        with tempfile.TemporaryDirectory() as tmp:
            leaf = self._write_leaf(Path(tmp), m)
            loaded = ev.load_metrics(leaf, [ev._copy_baseline_file("in_domain")])
        self.assertEqual(loaded, {})

    def test_dotted_section_does_not_leak_the_wrapper_level(self):
        """점 경로가 한 단만 내려가면 `copy_baseline`(예측과 무관한 기준선 절대값)이
        gain 열 자리에 섞여 들어온다 — 세팅 비교 표에서 전부 같은 값이 된다."""
        ev = __import__("eval_viewer")
        m = {
            "overall": {
                "copy_baseline": {"avg_hungarian_f1": 0.9},
                "model": {"avg_hungarian_f1": 0.4},
                "gain": {"avg_gain_hungarian_f1": -0.5},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            leaf = self._write_leaf(Path(tmp), m)
            loaded = ev.load_metrics(leaf, [ev._copy_baseline_file("overall")])
        self.assertEqual(loaded, {"avg_gain_hungarian_f1": -0.5})

    def test_top_level_string_stamps_do_not_break_the_section_scan(self):
        """산출물 최상위에는 섹션 dict 말고 **문자열 스탬프**도 있다
        (`element_set` · `metrics_schema` · `match_mode` · `truncated`).
        2026-08-21 에 `element_set` 이 추가되면서 최상위 키가 또 하나 늘었다 —
        아직 재채점된 leaf 가 없어 on-disk 로는 못 찌르므로 여기서 합성으로 고정한다.

        세 경로를 다 본다: 이름 조회(AC), 최상위 순회(MB/MC single-pair), 점 경로.
        점 경로가 문자열 위에서 `.get()` 을 부르면 `AttributeError` 로 **빌드가 죽는다**
        — silent skip 이 설계인 함수에서 그건 회귀다.
        """
        ev = __import__("eval_viewer")
        m = {
            "in_domain": {"avg_hungarian_f1": 0.75},
            "element_set": "full",
            "metrics_schema": "2026-08-04",
            "truncated": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            leaf = Path(tmp)
            (leaf / "hungarian_metrics.json").write_text(
                json.dumps(m, ensure_ascii=False), encoding="utf-8"
            )
            named = ev.load_metrics(leaf, [("hungarian_metrics.json", "in_domain")])
            flat = ev.load_metrics(leaf, [("hungarian_metrics.json", None)])
            dotted = ev.load_metrics(leaf, [("hungarian_metrics.json", "element_set.gain")])
        # 이름 조회: 스탬프는 섹션 밖이라 섞이지 않는다.
        self.assertEqual(named, {"avg_hungarian_f1": 0.75})
        # 최상위 순회: 스탬프가 함께 실리지만 STATE_METRIC_KEYS 에 없어 렌더링되지
        # 않는다. 중요한 건 **예외 없이** 지나가고 숫자 열을 오염시키지 않는 것이다.
        self.assertEqual(flat.get("element_set"), "full")
        self.assertNotIn("avg_hungarian_f1", flat)
        # 점 경로가 문자열을 만나면 조용히 빈 dict.
        self.assertEqual(dotted, {})


if __name__ == "__main__":
    unittest.main()
