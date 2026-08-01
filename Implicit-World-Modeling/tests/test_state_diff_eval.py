"""scripts/_state_diff_eval.py — copy-bias 진단 채점기 테스트.

이 채점기의 실패 모드는 전부 **조용하다**. 배선이 끊기면 전 행 0점, 프롬프트 파싱이
어긋나면 "전부 ADDED · copy_rate 0" 이라는 그럴듯한 오답표가 나온다. 값이 그럴듯해서
사람이 못 잡으므로, 여기서는 정답을 아는 합성 입력으로 각 실패 모드를 직접 찌른다.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import timedelta
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_hungarian_eval = __import__("_hungarian_eval")
_sd = __import__("_state_diff_eval")
_ps = __import__("_prompt_sections")


# ── 합성 상태 ────────────────────────────────────────────────────────────
# current: 버튼 OK + 문단 "unread inbox 3"
# next   : 버튼 OK 그대로(UNCHANGED)
#        + 문단 "unread inbox 7"(MODIFIED — 토큰 일부만 바뀜)
#        + 문단 "brand new banner"(ADDED)
#
# MODIFIED 쪽 텍스트는 **일부러 부분 겹침**으로 잡았다. 텍스트가 통째로 바뀌면
# `_text_sim` 이 0 → 매칭 cost 가 정확히 `W_TEXT`(1.5)가 되는데, index 모드의
# `MATCH_THRESHOLD` 도 1.5 이고 판정이 `cost < threshold` 라 매칭이 **떨어진다**
# (→ ADDED). pos 모드는 threshold 1.7 이라 붙는다(→ MODIFIED). 즉 "완전 교체"는
# 모드마다 유형이 갈리는 경계값이라, 세 유형 분류 자체를 검증하는 픽스처로는
# 부적절하다. 그 경계 동작은 아래 `test_full_text_swap_is_mode_dependent` 가 따로 고정한다.
CUR_IDX = (
    '<node index="0">'
    '<button index="1" aria-label="OK"/>'
    '<p index="2">unread inbox 3</p>'
    "</node>"
)
GT_IDX = (
    '<node index="0">'
    '<button index="1" aria-label="OK"/>'
    '<p index="2">unread inbox 7</p>'
    '<p index="3">brand new banner</p>'
    "</node>"
)
CUR_POS = (
    '<node bounds="[0,0][10,10]" point="[5,5]">'
    '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
    '<p bounds="[6,6][9,9]" point="[7,7]">unread inbox 3</p>'
    "</node>"
)
GT_POS = (
    '<node bounds="[0,0][10,10]" point="[5,5]">'
    '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
    '<p bounds="[6,6][9,9]" point="[7,7]">unread inbox 7</p>'
    '<p bounds="[6,20][9,24]" point="[7,22]">brand new banner</p>'
    "</node>"
)
MODES = (("index", CUR_IDX, GT_IDX), ("pos", CUR_POS, GT_POS))


class TestWiring(unittest.TestCase):
    """지연 로드 미초기화 → 전 행 0점 함정. 데이터와 무관한 고정 XML 로 배선만 본다."""

    def test_self_test_passes_both_modes(self):
        for mode, _, _ in MODES:
            with self.subTest(mode=mode):
                _sd.assert_scorer_wired(mode)  # 예외 없이 통과해야 한다

    def test_self_test_catches_broken_extractor(self):
        """extract_elements 가 죽으면 self-test 가 반드시 잡아야 한다."""
        orig = _hungarian_eval.extract_elements
        _hungarian_eval.extract_elements = lambda *a, **k: []
        try:
            with self.assertRaises(_sd.StateDiffError):
                _sd.assert_scorer_wired("index")
        finally:
            _hungarian_eval.extract_elements = orig


class TestClassifyDiff(unittest.TestCase):
    def test_three_way_split(self):
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                counts = _sd.summarize_diff(_sd.classify_diff(cur, gt, mode))
                self.assertEqual(counts["UNCHANGED"], 1, "OK 버튼은 그대로")
                self.assertEqual(counts["MODIFIED"], 1, "unread inbox 3 → 7")
                self.assertEqual(counts["ADDED"], 1, "brand new banner 는 신규")

    def test_identical_states_are_all_unchanged(self):
        for mode, cur, _ in MODES:
            with self.subTest(mode=mode):
                counts = _sd.summarize_diff(_sd.classify_diff(cur, cur, mode))
                self.assertEqual(counts["MODIFIED"], 0)
                self.assertEqual(counts["ADDED"], 0)
                self.assertGreater(counts["UNCHANGED"], 0)

    def test_full_text_swap_is_mode_dependent(self):
        """텍스트 완전 교체는 index 에서 ADDED, pos 에서 MODIFIED — 임계값 차이다.

        `_text_sim` 이 0 이면 cost 가 정확히 `W_TEXT`(1.5)인데 판정이 `cost < threshold`
        이고 index 의 threshold 가 1.5 라 딱 떨어진다. 두 모드의 diff 유형 분포를
        나란히 놓고 비교하면 안 되는 이유이므로, 값이 아니라 이 성질을 고정해 둔다.
        """
        cur_i = '<node index="0"><p index="1">alpha</p></node>'
        gt_i = '<node index="0"><p index="1">omega</p></node>'
        cur_p = (
            '<node bounds="[0,0][10,10]" point="[5,5]">'
            '<p bounds="[1,1][5,5]" point="[3,3]">alpha</p></node>'
        )
        gt_p = (
            '<node bounds="[0,0][10,10]" point="[5,5]">'
            '<p bounds="[1,1][5,5]" point="[3,3]">omega</p></node>'
        )
        self.assertEqual(
            _sd.summarize_diff(_sd.classify_diff(cur_i, gt_i, "index"))["ADDED"], 1
        )
        self.assertEqual(
            _sd.summarize_diff(_sd.classify_diff(cur_p, gt_p, "pos"))["MODIFIED"], 1
        )

    def test_empty_current_makes_everything_added(self):
        for mode, _, gt in MODES:
            with self.subTest(mode=mode):
                counts = _sd.summarize_diff(_sd.classify_diff("", gt, mode))
                self.assertEqual(counts["UNCHANGED"], 0)
                self.assertEqual(counts["MODIFIED"], 0)
                self.assertGreater(counts["ADDED"], 0)


class TestCopyVersusPrediction(unittest.TestCase):
    """이 채점기의 존재 이유 — 복사와 예측이 실제로 갈리는가."""

    def test_pure_copy_scores_zero_on_diff(self):
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                r = _sd.compute_state_diff(cur, gt, cur, mode)
                self.assertEqual(r["copy_rate_pred"], 1.0, "current 를 그대로 냈다")
                self.assertEqual(r["copy_exact"], 1.0)
                self.assertEqual(r["copy_near"], 1.0)
                self.assertGreater(
                    r["copy_excess"], 0.0, "GT 가 겹치는 정도를 초과해 베꼈다"
                )
                self.assertEqual(r["added_recall"], 0.0, "신규 요소는 하나도 못 냈다")

    def test_perfect_prediction_has_no_excess(self):
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                r = _sd.compute_state_diff(gt, gt, cur, mode)
                self.assertEqual(r["diff_recall"], 1.0)
                self.assertEqual(r["added_recall"], 1.0)
                self.assertEqual(r["unchanged_recall"], 1.0)
                self.assertAlmostEqual(
                    r["copy_excess"], 0.0, places=4, msg="GT 와 같으면 초과 복사가 0"
                )

    def test_copy_rate_alone_cannot_separate_them(self):
        """copy_rate 단독으로는 복사와 완벽예측이 안 갈린다 — copy_excess 가 필요한 이유."""
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                copied = _sd.compute_state_diff(cur, gt, cur, mode)
                perfect = _sd.compute_state_diff(gt, gt, cur, mode)
                self.assertGreater(
                    perfect["copy_rate_pred"], 0.5, "완벽예측도 많이 겹친다"
                )
                self.assertGreater(
                    copied["copy_excess"] - perfect["copy_excess"],
                    0.1,
                    "copy_excess 는 둘을 확실히 가른다",
                )


class TestStratumInvariant(unittest.TestCase):
    """recall 계열은 정본 hungarian_rec 의 층 분해여야 한다.

    diff 부분집합을 따로 매칭하면 UNCHANGED 에 붙었어야 할 예측 요소가 MODIFIED 로
    재배정되며 recall 이 부풀어 이 항등식이 깨진다. 그 구현 실수를 여기서 잡는다.
    """

    def _check(self, pred, gt, cur, mode):
        r = _sd.compute_state_diff(pred, gt, cur, mode)
        if not r["n_gt"]:
            return
        hits = sum(
            (r[f"{t}_recall"] or 0.0) * r[f"n_gt_{t}"]
            for t in ("added", "modified", "unchanged")
        )
        canonical = _hungarian_eval.compute_hungarian_acc(pred, gt, mode)
        self.assertAlmostEqual(
            hits / r["n_gt"],
            canonical["hungarian_rec"],
            places=3,
            msg=f"층 분해가 hungarian_rec 과 어긋남 ({mode})",
        )

    def test_invariant_holds(self):
        for mode, cur, gt in MODES:
            for name, pred in (("copy", cur), ("perfect", gt), ("empty", "")):
                with self.subTest(mode=mode, pred=name):
                    self._check(pred, gt, cur, mode)


class TestPromptFamilies(unittest.TestCase):
    """두 계열 모두에서 current state 가 나와야 한다 — 한 계열만 보면 woa 사고 재발."""

    FAMILY_A = (
        "system\nrole\nuser\n\n## Current State\n" + CUR_IDX + "\n\n## Action\n"
        '{"action_type":"click","index":"1"}\nassistant\n'
    )
    FAMILY_B = (
        "system\nrole\nuser\nCurrent UI State:\n" + CUR_POS + "\n\n[Screenshot]\n"
        '\nAction:\n<action>{"action": "click", "coordinate": [3, 3]}</action>\n'
        "assistant\n"
    )

    def test_both_families_yield_current_state(self):
        for name, prompt in (("A", self.FAMILY_A), ("B", self.FAMILY_B)):
            with self.subTest(family=name):
                cur = _ps.parse_prompt(prompt).get("current_state", "")
                self.assertTrue(cur, f"계열 {name} 에서 current_state 를 못 읽었다")
                self.assertIn("<button", cur)
                self.assertNotIn("Screenshot", cur)

    def test_unparseable_prompt_raises_not_silently_zero(self):
        """파싱 실패는 조용한 0 이 아니라 예외여야 한다."""
        gts = [{"messages": [{"value": "sys"}, {"value": "u"}, {"value": GT_IDX}]}]
        preds = [
            {
                "prompt": "system\nrole\nuser\n(마커 없음)\nassistant\n",
                "predict": GT_IDX,
            }
        ]
        with self.assertRaises(_sd.StateDiffError):
            _sd.evaluate_pairs(gts, preds, "index")


class TestAggregate(unittest.TestCase):
    def test_undefined_rows_are_excluded_not_zeroed(self):
        """GT 에 ADDED 가 없는 행의 added_recall 은 None — 0 으로 세면 평균이 왜곡된다."""
        rows = [
            {**{k: None for k in _sd._MEAN_KEYS}, "added_recall": 1.0},
            {**{k: None for k in _sd._MEAN_KEYS}},  # added_recall 정의 안 됨
        ]
        agg = _sd.aggregate(rows)
        self.assertEqual(agg["avg_added_recall"], 1.0, "정의된 행만 평균낸다")
        self.assertEqual(agg["n_added_recall"], 1, "몇 행 위에서 쟀는지 함께 기록")
        self.assertEqual(agg["total"], 2)

    def test_section_structure_matches_hungarian_metrics(self):
        """eval_viewer.load_metrics 의 section 조회는 부재를 silent skip 한다 —
        구조가 어긋나면 표에 빈칸만 뜨고 아무도 오류를 못 본다."""
        gts = [{"messages": [{"value": "s"}, {"value": "u"}, {"value": GT_IDX}]}]
        preds = [{"prompt": TestPromptFamilies.FAMILY_A, "predict": GT_IDX}]
        m = _sd.build_metrics(gts, preds, gts, preds, "index")
        self.assertEqual(set(m), {"overall", "in_domain", "out_of_domain"})


class TestTruncationGuard(unittest.TestCase):
    """절단 leaf 는 copy_rate 를 **한쪽으로** 과소평가하므로 채점 자체를 막아야 한다.

    가드가 백필 스크립트에만 있으면 `rebuild_woa_metrics.sh → _hungarian_eval score`
    경로가 그대로 통과한다. 그래서 채점기 안에 두고, 여기서 그 사실을 고정한다.
    """

    def _touch(self, tmpdir, name, when):
        p = Path(tmpdir) / name
        p.write_text("{}\n")
        ts = when.timestamp()
        os.utime(p, (ts, ts))
        return str(p)

    def test_pre_fix_prediction_is_flagged(self):
        import tempfile

        before = _sd.MAX_NEW_TOKENS_FIX_UTC - timedelta(hours=1)
        after = _sd.MAX_NEW_TOKENS_FIX_UTC + timedelta(hours=1)
        with tempfile.TemporaryDirectory() as d:
            old = self._touch(d, "generated_predictions_id.jsonl", before)
            new = self._touch(d, "generated_predictions_ood.jsonl", after)
            self.assertIsNotNone(_sd.truncated_reason(old))
            self.assertIsNone(_sd.truncated_reason(new))
            # 한쪽만 절단이어도 leaf 전체를 막는다 (섹션 합산이 오염되므로)
            self.assertIsNotNone(_sd.truncated_reason(new, old))

    def test_missing_and_empty_paths_are_ignored(self):
        self.assertIsNone(_sd.truncated_reason(None, "", "/nonexistent/x.jsonl"))

    def test_constant_is_shared_with_compare_site(self):
        """경계 상수가 두 벌이면 언젠가 조용히 갈린다."""
        cs = __import__("_compare_site")
        self.assertIs(cs.MAX_NEW_TOKENS_FIX_UTC, _sd.MAX_NEW_TOKENS_FIX_UTC)


class TestSanitySignals(unittest.TestCase):
    def test_unclosed_root_flags_truncation(self):
        self.assertEqual(_sd._unclosed_root(GT_IDX), 0.0)
        self.assertEqual(_sd._unclosed_root('<node index="0"><p index="1">cut'), 1.0)
        self.assertEqual(_sd._unclosed_root('<node index="0"/>'), 0.0)
        self.assertEqual(_sd._unclosed_root("no tags at all"), 1.0)


if __name__ == "__main__":
    unittest.main()
