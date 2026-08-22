"""scripts/_state_diff_eval.py — copy-bias 진단 채점기 테스트.

이 채점기의 실패 모드는 전부 **조용하다**. 배선이 끊기면 전 행 0점, 프롬프트 파싱이
어긋나면 "전부 ADDED · copy_rate 0" 이라는 그럴듯한 오답표가 나온다. 값이 그럴듯해서
사람이 못 잡으므로, 여기서는 정답을 아는 합성 입력으로 각 실패 모드를 직접 찌른다.
"""

from __future__ import annotations

import json
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
#
# ⚠️ element 집합이 `full` 이 된 뒤(2026-08-21) **root `<node>` 도 요소다.** 자체 텍스트가
# 없어 서브트리를 흡수하므로 자손의 변화가 조상에도 그대로 나타난다 — 아래 기대값이
# "요소 하나"가 아니라 "요소 + 그 조상"으로 세어지는 이유다. 옛 집합의 기대값은
# `tests/test_element_set.py` 가 `legacy` 로 따로 고정한다.
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

# 변화 둘(MODIFIED + ADDED) 중 하나만 낸 예측 — change_f1_strict 이 복사기와 완벽예측 사이에
# 놓이는지 보는 중간항이다. 0/1 두 점만 고정하면 상수를 돌려주는 구현도 통과한다.
PARTIAL_IDX = (
    '<node index="0">'
    '<button index="1" aria-label="OK"/>'
    '<p index="2">unread inbox 7</p>'
    "</node>"
)
PARTIAL_POS = (
    '<node bounds="[0,0][10,10]" point="[5,5]">'
    '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
    '<p bounds="[6,6][9,9]" point="[7,7]">unread inbox 7</p>'
    "</node>"
)
PARTIALS = {"index": PARTIAL_IDX, "pos": PARTIAL_POS}


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
                self.assertEqual(
                    counts["MODIFIED"], 2, "p(3→7) 와 그 텍스트를 흡수한 root node"
                )
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
        # 2 인 것은 p 와 그 텍스트를 흡수한 root node 가 같은 판정을 받기 때문이다
        # (full 집합). 고정하려는 것은 값이 아니라 "index=ADDED / pos=MODIFIED" 라는 성질이다.
        self.assertEqual(
            _sd.summarize_diff(_sd.classify_diff(cur_i, gt_i, "index"))["ADDED"], 2
        )
        self.assertEqual(
            _sd.summarize_diff(_sd.classify_diff(cur_p, gt_p, "pos"))["MODIFIED"], 2
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
                self.assertEqual(r["addmod_recall"], 1.0)
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


class TestChangeF1(unittest.TestCase):
    """change_f1_strict — "변화 자체를 예측했는가".

    `addmod_recall` 이 못 보는 두 가지를 본다: (a) **없어져야 할 요소를 지웠는가**
    (GT 를 분모로 잡는 recall 은 이 축을 볼 수 없다), (b) 자리만 맞고 내용이 틀린
    예측을 맞힌 것으로 세지 않는가 (매칭 임계 1.5/1.7 은 `_text_sim` 이 0 이어도
    붙을 만큼 느슨하다 — 그래서 τ 로 한 번 더 거른다).
    """

    def test_copier_scores_zero(self):
        """current 를 그대로 낸 예측은 변화를 하나도 주장하지 않았다 → 0.0.

        **None 이면 안 된다.** 정의불능으로 빼면 복사기가 평균에서 사라진다 —
        이 지표가 존재하는 이유 자체가 그것을 세는 것이다.
        """
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                r = _sd.compute_state_diff(cur, gt, cur, mode)
                self.assertEqual(r["change_f1_strict"], 0.0)
                self.assertEqual(r["n_change_pred"], 0, "변화를 하나도 안 냈다")
                self.assertGreater(r["n_change_gt"], 0, "실제로는 변화가 있었다")

    def test_perfect_prediction_scores_one_and_partial_lands_between(self):
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                perfect = _sd.compute_state_diff(gt, gt, cur, mode)
                partial = _sd.compute_state_diff(PARTIALS[mode], gt, cur, mode)
                copier = _sd.compute_state_diff(cur, gt, cur, mode)
                self.assertEqual(perfect["change_f1_strict"], 1.0)
                self.assertLess(copier["change_f1_strict"], partial["change_f1_strict"])
                self.assertLess(partial["change_f1_strict"], perfect["change_f1_strict"])

    def test_keeping_stale_text_scores_zero_while_hungarian_stays_perfect(self):
        """토글 하나가 뒤집히는 행 — 정본 지표로는 안 잡히는 실패를 잡는가.

        "wifi off" → "wifi on" 인데 예측이 옛 텍스트를 유지하면 요소 집합은 그대로라
        `hungarian_f1` 이 **1.0** 이다. 이 행에서 틀린 것은 딱 하나, 변화 그 자체다.
        """
        cur = '<node index="0"><p index="1">wifi off</p></node>'
        gt = '<node index="0"><p index="1">wifi on</p></node>'
        r = _sd.compute_state_diff(cur, gt, cur, "index")
        canonical = _hungarian_eval.compute_hungarian_acc(cur, gt, "index")
        self.assertEqual(canonical["hungarian_f1"], 1.0, "정본 지표는 만점을 준다")
        self.assertEqual(r["change_f1_strict"], 0.0, "change_f1 은 0 이어야 한다")

    def test_ignored_deletion_costs_recall(self):
        """사라져야 할 요소를 남겨두면 감점된다 — DELETED 축이 실제로 도는지 본다.

        예측은 신규 배너(ADDED)를 정확히 냈지만 사라진 행을 그대로 들고 있다.
        DELETED 축이 없으면 두 예측이 똑같이 1.0 이 되어 구분되지 않는다.
        """
        cur = '<node index="0"><button index="1"/><p index="2">old row</p></node>'
        gt = (
            '<node index="0"><button index="1"/>'
            '<span index="3">fresh banner</span></node>'
        )
        stale = (
            '<node index="0"><button index="1"/><p index="2">old row</p>'
            '<span index="3">fresh banner</span></node>'
        )
        # DELETED 가 2 인 것은 old row 하나 때문이 아니다: root node 의 흡수 텍스트가
        # "old row" → "fresh banner" 로 통째로 갈리는데 index 임계(1.5)에서는 그 쌍이
        # 떨어져 root 도 DELETED+ADDED 로 잡힌다 (full 집합).
        self.assertEqual(
            _sd.summarize_diff(_sd.classify_diff(cur, gt, "index"))["DELETED"],
            2,
            "GT 는 old row 를 지웠다 (+ 흡수 텍스트가 갈린 root)",
        )
        ignored = _sd.compute_state_diff(stale, gt, cur, "index")
        honored = _sd.compute_state_diff(gt, gt, cur, "index")
        self.assertEqual(honored["change_f1_strict"], 1.0)
        self.assertLess(ignored["change_f1_strict"], 1.0)
        self.assertEqual(
            ignored["n_change_gt"], 4, "ADDED span/root + DELETED p/root"
        )
        self.assertEqual(
            ignored["n_change_pred"], 2, "ADDED span + MODIFIED root 만 주장했다"
        )

    def test_tau_is_the_boundary(self):
        """τ 경계 — 매칭은 됐는데 내용이 τ 미만이면 맞힌 것으로 세지 않는다.

        τ 는 `_text_sim` 에 걸리고 그 함수는 mode 와 무관하므로 index 한 모드로 고정한다.
        토큰 집합 Jaccard 라 `t1..t10` 중 앞 k 개를 내면 sim 이 정확히 k/10 이 된다.

        root 에 `id="root"` 를 준 것은 의도다. full 집합에서 텍스트 없는 root 는 자손
        텍스트를 흡수해 p 와 함께 움직이는데, index 임계에서는 그 쌍이 떨어져
        **DELETED 공짜 hit** 이 생긴다 — τ 미만인데도 f1 이 0.33 이 되어 이 축이 재려던
        것("τ 미만은 못 맞힌 것")을 가린다. 자체 텍스트를 주면 root 가 UNCHANGED 로
        빠져 τ 게이트만 남는다. (root 의 흡수 동작 자체는 위 세 테스트가 고정한다.)
        """
        gt_tokens = " ".join(f"t{i}" for i in range(1, 11))
        cur = '<node index="0" id="root"><button index="1"/></node>'
        gt = (
            f'<node index="0" id="root"><button index="1"/>'
            f'<p index="2">{gt_tokens}</p></node>'
        )

        def _f1(k):
            toks = " ".join(f"t{i}" for i in range(1, k + 1))
            pred = (
                f'<node index="0" id="root"><button index="1"/>'
                f'<p index="2">{toks}</p></node>'
            )
            self.assertAlmostEqual(
                _hungarian_eval._text_sim(toks, gt_tokens), k / 10, places=6
            )
            r = _sd.compute_state_diff(pred, gt, cur, "index")
            self.assertEqual(r["n_change_gt"], 1)
            self.assertEqual(r["n_change_pred"], 1, "양쪽 다 '문단 하나 추가'를 주장")
            return r["change_f1_strict"]

        self.assertEqual(_sd.CHANGE_TEXT_SIM_TAU, 0.9)
        self.assertEqual(_f1(9), 1.0, "sim == τ 는 맞힌 것 (>= 비교)")
        self.assertEqual(_f1(8), 0.0, "sim < τ 는 못 맞힌 것")

    def test_hallucinated_change_is_penalized_not_dropped(self):
        """GT 에 변화가 없는데 예측이 변화를 지어내면 0.0 이다.

        분모(`|C_gt|`)가 없다고 None 으로 빼면 환각이 평균에서 사라진다.
        양쪽 다 변화가 없는 행에서만 None 이다 — 그때는 정말 잴 것이 없다.
        """
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                halluc = _sd.compute_state_diff(gt, cur, cur, mode)  # GT = current
                self.assertEqual(halluc["n_change_gt"], 0)
                self.assertGreater(halluc["n_change_pred"], 0)
                self.assertEqual(halluc["change_f1_strict"], 0.0)

                still = _sd.compute_state_diff(cur, cur, cur, mode)
                self.assertEqual(still["n_change_pred"], 0)
                self.assertIsNone(still["change_f1_strict"], "양쪽 다 변화 없음 = 정의불능")


class TestChangeF1Floor(unittest.TestCase):
    """change_f1_strict 의 바닥은 0 이 아니다 — 그 눈금(`change_f1_floor`)을 고정한다.

    복사기는 0.0 이지만 **반대쪽 퇴화**인 빈 예측은 그렇지 않다: 아무것도 안 내면
    current 전체를 지운 것으로 분류되고, 화면 전환은 실제로 current 의 상당 부분을
    지우므로 그 교집합이 공짜 hit 이 된다. 2026-08-04 실측(각 200행)에서 빈 예측이
    EXP01 0.383 · EXP05 0.235 · EXP07v1 0.258 을 받았고, 같은 leaf 의 **학습된**
    EXP07v1 lora ep1 은 0.114 라 바닥에 진다. 눈금 없이 0 기준으로 읽으면 그게
    "base > trained" 라는 결과처럼 보인다 — 그래서 두 값은 항상 같이 나와야 한다.
    """

    # 사라지는 요소가 있는 픽스처. 상단 MODES 는 DELETED 가 0 이라 이 축이 안 켜진다.
    DEL_IDX = (
        '<node index="0"><button index="1"/><p index="2">old row</p></node>',
        '<node index="0"><button index="1"/>'
        '<span index="3">fresh banner</span></node>',
    )
    DEL_POS = (
        '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<p bounds="[6,6][9,9]" point="[7,7]">old row</p></node>',
        '<node bounds="[0,0][10,10]" point="[5,5]">'
        '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
        '<span bounds="[6,20][9,24]" point="[7,22]">fresh banner</span></node>',
    )
    DELS = {"index": DEL_IDX, "pos": DEL_POS}

    # 비어 있지 않으면서 current 를 하나도 재현하지 않는 예측 — 퇴화의 최소 형태.
    MAXDEL = {
        "index": '<node index="900"><select index="901"/></node>',
        "pos": '<node bounds="[900,900][910,910]" point="[905,905]">'
        '<select bounds="[920,920][930,930]" point="[925,925]"/></node>',
    }

    def test_empty_prediction_scores_zero(self):
        """생성 실패는 0 이다 (2026-08-04 규칙 변경).

        이전에는 빈 예측이 "current 를 전부 지웠다는 주장"으로 분류돼 바닥값
        (0.24~0.38)을 공짜로 받았다. ScratchWorld 선례("파싱 실패 = 오답")를 따라
        주장 자체를 비운다. **None 이 아니라 0.0** 이어야 한다 — 평균에서 빠지면
        아무것도 못 낸 모델의 실패가 감춰진다.
        """
        for mode, (cur, gt) in self.DELS.items():
            with self.subTest(mode=mode):
                # 값이 아니라 전제("이 축이 켜져 있다")를 본다. full 집합에서는 모드마다
                # 개수가 갈린다 — root 의 흡수 텍스트가 통째로 바뀌는데 그 쌍의 cost 가
                # 정확히 1.5 라, index(임계 1.5)에서는 떨어져 DELETED 가 하나 더 생기고
                # pos(1.7)에서는 붙어 MODIFIED 가 된다.
                self.assertGreaterEqual(
                    _sd.summarize_diff(_sd.classify_diff(cur, gt, mode))["DELETED"],
                    1,
                    "픽스처에 사라지는 요소가 있어야 이 축이 켜진다",
                )
                empty = _sd.compute_state_diff("", gt, cur, mode)
                self.assertEqual(empty["change_f1_strict"], 0.0)
                self.assertEqual(empty["change_f1_loose"], 0.0)
                self.assertEqual(empty["n_change_pred"], 0, "주장 없음이어야 한다")
                self.assertEqual(empty["parse_fail"], 1.0)
                self.assertGreater(
                    empty["change_f1_floor"],
                    0.0,
                    "눈금 자체는 계속 나와야 한다 — 그 행의 바닥은 여전히 0 이 아니다",
                )

    def test_minimal_nonempty_prediction_still_gets_the_floor(self):
        """⚠️ **빈 예측을 0 으로 만들어도 바닥은 0 이 되지 않는다.**

        요소가 단 하나라도 있으면 나머지 current 요소 전부가 그대로 DELETED 주장이
        되므로, 퇴화 점수는 `change_f1_floor` 바로 아래로 돌아온다. 이 테스트는
        "빈 예측 규칙을 고쳤으니 이제 0 이 바닥"이라는 오독을 코드가 반박하게 만든다.
        실측(정본 probe): index 0.500/0.5714(87.5%) · pos 0.2857/0.3333(85.7%).
        """
        for mode, (cur, gt) in self.DELS.items():
            with self.subTest(mode=mode):
                r = _sd.compute_state_diff(self.MAXDEL[mode], gt, cur, mode)
                self.assertEqual(r["parse_fail"], 0.0, "비어 있지 않은 예측이다")
                self.assertGreater(r["change_f1_strict"], 0.0)
                self.assertLessEqual(
                    r["change_f1_strict"],
                    r["change_f1_floor"],
                    "닫힌식 바닥은 이 전략의 상한이어야 한다",
                )
                self.assertGreater(
                    r["change_f1_strict"],
                    0.5 * r["change_f1_floor"],
                    "쓰레기 하나만 내도 바닥의 절반 이상을 가져간다",
                )
                copier = _sd.compute_state_diff(cur, gt, cur, mode)
                self.assertEqual(copier["change_f1_strict"], 0.0)

    def test_floor_is_independent_of_the_prediction(self):
        """`change_f1_floor` 는 `(current, gt)` 만의 함수다 — 예측과 무관한 상수.

        닫힌 식(hits=|gt_deleted|, n_pred=n_cur)으로 계산하므로 어떤 예측을 넣어도
        같은 행이면 같은 값이어야 한다. 어긋나면 눈금이 지표와 다른 것을 잰다.
        """
        for mode, (cur, gt) in self.DELS.items():
            with self.subTest(mode=mode):
                base = _sd.compute_state_diff("", gt, cur, mode)["change_f1_floor"]
                self.assertGreater(base, 0.0)
                for pred in (cur, gt, "", self.MAXDEL[mode]):
                    r = _sd.compute_state_diff(pred, gt, cur, mode)
                    self.assertEqual(r["change_f1_floor"], base)

    def test_index_floor_is_hand_checkable(self):
        """손으로 검산되는 값 하나를 못으로 박는다.

        cur 는 root node/button/p 3요소다 (full 집합 — 옛 화이트리스트는 root 를
        뽑지 않아 2요소였고, 그 값은 `tests/test_element_set.py` 가 legacy 로 고정한다).
        GT 는 p 를 지우고 span 을 넣는다. root 는 흡수 텍스트가 "old row" → "fresh
        banner" 로 통째로 갈리고 그 cost 가 정확히 1.5 라 index 임계에서 떨어진다
        → root 도 DELETED(cur 쪽) + ADDED(gt 쪽)로 잡힌다.
        바닥: hits=|gt_deleted|={root,p}=2, n_pred=n_cur=3,
              n_gt=ADDED{root,span}2+DELETED2=4 → prec 2/3, rec 2/4, f1 = 0.5714

        최대삭제 예측(root+select 2요소): pred_deleted={root,button,p},
        pred_changed={root',select} → n_pred=5, hits=|{root,button,p} ∩ {root,p}|=2
        → prec 2/5, rec 2/4, f1 = 0.4444
        """
        cur, gt = self.DEL_IDX
        empty = _sd.compute_state_diff("", gt, cur, "index")
        self.assertEqual(empty["n_cur"], 3)
        self.assertEqual(empty["n_change_gt"], 4)
        self.assertEqual(empty["n_change_pred"], 0, "빈 예측은 주장이 없다")
        self.assertEqual(empty["change_f1_floor"], 0.5714)
        self.assertEqual(empty["change_f1_strict"], 0.0)

        maxdel = _sd.compute_state_diff(self.MAXDEL["index"], gt, cur, "index")
        self.assertEqual(maxdel["n_change_pred"], 5)
        self.assertEqual(maxdel["change_f1_strict"], 0.4444)

    def test_floor_defined_on_exactly_the_same_rows_as_change_f1_strict(self):
        """정의 구간이 어긋나면 두 평균의 분모가 달라져 나란히 못 읽는다."""
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                still = _sd.compute_state_diff(cur, cur, cur, mode)
                self.assertIsNone(still["change_f1_strict"], "양쪽 다 변화 없음 = 정의불능")
                self.assertIsNone(still["change_f1_floor"], "눈금도 같이 빠져야 한다")
        rows = [
            _sd.compute_state_diff(p, g, c, "index")
            for p, g, c in (
                ("", GT_IDX, CUR_IDX),
                (CUR_IDX, GT_IDX, CUR_IDX),
                (CUR_IDX, CUR_IDX, CUR_IDX),  # 변화 없는 행 — 양쪽 다 빠진다
            )
        ]
        agg = _sd.aggregate(rows)
        self.assertEqual(agg["n_change_f1_floor"], agg["n_change_f1_strict"])
        self.assertEqual(agg["n_change_f1_strict"], 2)
        # 옛 이름 alias 도 같은 값으로 함께 나와야 한다 (2026-08-04 개명, 기존 34개
        # leaf 재빌드를 피하기 위한 하위호환).
        self.assertEqual(agg["n_change_f1_null"], agg["n_change_f1_strict"])
        self.assertEqual(agg["n_change_f1"], agg["n_change_f1_strict"])

    def test_self_test_catches_a_probe_without_deletions(self):
        """`_PROBE` 에서 DELETED 가 빠지면 배선 self-test 가 터져야 한다.

        구 probe 의 pos 는 실제로 DELETED 가 0 이었다 — 그러면 빈 예측의 change_f1
        이 0.0 이 되어 이 축의 지배항을 self-test 가 **구조적으로 못 본다**.
        (구 index probe 는 DELETED 가 1 이었지만, 어느 쪽이든 빈 예측을 찍는
         assert 자체가 없어 퇴화 바닥은 검사되지 않았다.)
        """
        no_del = {
            "cur": '<node index="0"><button index="1" aria-label="OK"/></node>',
            "gt": '<node index="0"><button index="1" aria-label="OK"/>'
            '<p index="3">brand new</p></node>',
            "maxdel": _sd._PROBE["index"]["maxdel"],
        }
        self.assertEqual(
            _sd.summarize_diff(_sd.classify_diff(no_del["cur"], no_del["gt"], "index"))[
                "DELETED"
            ],
            0,
        )
        original = _sd._PROBE["index"]
        _sd._PROBE["index"] = no_del
        try:
            with self.assertRaises(_sd.StateDiffError) as ctx:
                _sd.assert_scorer_wired("index")
            # 빈 예측이 아니라 **최대삭제** probe 가 이 역할을 이어받았다 (2026-08-04):
            # 빈 예측은 이제 규칙상 0 이라 DELETED 유무를 구분하지 못한다.
            self.assertIn("최대삭제", str(ctx.exception))
        finally:
            _sd._PROBE["index"] = original

    def test_shipped_probe_exercises_deletions_in_both_modes(self):
        """정본 probe 자체가 두 모드 모두에서 DELETED 를 통과시키는가."""
        for mode in ("index", "pos"):
            with self.subTest(mode=mode):
                probe = _sd._PROBE[mode]
                counts = _sd.summarize_diff(
                    _sd.classify_diff(probe["cur"], probe["gt"], mode)
                )
                self.assertGreater(counts["DELETED"], 0, "빈 예측 퇴화를 못 본다")
                self.assertGreater(counts["ADDED"], 0, "copy_excess probe 가 죽는다")


class TestLooseAxis(unittest.TestCase):
    """loose(자리만) ↔ strict(내용까지) — 두 축의 갭이 실패 유형을 가른다.

    ScratchWorld 의 `F₁^pres` ↔ `F₁^VA` 대응. strict 하나만 보면 "자리를 못 찾은 것"과
    "자리는 찾고 내용이 틀린 것"이 같은 낮은 점수로 뭉뚱그려진다.
    """

    CASES = (
        ("copier", lambda c, g: c),
        ("perfect", lambda c, g: g),
        ("empty", lambda c, g: ""),
        ("partial", lambda c, g: PARTIALS["index"]),
    )

    def test_loose_is_an_upper_bound_on_strict(self):
        for mode, cur, gt in MODES:
            for name, make in self.CASES:
                if name == "partial" and mode != "index":
                    continue
                with self.subTest(mode=mode, case=name):
                    r = _sd.compute_state_diff(make(cur, gt), gt, cur, mode)
                    if r["change_f1_strict"] is None:
                        continue
                    self.assertGreaterEqual(
                        r["change_f1_loose"],
                        r["change_f1_strict"],
                        "τ 게이트를 빼면 hit 이 줄 수 없다",
                    )
                    self.assertGreaterEqual(
                        r["change_prec_loose"], r["change_prec_strict"]
                    )
                    self.assertGreaterEqual(
                        r["change_recall_loose"], r["change_recall_strict"]
                    )

    def test_stale_content_separates_the_two_axes(self):
        """자리는 맞고 내용이 틀린 예측 — loose 는 잡고 strict 는 못 잡아야 한다.

        이게 loose 축을 만든 이유다. 두 값이 항상 같이 움직이면 새 축은 정보가 없다.
        """
        # GT 는 p 를 "unread inbox 7" 로 바꾼다. 예측은 **바꾸긴 했는데 틀리게** 바꿨다
        # ("9"). cur 대비로는 MODIFIED 라 C_pred 에 들어가고, gt 의 p 와 매칭도 되지만
        # text_sim = |{unread,inbox}|/|{unread,inbox,9,7}| = 0.5 < τ(0.9) 다.
        # → loose 는 hit, strict 는 miss. **옛 텍스트를 그대로 두면 안 된다** — 그건
        #   cur 대비 UNCHANGED 라 애초에 C_pred 에 안 들어가 갭이 안 생긴다.
        wrong_value = (
            '<node index="0">'
            '<button index="1" aria-label="OK"/>'
            '<p index="2">unread inbox 9</p>'
            '<p index="3">brand new banner</p>'
            "</node>"
        )
        r = _sd.compute_state_diff(wrong_value, GT_IDX, CUR_IDX, "index")
        self.assertEqual(r["change_f1_loose"], 1.0, "자리는 둘 다 찾았다")
        self.assertLess(
            r["change_f1_strict"],
            r["change_f1_loose"],
            "내용이 틀린 자리를 loose 만 hit 으로 세야 갭이 생긴다",
        )

    def test_both_axes_are_defined_on_exactly_the_same_rows(self):
        """정의 구간이 갈리면 두 평균의 분모가 달라져 나란히 못 읽는다."""
        rows = []
        for mode, cur, gt in MODES:
            for pred in (cur, gt, "", PARTIALS[mode]):
                r = _sd.compute_state_diff(pred, gt, cur, mode)
                rows.append(r)
                self.assertEqual(
                    r["change_f1_strict"] is None,
                    r["change_f1_loose"] is None,
                    "한쪽만 None 이면 안 된다",
                )
            # 변화가 아예 없는 행 — 양쪽 다 None 이어야 한다
            still = _sd.compute_state_diff(cur, cur, cur, mode)
            rows.append(still)
            self.assertIsNone(still["change_f1_strict"])
            self.assertIsNone(still["change_f1_loose"])
        agg = _sd.aggregate(rows)
        self.assertEqual(agg["n_change_f1_loose"], agg["n_change_f1_strict"])
        self.assertEqual(agg["n_change_f1_floor"], agg["n_change_f1_strict"])


class TestDerivabilitySplit(unittest.TestCase):
    """`addmod_recall_derivable` / `addmod_recall_non_derivable` — action 유도성 축.

    GT_IDX 의 변화 셋: root(STRUCTURE, absorbed text 변화로 MODIFIED) +
    p"unread inbox 7"(action.text 와 동일 → ACTION_PAYLOAD) + p"brand new banner"
    (근거 없음 → NON_DERIVABLE). 셋 다 유도 가능/불가능이 실측으로 확정된 라벨이다
    (분류기 자체의 정확성은 `tests/test_diff_loss_v3.py` 소관 — 여기서는 그 라벨이
    `addmod_recall` 을 쪼개는 배선만 검증한다).
    """

    ACTION = {"action": "type", "text": "unread inbox 7"}
    PARTIAL_MISSES_NON_DERIVABLE = (
        '<node index="0">'
        '<button index="1" aria-label="OK"/>'
        '<p index="2">unread inbox 7</p>'
        "</node>"
    )

    def test_action_none_skips_split(self):
        """action 을 안 주면(기존 호출부) 분리축은 조용히 None — 하위호환."""
        row = _sd.compute_state_diff(GT_IDX, GT_IDX, CUR_IDX, "index")
        self.assertIsNone(row["addmod_recall_derivable"])
        self.assertIsNone(row["addmod_recall_non_derivable"])
        self.assertEqual(row["addmod_recall"], 1.0)

    def test_perfect_prediction_hits_both(self):
        row = _sd.compute_state_diff(
            GT_IDX, GT_IDX, CUR_IDX, "index", action=self.ACTION
        )
        self.assertEqual(row["addmod_recall"], 1.0)
        self.assertEqual(row["addmod_recall_derivable"], 1.0)
        self.assertEqual(row["addmod_recall_non_derivable"], 1.0)

    def test_split_separates_derivable_hit_from_non_derivable_miss(self):
        """모델이 action 유도 콘텐츠는 맞히고 근거 없는 신규 콘텐츠는 못 맞힌 사례.

        블렌드된 `addmod_recall` 만 보면 "0.67 = 어중간한 실패"로 읽히지만, 실제로는
        유도 가능한 것은 **전부** 맞히고 유도 불가능한 것은 **전부** 못 맞힌
        정반대의 두 극단이 섞인 값이다 — 이 축이 없으면 그 구분이 안 보인다.
        """
        row = _sd.compute_state_diff(
            self.PARTIAL_MISSES_NON_DERIVABLE,
            GT_IDX,
            CUR_IDX,
            "index",
            action=self.ACTION,
        )
        self.assertEqual(row["addmod_recall"], round(2 / 3, 4))
        self.assertEqual(row["addmod_recall_derivable"], 1.0)
        self.assertEqual(row["addmod_recall_non_derivable"], 0.0)

    def test_aggregate_carries_avg_and_n_keys(self):
        rows = [
            _sd.compute_state_diff(GT_IDX, GT_IDX, CUR_IDX, "index", action=self.ACTION),
            _sd.compute_state_diff(
                self.PARTIAL_MISSES_NON_DERIVABLE,
                GT_IDX,
                CUR_IDX,
                "index",
                action=self.ACTION,
            ),
        ]
        agg = _sd.aggregate(rows)
        self.assertEqual(agg["n_addmod_recall_derivable"], 2)
        self.assertEqual(agg["n_addmod_recall_non_derivable"], 2)
        self.assertEqual(agg["avg_addmod_recall_derivable"], 1.0)
        self.assertEqual(agg["avg_addmod_recall_non_derivable"], 0.5)


class TestNoChangeAccuracy(unittest.TestCase):
    """GT 가 current 와 같은 행에서는 **복사가 정답**이다.

    그 행에서는 change 축이 전부 None 이라, 이 열이 없으면 "화면이 안 바뀌는 step"
    구간의 성능을 아무도 재지 않는다.
    """

    def test_copying_is_correct_when_nothing_changed(self):
        for mode, cur, _ in MODES:
            with self.subTest(mode=mode):
                r = _sd.compute_state_diff(cur, cur, cur, mode)
                self.assertEqual(r["n_change_gt"], 0)
                self.assertEqual(r["no_change_acc"], 1.0)

    def test_inventing_a_change_is_wrong_when_nothing_changed(self):
        r = _sd.compute_state_diff(GT_IDX, CUR_IDX, CUR_IDX, "index")
        self.assertEqual(r["n_change_gt"], 0, "GT == current 인 행이어야 한다")
        self.assertGreater(r["n_change_pred"], 0, "예측은 변화를 지어냈다")
        self.assertEqual(r["no_change_acc"], 0.0)

    def test_generation_failure_is_not_credited_as_no_change(self):
        """⚠️ 빈 예측은 `n_change_pred == 0` 이지만 **정답이 아니다.**

        규칙 변경(빈 예측 = 주장 없음) 뒤 이 조건만 보면 아무것도 못 낸 모델이 1.0 을
        받는다 — 실측으로 EXP01 base 는 파싱 실패율 93.9% 인데 no_change_acc 가
        204/204 = 1.0 이 나왔다. 이 축은 "변화를 지어내지 않았나"가 아니라 **"화면을
        그대로 재현했나"** 를 묻는다.
        """
        for mode, cur, _ in MODES:
            with self.subTest(mode=mode):
                empty = _sd.compute_state_diff("", cur, cur, mode)
                self.assertEqual(empty["n_change_gt"], 0)
                self.assertEqual(empty["n_change_pred"], 0, "주장은 없다")
                self.assertEqual(
                    empty["no_change_acc"], 0.0, "그러나 재현하지 못했으므로 오답이다"
                )
                junk = _sd.compute_state_diff(
                    "no tags here whatsoever, just prose. " * 5, cur, cur, mode
                )
                self.assertEqual(junk["no_change_acc"], 0.0)

    def test_undefined_on_rows_that_do_change(self):
        for mode, cur, gt in MODES:
            with self.subTest(mode=mode):
                r = _sd.compute_state_diff(gt, gt, cur, mode)
                self.assertGreater(r["n_change_gt"], 0)
                self.assertIsNone(
                    r["no_change_acc"], "변화 있는 행까지 세면 분모가 뒤섞인다"
                )

    def test_denominator_is_reported(self):
        """`n_no_change_acc` 가 곧 '변화 없는 step 이 몇 행인가' (sparsity 통계)."""
        rows = [
            _sd.compute_state_diff(CUR_IDX, CUR_IDX, CUR_IDX, "index"),  # 무변화·정답
            _sd.compute_state_diff(GT_IDX, CUR_IDX, CUR_IDX, "index"),  # 무변화·오답
            _sd.compute_state_diff(GT_IDX, GT_IDX, CUR_IDX, "index"),  # 변화 있음
        ]
        agg = _sd.aggregate(rows)
        self.assertEqual(agg["n_no_change_acc"], 2)
        self.assertEqual(agg["avg_no_change_acc"], 0.5)


class TestParseFailure(unittest.TestCase):
    """"안 냈다" 와 "태그 없는 장문 쓰레기를 냈다" 는 채점기에게 같은 값이다.

    둘 다 `pred_els == []` 라 구분이 안 되는데, 실측에서 `predict` 58,303자인데 추출
    요소 0개인 행이 나왔다. `parse_fail_long` 이 유일한 구분 수단이다. 그리고 이 비율은
    `copy_excess` 의 **분모 손실량**이기도 하다 — 그 행들은 copy_excess 에서 빠진다.
    """

    LONG_JUNK = "this prediction has no parseable tags whatsoever. " * 5

    def test_empty_is_parse_fail_but_not_long(self):
        r = _sd.compute_state_diff("", GT_IDX, CUR_IDX, "index")
        self.assertEqual(r["parse_fail"], 1.0)
        self.assertEqual(r["parse_fail_long"], 0.0)

    def test_long_untagged_output_is_flagged_separately(self):
        self.assertGreater(len(self.LONG_JUNK), _sd.PARSE_FAIL_LONG_CHARS)
        r = _sd.compute_state_diff(self.LONG_JUNK, GT_IDX, CUR_IDX, "index")
        self.assertEqual(r["n_pred"], 0, "태그가 없어 요소 0개")
        self.assertEqual(r["parse_fail"], 1.0)
        self.assertEqual(r["parse_fail_long"], 1.0)
        self.assertEqual(r["change_f1_strict"], 0.0, "파싱 실패 = 오답")

    def test_normal_prediction_is_not_flagged(self):
        r = _sd.compute_state_diff(GT_IDX, GT_IDX, CUR_IDX, "index")
        self.assertEqual(r["parse_fail"], 0.0)
        self.assertEqual(r["parse_fail_long"], 0.0)

    def test_rate_reaches_aggregate(self):
        rows = [
            _sd.compute_state_diff("", GT_IDX, CUR_IDX, "index"),
            _sd.compute_state_diff(self.LONG_JUNK, GT_IDX, CUR_IDX, "index"),
            _sd.compute_state_diff(GT_IDX, GT_IDX, CUR_IDX, "index"),
            _sd.compute_state_diff(CUR_IDX, GT_IDX, CUR_IDX, "index"),
        ]
        agg = _sd.aggregate(rows)
        self.assertEqual(agg["parse_fail_rate"], 0.5)
        self.assertEqual(agg["parse_fail_long_rate"], 0.25)
        # 그 두 행이 copy_excess 에서 빠졌다는 사실이 분모로 드러나야 한다.
        self.assertEqual(agg["n_copy_excess"], 2)
        self.assertEqual(agg["total"], 4)

    def test_extractor_crash_is_also_counted(self):
        """추출 자체가 예외를 던지는 경로에서도 실패로 세야 한다.

        조기 반환 경로에서 안 채우면 `aggregate` 의 `r.get(k, 0.0)` 이 조용히 0 으로
        세어 실패율이 **과소보고**된다 — 실패가 감춰지는 방향이라 더 위험하다.
        """
        orig = _hungarian_eval.extract_elements

        def boom(*a, **k):
            raise ValueError("parser exploded")

        _hungarian_eval.extract_elements = boom
        try:
            r = _sd.compute_state_diff(self.LONG_JUNK, GT_IDX, CUR_IDX, "index")
        finally:
            _hungarian_eval.extract_elements = orig
        self.assertEqual(r["parse_fail"], 1.0)
        self.assertEqual(r["parse_fail_long"], 1.0)


class TestLegacyAlias(unittest.TestCase):
    """2026-08-04 개명 — 옛 키가 같은 값으로 함께 나와야 한다.

    ⚠️ alias 는 "이름만 다른 같은 수"지만, **`change_f1` 은 정의가 바뀌었다**
    (빈 예측 규칙). 그래서 산출물에 `metrics_schema` 를 박아 옛 파일과 구분한다.
    """

    def test_avg_and_n_are_both_aliased(self):
        rows = [
            _sd.compute_state_diff(p, GT_IDX, CUR_IDX, "index")
            for p in (CUR_IDX, GT_IDX, "", PARTIAL_IDX)
        ]
        agg = _sd.aggregate(rows)
        for new, old in _sd._LEGACY_KEY_ALIAS.items():
            with self.subTest(key=new):
                self.assertIn(f"avg_{old}", agg, "avg_ alias 가 빠졌다")
                self.assertEqual(agg[f"avg_{old}"], agg[f"avg_{new}"])
                # `n_` 도 반드시 — 조건부 분모(예: n_diff_prec)는 해석에 필수다.
                self.assertIn(f"n_{old}", agg, "n_ alias 가 빠졌다")
                self.assertEqual(agg[f"n_{old}"], agg[f"n_{new}"])

    def test_schema_version_is_stamped(self):
        metrics = {"overall": {}, "in_domain": {}, "out_of_domain": {}}
        stamped = _sd.stamp_schema(metrics)
        self.assertEqual(stamped["metrics_schema"], _sd.METRICS_SCHEMA)
        self.assertEqual(_sd.METRICS_SCHEMA, "2026-08-04")


class TestWiringGuardActuallyFires(unittest.TestCase):
    """가드가 **조용히 통과**하지 않는지 — 개명·규칙변경에서 가장 위험한 자리.

    `.get()` 으로 키를 읽으면 개명 누락 시 `None` 이 돌아오고 비교가 조용히 False 가
    되어 가드가 무력화된다. 그래서 (a) 실제로 예외가 나는지, (b) 메시지에 **실제
    숫자값**이 찍히는지 둘 다 본다 — 숫자가 찍힌다는 건 키를 제대로 읽었다는 뜻이다.
    """

    def _assert_fires(self, mode="index"):
        with self.assertRaises(_sd.StateDiffError) as ctx:
            _sd.assert_scorer_wired(mode)
        return str(ctx.exception)

    def test_empty_prediction_regression_is_caught(self):
        """빈 예측 규칙이 되돌아가면(= 다시 DELETED 주장) 가드가 잡아야 한다."""
        orig = _sd._classify_from_els

        def legacy(cur_els, next_els, match_mode="index", **kw):
            # 옛 동작 재현 — `empty_next_is_deletion` 을 무시한다.
            kw.pop("empty_next_is_deletion", None)
            return orig(cur_els, next_els, match_mode, **kw)

        _sd._classify_from_els = legacy
        try:
            msg = self._assert_fires()
        finally:
            _sd._classify_from_els = orig
        self.assertIn("빈 예측", msg)
        self.assertNotIn("None", msg, "키를 못 읽어 None 이 찍히면 가드가 무력하다")

    def test_dead_change_axis_is_caught(self):
        """compute_change_items 가 상수 0 을 돌려주면 잡아야 한다."""
        orig = _sd.compute_change_items
        _sd.compute_change_items = lambda *a, **k: {
            **{k2: 0.0 for k2 in _sd._CHANGE_METRIC_KEYS},
            "n_change_gt": 0,
            "n_change_pred": 0,
        }
        try:
            msg = self._assert_fires()
        finally:
            _sd.compute_change_items = orig
        # 어느 probe 에서 걸리든 상관없다 — 중요한 건 (a) 터졌고 (b) 메시지에 **실제
        # 숫자**가 찍혔다는 것이다. 키를 못 읽었으면 여기에 `None` 이 찍힌다.
        self.assertIn("change_f1_strict=0.0", msg)
        self.assertNotIn("None", msg)

    def test_inverted_tau_gate_is_caught(self):
        """loose < strict 로 뒤집히면(τ 가 반대로 걸리면) 잡아야 한다."""
        orig = _sd.compute_change_items

        def flipped(*a, **k):
            r = orig(*a, **k)
            if r.get("change_f1_loose") is not None:
                r["change_f1_loose"] = 0.0
                r["change_f1_strict"] = 1.0
            return r

        _sd.compute_change_items = flipped
        try:
            msg = self._assert_fires()
        finally:
            _sd.compute_change_items = orig
        self.assertNotIn("None", msg)

    def test_guard_message_carries_real_numbers(self):
        """정상 경로에서 probe 값이 실제로 계산되는지 — 숫자를 직접 확인."""
        for mode in ("index", "pos"):
            with self.subTest(mode=mode):
                probe = _sd._PROBE[mode]
                maxdel = _sd.compute_state_diff(
                    probe["maxdel"], probe["gt"], probe["cur"], mode
                )
                self.assertIsNotNone(maxdel["change_f1_strict"])
                self.assertGreater(maxdel["change_f1_strict"], 0.0)
                self.assertGreater(maxdel["n_pred"], 0, "maxdel probe 가 비면 안 된다")
                self.assertEqual(maxdel["copy_rate_pred"], 0.0, "아무것도 안 베꼈다")


class TestStratumInvariant(unittest.TestCase):
    """recall 계열은 정본 hungarian_rec 의 층 분해여야 한다.

    diff 부분집합을 따로 매칭하면 UNCHANGED 에 붙었어야 할 예측 요소가 MODIFIED 로
    재배정되며 recall 이 부풀어 이 항등식이 깨진다. 그 구현 실수를 여기서 잡는다.
    """

    def _check(self, pred, gt, cur, mode, **opts):
        r = _sd.compute_state_diff(pred, gt, cur, mode, **opts)
        if not r["n_gt"]:
            return
        hits = sum(
            (r[f"{t}_recall"] or 0.0) * r[f"n_gt_{t}"]
            for t in ("added", "modified", "unchanged")
        )
        canonical = _hungarian_eval.compute_hungarian_acc(pred, gt, mode, **opts)
        self.assertAlmostEqual(
            hits / r["n_gt"],
            canonical["hungarian_rec"],
            places=3,
            msg=f"층 분해가 hungarian_rec 과 어긋남 ({mode}, {opts})",
        )

    def test_invariant_holds(self):
        for mode, cur, gt in MODES:
            for name, pred in (("copy", cur), ("perfect", gt), ("empty", "")):
                with self.subTest(mode=mode, pred=name):
                    self._check(pred, gt, cur, mode)

    def test_invariant_holds_under_strict_pos(self):
        """`--strict-pos-match` 를 켜도 항등식이 유지돼야 한다.

        두 채점기가 같은 매칭 함수를 쓰므로, 임계 플래그가 한쪽에만 흘러가면 여기서
        깨진다. 그래서 플래그를 전역이 아니라 인자로 흘린다.

        `alpha`→`omega` 픽스처는 그 임계가 **실제로 무는** 자리다: `_text_sim` 이 0 이라
        cost 가 정확히 1.5 여서 1.7 에서는 붙고(MODIFIED) 1.5 에서는 떨어진다
        (ADDED + DELETED). 플래그가 죽어 있으면 아래 sanity 가 먼저 터진다.
        """
        cur_swap = (
            '<node bounds="[0,0][10,10]" point="[5,5]">'
            '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
            '<p bounds="[6,6][9,9]" point="[7,7]">alpha</p></node>'
        )
        gt_swap = (
            '<node bounds="[0,0][10,10]" point="[5,5]">'
            '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
            '<p bounds="[6,6][9,9]" point="[7,7]">omega</p></node>'
        )
        loose = _sd.summarize_diff(_sd.classify_diff(cur_swap, gt_swap, "pos"))
        strict = _sd.summarize_diff(
            _sd.classify_diff(cur_swap, gt_swap, "pos", strict_pos=True)
        )
        # MODIFIED 2 = p + 그 텍스트를 흡수한 root (full 집합). root 의 텍스트는
        # "OK|alpha" → "OK|omega" 라 sim 이 0 이 아니어서 두 임계 모두에서 붙는다 —
        # 그래서 strict 쪽 (ADDED, DELETED) 는 p 하나 몫인 (1, 1) 그대로다.
        self.assertEqual((loose["MODIFIED"], loose["DELETED"]), (2, 0))
        self.assertEqual((strict["ADDED"], strict["DELETED"]), (1, 1), "임계가 물었다")

        _, cur, gt = MODES[1]  # pos
        for name, c, g in (("standard", cur, gt), ("text-swap", cur_swap, gt_swap)):
            for pred_name, pred in (("copy", c), ("perfect", g), ("empty", "")):
                with self.subTest(fixture=name, pred=pred_name):
                    self._check(pred, g, c, "pos", strict_pos=True)


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

    판정은 **토큰 실측**이다. 앞서 쓰던 두 기준이 2026-08-03 전수 실측으로 반증됐다:
    mtime 은 날짜가 아니라 실행 경로가 절단을 정하므로 틀렸고, 문자 길이는 **양방향으로**
    틀렸다 — 공백 꼬리가 붙은 29,286자 예측이 실은 정확히 1024 토큰이었고(절단을 놓침),
    5,567자짜리 짧은 예측은 길이가 아니라 37.9% 가 1024 토큰이라는 사실 때문에 절단이다.
    그래서 픽스처를 **토큰 길이 분포**로 쓰고, tokenizer 는 주입해 갈아끼운다.
    """

    def setUp(self):
        self._real_token_lengths = _sd._token_lengths
        _sd._MODE_SHARE_CACHE.clear()

    def tearDown(self):
        _sd._token_lengths = self._real_token_lengths
        _sd._MODE_SHARE_CACHE.clear()

    def _write(self, tmpdir, name, tok_lens, when=None):
        """토큰 길이가 `tok_lens` 인 prediction jsonl 을 만든다.

        `predict` 본문에 길이를 그대로 심고 가짜 tokenizer 가 그것을 되읽는다 — 실제
        tokenizer 를 로드하면 테스트가 로컬 HF 캐시에 묶인다. 경로에 `eval/<model>/` 을
        끼우는 것도 계약의 일부다: `truncated_reason` 은 경로에서 모델을 해석한다.
        """
        leaf = Path(tmpdir) / "eval" / "qwen3-vl-8b" / "stage1_eval" / "base"
        leaf.mkdir(parents=True, exist_ok=True)
        p = leaf / name
        with p.open("w", encoding="utf-8") as f:
            for n in tok_lens:
                f.write(json.dumps({"predict": f"TOK:{n}", "label": "y" * 100}) + "\n")
        if when is not None:
            ts = when.timestamp()
            os.utime(p, (ts, ts))
        _sd._token_lengths = lambda texts, key: [int(t.split(":")[1]) for t in texts]
        return str(p)

    def test_truncated_prediction_is_flagged(self):
        """절단 지문 — 1024 에 대량으로 몰리고, 재토크나이즈 드리프트 행이 1개 섞인다.

        EXP02 `lora_world-model/epoch-1` 재현(37.9% @1024, 최대 1025). `max_tok <= 1024`
        를 조건에 넣으면 이 leaf 가 드리프트 행 하나 때문에 정상으로 빠진다 — 그래서
        판별량은 최댓값이 아니라 **모드 비율**이다.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            cut = self._write(
                d, "generated_predictions_id.jsonl", [1024] * 379 + [700] * 620 + [1025]
            )
            self.assertIsNotNone(_sd.truncated_reason(cut))

    def test_intact_prediction_is_not_flagged(self):
        """정상 leaf — 새 예산(12288)까지 퍼져 있고 1024 는 우연히 2행뿐."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            ok = self._write(
                d,
                "generated_predictions_id.jsonl",
                [1024, 1024] + [430, 900, 2500, 7000, 12288] * 400,
            )
            self.assertIsNone(_sd.truncated_reason(ok))

    def test_max_just_over_budget_is_not_by_itself_truncation(self):
        """`tok_max` 가 1024 를 갓 넘었다는 것만으로는 절단도 정상도 아니다.

        판별량은 최댓값이 아니라 **1024 빈의 밀도**다. 같은 `tok_max=1025` 라도
        1024 에 몰림이 없으면 정상, 몰려 있으면 절단이어야 한다 — 최댓값만 보면
        두 경우가 구분되지 않는다.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            # tok_max=1025 인데 1024 는 1행뿐 → 그냥 짧게 생성한 leaf
            ok = self._write(d, "ok.jsonl", [1025, 1024] + [300, 500, 700, 900] * 250)
            # tok_max=1025 로 같지만 1024 에 37.9% 가 몰림 → 절단 (EXP02 ep1 실측)
            cut = self._write(d, "cut.jsonl", [1025] + [1024] * 1136 + [700] * 1863)
            self.assertIsNone(_sd.truncated_reason(ok))
            self.assertIsNotNone(_sd.truncated_reason(cut))

    def test_one_truncated_split_blocks_the_leaf(self):
        """한쪽 split 만 절단이어도 leaf 전체를 막는다 (섹션 합산이 오염되므로)."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            ok = self._write(d, "generated_predictions_ood.jsonl", [500] * 100)
            cut = self._write(d, "generated_predictions_id.jsonl", [1024] * 100)
            self.assertIsNone(_sd.truncated_reason(ok))
            self.assertIsNotNone(_sd.truncated_reason(ok, cut))

    def test_recent_mtime_does_not_excuse_a_truncated_file(self):
        """mtime 이 수정 시각 이후여도 토큰이 몰려 있으면 막는다 — EXP03(2026-07-26~27)."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            after = _sd.MAX_NEW_TOKENS_FIX_UTC + timedelta(days=1)
            cut = self._write(
                d, "generated_predictions_id.jsonl", [1024] * 70 + [600] * 30, after
            )
            self.assertIsNotNone(_sd.truncated_reason(cut))

    def test_old_mtime_does_not_condemn_an_intact_file(self):
        """옛 mtime 규칙이 부당하게 막던 케이스 — EXP01 `qwen2.5-vl-7b_ratio73/base`."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            before = _sd.MAX_NEW_TOKENS_FIX_UTC - timedelta(days=60)
            ok = self._write(
                d, "generated_predictions_id.jsonl", [617, 12288] * 50, before
            )
            self.assertIsNone(_sd.truncated_reason(ok))

    def test_threshold_is_the_named_constant(self):
        """임계가 상수로 빠져 있고 비교 방향이 `>=` 라는 것을 같이 고정한다."""
        import tempfile

        share = _sd.TRUNC_MODE_SHARE
        at = int(round(share * 1000))
        with tempfile.TemporaryDirectory() as d:
            below = self._write(
                d, "below.jsonl", [1024] * (at - 1) + [700] * (1000 - at + 1)
            )
            exactly = self._write(d, "at.jsonl", [1024] * at + [700] * (1000 - at))
            self.assertIsNone(_sd.truncated_reason(below))
            self.assertIsNotNone(_sd.truncated_reason(exactly))

    def test_short_action_predictions_are_never_truncation(self):
        """action/stage2 leaf 는 예측이 수십 토큰이라 모드가 1024 에 생기지 않는다.

        실측 97 leaf 의 share@1024 최댓값이 0.0030 이라 임계 0.05 에 한참 못 미친다.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            action = self._write(
                d, "generated_predictions_id.jsonl", [29, 82, 91, 112] * 250
            )
            self.assertIsNone(_sd.truncated_reason(action))

    def test_unreadable_file_falls_back_to_mtime(self):
        """토큰을 못 재면(빈 파일 등) 옛 mtime 기준으로 물러선다 — 유일한 mtime 용도.

        **문자 길이로는 물러서지 않는다** — 문자 기준은 잘린 것을 통과시킨다.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            leaf = Path(d) / "eval" / "qwen3-vl-8b"
            leaf.mkdir(parents=True)
            old, new = leaf / "old_empty.jsonl", leaf / "new_empty.jsonl"
            for p, delta in ((old, -1), (new, 1)):
                p.write_text("")
                ts = (_sd.MAX_NEW_TOKENS_FIX_UTC + timedelta(hours=delta)).timestamp()
                os.utime(p, (ts, ts))
            self.assertIsNotNone(_sd.truncated_reason(str(old)))
            self.assertIsNone(_sd.truncated_reason(str(new)))

    def test_unresolvable_model_falls_back_to_mtime(self):
        """경로에서 모델을 못 읽으면 tokenizer 를 못 고른다 — 그때도 mtime 폴백."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "generated_predictions_id.jsonl"  # eval/<model>/ 없음
            p.write_text(json.dumps({"predict": "TOK:1024"}) + "\n")
            ts = (_sd.MAX_NEW_TOKENS_FIX_UTC - timedelta(hours=1)).timestamp()
            os.utime(p, (ts, ts))
            self.assertIsNone(_sd._model_key_for(str(p)))
            self.assertIsNotNone(_sd.truncated_reason(str(p)))

    def test_model_key_is_read_from_path(self):
        """`qwen3-vl-8b_ratio37` 같은 변형 접미사도 같은 모델로 해석돼야 한다."""
        base = "/x/outputs/AndroidControl_EXP01/eval/{}/stage1_eval/base/p.jsonl"
        self.assertEqual(_sd._model_key_for(base.format("qwen3-vl-8b")), "qwen3-vl-8b")
        self.assertEqual(
            _sd._model_key_for(base.format("qwen3-vl-8b_ratio37")), "qwen3-vl-8b"
        )
        self.assertEqual(
            _sd._model_key_for(base.format("qwen2.5-vl-3b_v1")), "qwen2.5-vl-3b"
        )
        self.assertIsNone(_sd._model_key_for(base.format("llama-3")))

    def test_probe_forget_model_key_resolves_via_exp_mapping(self):
        """`probe_forget` 경로엔 모델명이 없다 — EXP 디렉터리명으로 매핑해야 한다.

        `outputs/<EXP>/probe_forget/<variant>/...` 는 `eval/<model>/` 규약 밖이라
        prefix 매칭이 그냥은 None 을 돌려준다(→ mtime 폴백으로 조용히 내려간다). 그걸
        막는 `_PROBE_FORGET_MODEL` 분기를 고정한다.
        """
        probe_forget = (
            "outputs/AndroidControl_EXP07/probe_forget/mergeO-v1-s2ep1/"
            "generated_predictions_id.jsonl"
        )
        self.assertEqual(_sd._model_key_for(probe_forget), "qwen2.5-vl-3b")

        # 기존 `eval/<model>/` 규약은 이 분기 신설 후에도 그대로 살아 있어야 한다.
        eval_path = (
            "outputs/AndroidControl_EXP01/eval/qwen3-vl-8b_ratio37/stage1_eval/"
            "base/on-AC_EXP01-state/generated_predictions_id.jsonl"
        )
        self.assertEqual(_sd._model_key_for(eval_path), "qwen3-vl-8b")
        eval_v1 = "outputs/AndroidControl_EXP05/eval/qwen2.5-vl-3b_v1/base/p.jsonl"
        self.assertEqual(_sd._model_key_for(eval_v1), "qwen2.5-vl-3b")

        # `_PROBE_FORGET_MODEL` 에 없는 EXP 의 probe_forget → None. 조용한 오답보다
        # mtime 폴백이 낫다.
        unmapped = "outputs/AndroidControl_EXP01/probe_forget/some-variant/p.jsonl"
        self.assertIsNone(_sd._model_key_for(unmapped))

    def test_missing_and_empty_paths_are_ignored(self):
        self.assertIsNone(_sd.truncated_reason(None, "", "/nonexistent/x.jsonl"))

    def test_constant_is_shared_with_compare_site(self):
        """경계 상수가 두 벌이면 언젠가 조용히 갈린다."""
        cs = __import__("_compare_site")
        self.assertIs(cs.MAX_NEW_TOKENS_FIX_UTC, _sd.MAX_NEW_TOKENS_FIX_UTC)


class TestCachedSnapshot(unittest.TestCase):
    """`_cached_snapshot` — HF 캐시에서 tokenizer 파일이 실제로 있는 snapshot 을 고른다.

    `Qwen/Qwen2.5-VL-3B-Instruct` 캐시에 `config.json` 이 없어
    `AutoTokenizer.from_pretrained(repo_id)` 가 오프라인에서 OSError 로 죽는 문제의
    우회로다(2026-08-11 실측). "HF repo 껍데기 선생성"이 이 저장소의 실제 운영
    함정으로 기록돼 있으므로, **껍데기 캐시(파일이 없는 snapshot 디렉터리)를
    tokenizer 로 착각하면 안 된다**는 것이 이 테스트들의 핵심이다.
    """

    def setUp(self):
        import huggingface_hub.constants as _hf_const

        self._hf_const = _hf_const
        self._orig_cache = _hf_const.HF_HUB_CACHE

    def tearDown(self):
        self._hf_const.HF_HUB_CACHE = self._orig_cache

    def test_finds_snapshot_that_has_tokenizer_json(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._hf_const.HF_HUB_CACHE = d
            snap = (
                Path(d) / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / "abc"
            )
            snap.mkdir(parents=True)
            (snap / "tokenizer.json").write_text("{}")
            self.assertEqual(
                _sd._cached_snapshot("Qwen/Qwen2.5-VL-3B-Instruct"), str(snap)
            )

    def test_finds_snapshot_that_has_only_tokenizer_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._hf_const.HF_HUB_CACHE = d
            snap = (
                Path(d) / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / "abc"
            )
            snap.mkdir(parents=True)
            (snap / "tokenizer_config.json").write_text("{}")
            self.assertEqual(
                _sd._cached_snapshot("Qwen/Qwen2.5-VL-3B-Instruct"), str(snap)
            )

    def test_shell_snapshot_without_tokenizer_files_is_not_mistaken_for_one(self):
        """껍데기 캐시(파일 없는 snapshot)를 건너뛰고, 진짜 tokenizer 를 가진 snapshot 을 고른다.

        `sorted()` 로 순회하므로 사전순으로 먼저 오는 "aaa" 를 일부러 껍데기로 두고
        "bbb" 에만 tokenizer 파일을 둔다 — 첫 snapshot 을 그냥 집는 구현이었다면
        여기서 잡힌다.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._hf_const.HF_HUB_CACHE = d
            repo = Path(d) / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots"
            shell = repo / "aaa"
            shell.mkdir(parents=True)
            (shell / "config.json").write_text("{}")  # tokenizer 파일 없음 — 껍데기
            real = repo / "bbb"
            real.mkdir(parents=True)
            (real / "tokenizer.json").write_text("{}")
            self.assertEqual(
                _sd._cached_snapshot("Qwen/Qwen2.5-VL-3B-Instruct"), str(real)
            )

    def test_snapshot_dir_with_no_tokenizer_files_at_all_returns_none(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._hf_const.HF_HUB_CACHE = d
            snap = (
                Path(d) / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / "abc"
            )
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            self.assertIsNone(_sd._cached_snapshot("Qwen/Qwen2.5-VL-3B-Instruct"))

    def test_missing_repo_dir_returns_none(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self._hf_const.HF_HUB_CACHE = d
            self.assertIsNone(_sd._cached_snapshot("Qwen/Qwen2.5-VL-3B-Instruct"))


class _RepoIdFailsAutoTokenizer:
    """repo id 호출은 실패하고 snapshot 경로 호출만 성공하는 가짜 `AutoTokenizer`.

    `Qwen/Qwen2.5-VL-3B-Instruct` 캐시에 `config.json` 이 없어 repo id 호출이
    오프라인에서 OSError 로 죽는 실측 상황(2026-08-11)의 재현이다.
    """

    @staticmethod
    def from_pretrained(name_or_path, local_files_only=True):
        if name_or_path == _sd._MODEL_ID["qwen2.5-vl-3b"]:
            raise OSError("config.json 없음 (실측 재현)")
        return _FakeTokenizerForLengths()


class _FakeTokenizerForLengths:
    """토큰 수를 문자열 길이로 그대로 돌려주는 가짜 tokenizer.

    실제 로드 없이 `_token_lengths` 의 재시도 배선만 검증하려는 것이라, "몇 토큰인가"
    자체는 관심사가 아니다 — 문자열 길이를 그대로 쓰면 값 검증이 쉬워진다.
    """

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[0] * len(t) for t in texts]}


class TestTokenLengthsFallback(unittest.TestCase):
    """`_token_lengths` 가 repo id 실패 시 `_cached_snapshot` 경로로 재시도하는가.

    재시도가 없으면 `_mode_share` 가 None 을 돌려주고 `truncated_reason` 이 mtime
    폴백으로 내려가 절단 판정을 **안 한 것**이 "정상"으로 보고된다 (2026-08-11 실측:
    3B 계열 EXP05·EXP07 22 leaf 전량이 이 상태였다).
    """

    def setUp(self):
        import transformers

        self._transformers = transformers
        self._orig_auto_tokenizer = transformers.AutoTokenizer
        self._orig_cached_snapshot = _sd._cached_snapshot
        # 모듈 전역 캐시 — 비우지 않으면 앞선 테스트의 tokenizer 를 재사용해 이
        # 테스트가 아무것도 검증하지 못한다.
        _sd._TOKENIZERS.clear()

    def tearDown(self):
        self._transformers.AutoTokenizer = self._orig_auto_tokenizer
        _sd._cached_snapshot = self._orig_cached_snapshot
        _sd._TOKENIZERS.clear()

    def test_retries_via_cached_snapshot_when_repo_id_call_fails(self):
        self._transformers.AutoTokenizer = _RepoIdFailsAutoTokenizer
        _sd._cached_snapshot = lambda model_id: "/fake/snapshot/path"
        lens = _sd._token_lengths(["abc", "de"], "qwen2.5-vl-3b")
        self.assertEqual(lens, [3, 2], "snapshot 경로 재시도로 길이를 돌려줘야 한다")

    def test_original_exception_propagates_when_no_cached_snapshot(self):
        """`_cached_snapshot` 이 None 이면 조용히 삼키지 않고 원래 예외가 그대로 올라온다."""
        self._transformers.AutoTokenizer = _RepoIdFailsAutoTokenizer
        _sd._cached_snapshot = lambda model_id: None
        with self.assertRaises(OSError):
            _sd._token_lengths(["abc"], "qwen2.5-vl-3b")


class TestSanitySignals(unittest.TestCase):
    def test_unclosed_root_flags_truncation(self):
        self.assertEqual(_sd._unclosed_root(GT_IDX), 0.0)
        self.assertEqual(_sd._unclosed_root('<node index="0"><p index="1">cut'), 1.0)
        self.assertEqual(_sd._unclosed_root('<node index="0"/>'), 0.0)
        self.assertEqual(_sd._unclosed_root("no tags at all"), 1.0)


class TestRawCurrentState(unittest.TestCase):
    """AC_EXP08 관측성 3 포맷(full/masked/dropped) — copy 진단은 **마스킹 전 원본**으로.

    프롬프트의 current 를 그대로 쓰면 `dropped`(`(none)`)는 `copy_rate` 가 0 으로
    붕괴하고 `masked` 는 구조적으로 낮은 값을 받아 **가짜 개선**이 된다. GT 레코드의
    `raw_current_state` 가 그 축을 원본으로 되돌린다.
    """

    # `scripts/build_wm_formats.py` 의 실제 마커/본문 (INLINE_MARKER · DROPPED_BODY).
    FULL_PROMPT = (
        "system\nrole\nuser\nCurrent UI State (FULL):\n"
        + CUR_POS
        + "\n\n[Screenshot]\n"
        '\nAction:\n<action>{"action": "click", "coordinate": [3, 3]}</action>\n'
        "assistant\n"
    )
    DROPPED_PROMPT = (
        "system\nrole\nuser\nCurrent UI State (NOT PROVIDED):\n(none)\n\n[Screenshot]\n"
        '\nAction:\n<action>{"action": "click", "coordinate": [3, 3]}</action>\n'
        "assistant\n"
    )

    def _pair(self, prompt: str, raw: str | None):
        gt = {"messages": [{"value": "sys"}, {"value": prompt}, {"value": GT_POS}]}
        if raw is not None:
            gt["raw_current_state"] = raw
        # 예측 = current 를 그대로 베낀 복사기. copy_rate_pred 가 1.0 이어야 정상이다.
        return [gt], [{"prompt": prompt, "predict": CUR_POS}]

    def test_absent_field_keeps_the_prompt_path(self):
        gts, preds = self._pair(self.FULL_PROMPT, None)
        m = _sd.evaluate_pairs(gts, preds, "pos")
        self.assertEqual(m["n_current_state_raw"], 0)
        self.assertEqual(m["n_current_state_prompt"], 1)
        self.assertEqual(_sd.stamp_schema(m)["current_state_source"], "prompt")

    def test_raw_field_wins_and_is_stamped(self):
        gts, preds = self._pair(self.FULL_PROMPT, CUR_POS)
        m = _sd.evaluate_pairs(gts, preds, "pos")
        self.assertEqual(m["n_current_state_raw"], 1)
        self.assertEqual(m["n_current_state_prompt"], 0)
        self.assertEqual(_sd.stamp_schema(m)["current_state_source"], "raw_field")

    def test_same_current_gives_identical_numbers(self):
        """필드가 **같은 XML** 을 실어 나르면 수치가 한 자리도 달라지면 안 된다 —
        경로만 바뀌었지 채점 입력은 같기 때문이다."""
        without = _sd.evaluate_pairs(*self._pair(self.FULL_PROMPT, None), "pos")
        with_raw = _sd.evaluate_pairs(*self._pair(self.FULL_PROMPT, CUR_POS), "pos")
        skip = {"n_current_state_raw", "n_current_state_prompt"}
        self.assertEqual(
            {k: v for k, v in without.items() if k not in skip},
            {k: v for k, v in with_raw.items() if k not in skip},
        )

    def test_dropped_format_collapses_without_the_field(self):
        """`(none)` 은 빈 청크가 아니라 **문자열**이라 파싱 가드에 안 걸린다 —
        예외 대신 `copy_rate` 0 이라는 그럴듯한 오답표가 나온다. 그게 이 필드의 이유다."""
        m = _sd.evaluate_pairs(*self._pair(self.DROPPED_PROMPT, None), "pos")
        self.assertEqual(m["avg_copy_rate_pred"], 0.0)
        self.assertEqual(m["avg_copy_rate_gt"], 0.0)
        self.assertEqual(m["avg_copy_excess"], 0.0)

    def test_dropped_format_is_restored_by_the_field(self):
        m = _sd.evaluate_pairs(*self._pair(self.DROPPED_PROMPT, CUR_POS), "pos")
        self.assertEqual(
            m["avg_copy_rate_pred"], 1.0, "복사기는 current 를 전부 재현한다"
        )
        self.assertGreater(m["avg_copy_rate_gt"], 0.0)
        self.assertGreater(m["avg_copy_excess"], 0.0, "베낀 만큼이 초과분으로 잡힌다")

    def test_empty_field_falls_back_to_the_prompt(self):
        m = _sd.evaluate_pairs(*self._pair(self.FULL_PROMPT, "   "), "pos")
        self.assertEqual(m["n_current_state_prompt"], 1)

    def test_mixed_sources_raise(self):
        """빌더가 전량 채우거나 전량 비우거나 둘 중 하나다 — 혼재는 배선 고장이다."""
        gts, preds = self._pair(self.FULL_PROMPT, CUR_POS)
        gts2, preds2 = self._pair(self.FULL_PROMPT, None)
        with self.assertRaises(_sd.StateDiffError) as cm:
            _sd.evaluate_pairs(gts + gts2, preds + preds2, "pos")
        self.assertIn("혼재", str(cm.exception))

    def test_split_mode_catches_cross_file_mixing(self):
        """ID 는 필드가 있고 OOD 는 없는 경우 — `overall` 이 합쳐 채점하므로 걸린다."""
        gt_id, pr_id = self._pair(self.FULL_PROMPT, CUR_POS)
        gt_ood, pr_ood = self._pair(self.FULL_PROMPT, None)
        with self.assertRaises(_sd.StateDiffError):
            _sd.build_metrics(gt_id, pr_id, gt_ood, pr_ood, "pos")

    def test_source_stamp_is_derived_per_section(self):
        gt_id, pr_id = self._pair(self.FULL_PROMPT, CUR_POS)
        gt_ood, pr_ood = self._pair(self.DROPPED_PROMPT, CUR_POS)
        metrics = _sd.build_metrics(gt_id, pr_id, gt_ood, pr_ood, "pos")
        self.assertEqual(_sd.stamp_schema(metrics)["current_state_source"], "raw_field")
        for sec in ("overall", "in_domain", "out_of_domain"):
            self.assertEqual(metrics[sec]["n_current_state_prompt"], 0)

    def test_unknown_when_nothing_was_scored(self):
        """행이 없으면 출처를 지어내지 않는다."""
        self.assertEqual(_sd._current_state_source({"overall": {}}), "unknown")


if __name__ == "__main__":
    unittest.main()
