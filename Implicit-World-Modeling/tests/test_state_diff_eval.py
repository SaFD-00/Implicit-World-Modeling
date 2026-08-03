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
