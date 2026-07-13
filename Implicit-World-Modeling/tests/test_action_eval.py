"""Regression tests for scripts/_action_eval.py Step Accuracy metric.

스키마는 실제 AndroidControl_3 action schema 를 따른다:
  키 `action_type` · click/long_press→index · scroll→direction(+index) ·
  open_app→app_name · input_text→text(+index) · finish→status/answer ·
  wait/navigate_back/navigate_home → 검증 필드 없음.
`_atype` 는 구 `type` 키 fallback 을 유지하므로 일부 케이스는 fallback 도 검증.

Run:
    cd GUI-Model
    python -m unittest tests.test_action_eval -v
    # or with pytest if available:
    pytest tests/test_action_eval.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import importlib

_action_eval = importlib.import_module("_action_eval")
evaluate_single = _action_eval.evaluate_single
evaluate_predictions = _action_eval.evaluate_predictions
evaluate_pairs = _action_eval.evaluate_pairs
parse_action = _action_eval.parse_action


def _gt_pred_jsonl(pairs):
    """Build temp gt/pred jsonl files from list of (gt_action, pred_text) tuples."""
    gt_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    pr_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for gt, pred_text in pairs:
        gt_f.write(
            json.dumps({"messages": [{"from": "gpt", "value": json.dumps(gt)}]}) + "\n"
        )
        pr_f.write(json.dumps({"predict": pred_text}) + "\n")
    gt_f.close()
    pr_f.close()
    return Path(gt_f.name), Path(pr_f.name)


class StepAccuracySingle(unittest.TestCase):
    """Per-type field_match rules for evaluate_single (AndroidControl_3 schema)."""

    # ── click ────────────────────────────────────────────────────────────
    def test_click_correct(self):
        r = evaluate_single(
            {"action_type": "click", "index": "12"},
            {"action_type": "click", "index": "12"},
        )
        self.assertTrue(r["step_correct"])

    def test_click_wrong_index(self):
        r = evaluate_single(
            {"action_type": "click", "index": "12"},
            {"action_type": "click", "index": "9"},
        )
        self.assertFalse(r["step_correct"])

    def test_click_int_vs_str(self):
        # str(12) == str("12") — robustness
        r = evaluate_single(
            {"action_type": "click", "index": "12"},
            {"action_type": "click", "index": 12},
        )
        self.assertTrue(r["step_correct"])

    # ── long_press ───────────────────────────────────────────────────────
    def test_long_press_correct(self):
        r = evaluate_single(
            {"action_type": "long_press", "index": "5"},
            {"action_type": "long_press", "index": "5"},
        )
        self.assertTrue(r["step_correct"])

    def test_long_press_wrong(self):
        r = evaluate_single(
            {"action_type": "long_press", "index": "5"},
            {"action_type": "long_press", "index": "6"},
        )
        self.assertFalse(r["step_correct"])

    # ── scroll ───────────────────────────────────────────────────────────
    def test_scroll_correct(self):
        r = evaluate_single(
            {"action_type": "scroll", "direction": "down"},
            {"action_type": "scroll", "direction": "down"},
        )
        self.assertTrue(r["step_correct"])

    def test_scroll_wrong_direction(self):
        r = evaluate_single(
            {"action_type": "scroll", "direction": "down"},
            {"action_type": "scroll", "direction": "up"},
        )
        self.assertFalse(r["step_correct"])

    def test_scroll_normalization(self):
        r = evaluate_single(
            {"action_type": "scroll", "direction": "down"},
            {"action_type": "scroll", "direction": " DOWN "},
        )
        self.assertTrue(r["step_correct"])

    # ── open_app (app_name) ──────────────────────────────────────────────
    def test_open_app_correct(self):
        r = evaluate_single(
            {"action_type": "open_app", "app_name": "Gmail"},
            {"action_type": "open_app", "app_name": "Gmail"},
        )
        self.assertTrue(r["step_correct"])

    def test_open_app_wrong_app(self):
        r = evaluate_single(
            {"action_type": "open_app", "app_name": "Gmail"},
            {"action_type": "open_app", "app_name": "Calendar"},
        )
        self.assertFalse(r["step_correct"])

    def test_open_app_normalization(self):
        # 'Gmail' vs ' gmail ' → _norm 정규화로 일치
        r = evaluate_single(
            {"action_type": "open_app", "app_name": "Gmail"},
            {"action_type": "open_app", "app_name": " gmail "},
        )
        self.assertTrue(r["step_correct"])

    def test_open_app_nested_params_fallback(self):
        # _pval 은 top-level 우선, 부재 시 params 로 fallback (관용 처리 고정)
        r = evaluate_single(
            {"action_type": "open_app", "params": {"app_name": "Gmail"}},
            {"action_type": "open_app", "app_name": "gmail"},
        )
        self.assertTrue(r["step_correct"])

    # ── input_text (text; index 는 무시) ─────────────────────────────────
    def test_input_text_correct(self):
        r = evaluate_single(
            {"action_type": "input_text", "index": "3", "text": "hello"},
            {"action_type": "input_text", "index": "3", "text": "hello"},
        )
        self.assertTrue(r["step_correct"])

    def test_input_text_mismatch(self):
        r = evaluate_single(
            {"action_type": "input_text", "index": "3", "text": "hello"},
            {"action_type": "input_text", "index": "3", "text": "world"},
        )
        self.assertFalse(r["step_correct"])

    def test_input_text_normalization(self):
        r = evaluate_single(
            {"action_type": "input_text", "index": "3", "text": "Hello"},
            {"action_type": "input_text", "index": "9", "text": " hello "},
        )
        # text 만 비교 — index 가 달라도 통과
        self.assertTrue(r["step_correct"])

    # ── navigate_back / navigate_home / wait (type-only) ─────────────────
    def test_navigate_back_correct(self):
        r = evaluate_single(
            {"action_type": "navigate_back"}, {"action_type": "navigate_back"}
        )
        self.assertTrue(r["step_correct"])

    def test_navigate_back_type_wrong(self):
        r = evaluate_single(
            {"action_type": "navigate_back"}, {"action_type": "click", "index": "1"}
        )
        self.assertFalse(r["step_correct"])

    def test_navigate_home_correct(self):
        r = evaluate_single(
            {"action_type": "navigate_home"}, {"action_type": "navigate_home"}
        )
        self.assertTrue(r["step_correct"])

    def test_wait_correct(self):
        r = evaluate_single({"action_type": "wait"}, {"action_type": "wait"})
        self.assertTrue(r["step_correct"])

    # ── finish (type-only, status/answer 비교 안 함) ─────────────────────
    def test_finish_correct(self):
        r = evaluate_single(
            {"action_type": "finish", "status": "complete", "answer": "42"},
            {"action_type": "finish", "status": "complete", "answer": "42"},
        )
        self.assertTrue(r["step_correct"])

    def test_finish_status_irrelevant(self):
        # finish 는 type-only 정책 → status/answer 가 달라도 정답
        r = evaluate_single(
            {"action_type": "finish", "status": "complete", "answer": "42"},
            {"action_type": "finish"},
        )
        self.assertTrue(r["step_correct"])

    # ── 공통: type 불일치 / pred=None / unknown ─────────────────────────
    def test_type_mismatch_zero(self):
        r = evaluate_single(
            {"action_type": "click", "index": "1"},
            {"action_type": "scroll", "direction": "down"},
        )
        self.assertFalse(r["step_correct"])
        self.assertFalse(r["type_correct"])

    def test_pred_none(self):
        r = evaluate_single({"action_type": "click", "index": "1"}, None)
        self.assertFalse(r["step_correct"])

    def test_unknown_gt_type(self):
        # 알 수 없는 type — type 일치는 하나 field_match 미정의 → step False
        r = evaluate_single(
            {"action_type": "spell_cast", "magic": "fireball"},
            {"action_type": "spell_cast", "magic": "fireball"},
        )
        self.assertFalse(r["step_correct"])
        self.assertTrue(r["type_correct"])

    def test_legacy_type_key_fallback(self):
        # _atype 는 구 'type' 키도 인식 (action_type 부재 시 fallback)
        r = evaluate_single(
            {"type": "click", "index": "7"}, {"type": "click", "index": "7"}
        )
        self.assertTrue(r["step_correct"])
        self.assertTrue(r["type_correct"])

    # ── 추가 pin-down 케이스 ─────────────────────────────────────────────
    def test_click_both_index_none(self):
        # 양쪽 index=None → str(None)==str(None) 로 True (현 구현 동작 고정)
        r = evaluate_single(
            {"action_type": "click", "index": None},
            {"action_type": "click", "index": None},
        )
        self.assertTrue(r["step_correct"])
        self.assertTrue(r["has_index_check"])

    def test_scroll_both_direction_missing(self):
        # 양쪽 direction 부재 → _norm(None)=='' 동치 → True
        r = evaluate_single({"action_type": "scroll"}, {"action_type": "scroll"})
        self.assertTrue(r["step_correct"])
        self.assertTrue(r["has_dir_check"])

    def test_open_app_both_app_missing(self):
        # 양쪽 app_name 부재 → _pval None → _norm 동치 True
        r = evaluate_single({"action_type": "open_app"}, {"action_type": "open_app"})
        self.assertTrue(r["step_correct"])
        self.assertTrue(r["has_app_check"])

    def test_type_case_insensitive(self):
        # 'click' vs 'CLICK' → lower() 정규화로 type_correct True
        r = evaluate_single(
            {"action_type": "click", "index": "3"},
            {"action_type": "CLICK", "index": "3"},
        )
        self.assertTrue(r["step_correct"])
        self.assertTrue(r["type_correct"])

    def test_type_whitespace_not_stripped(self):
        # 현 구현은 type 에 strip 하지 않음 (lower 만) — 회귀 방지용 고정
        r = evaluate_single(
            {"action_type": "click", "index": "3"},
            {"action_type": " click ", "index": "3"},
        )
        self.assertFalse(r["type_correct"])
        self.assertFalse(r["step_correct"])

    def test_unknown_type_has_checks_all_false(self):
        # unknown type 분기에서 has_*_check 가 어느 것도 켜지지 않아야 함
        r = evaluate_single(
            {"action_type": "spell_cast"}, {"action_type": "spell_cast"}
        )
        self.assertFalse(r["has_index_check"])
        self.assertFalse(r["has_dir_check"])
        self.assertFalse(r["has_app_check"])
        self.assertFalse(r["has_text_check"])

    def test_pred_none_all_flags_false(self):
        # pred_action=None → parsed 포함 모든 플래그 False
        r = evaluate_single({"action_type": "click", "index": "1"}, None)
        self.assertFalse(r["parsed"])
        self.assertFalse(r["type_correct"])
        self.assertFalse(r["step_correct"])
        self.assertFalse(r["has_index_check"])
        self.assertFalse(r["has_dir_check"])
        self.assertFalse(r["has_app_check"])
        self.assertFalse(r["has_text_check"])


class StepAccuracyAggregate(unittest.TestCase):
    """evaluate_predictions 집계 로직."""

    def test_micro_macro_aggregation(self):
        pairs = [
            # 5 click: 4 correct (80%)
            (
                {"action_type": "click", "index": "1"},
                '{"action_type":"click","index":"1"}',
            ),
            (
                {"action_type": "click", "index": "2"},
                '{"action_type":"click","index":"2"}',
            ),
            (
                {"action_type": "click", "index": "3"},
                '{"action_type":"click","index":"3"}',
            ),
            (
                {"action_type": "click", "index": "4"},
                '{"action_type":"click","index":"4"}',
            ),
            (
                {"action_type": "click", "index": "5"},
                '{"action_type":"click","index":"X"}',
            ),  # wrong
            # 2 scroll: 1 correct (50%)
            (
                {"action_type": "scroll", "direction": "down"},
                '{"action_type":"scroll","direction":"down"}',
            ),
            (
                {"action_type": "scroll", "direction": "up"},
                '{"action_type":"scroll","direction":"down"}',
            ),  # wrong
            # 1 navigate_back: 1 correct
            ({"action_type": "navigate_back"}, '{"action_type":"navigate_back"}'),
        ]
        gt_p, pr_p = _gt_pred_jsonl(pairs)
        try:
            m = evaluate_predictions(str(gt_p), str(pr_p))
        finally:
            gt_p.unlink()
            pr_p.unlink()

        self.assertEqual(m["total"], 8)
        # micro SA = 6/8 = 0.75
        self.assertAlmostEqual(m["step_accuracy"], 6 / 8, places=4)
        # macro SA = mean(click=0.8, scroll=0.5, navigate_back=1.0) = 0.7666...
        self.assertAlmostEqual(
            m["macro_step_accuracy"], (0.8 + 0.5 + 1.0) / 3, places=4
        )
        # type_acc = 8/8 = 1.0 (모두 type 맞음)
        self.assertAlmostEqual(m["type_accuracy"], 1.0, places=4)
        # cond_index_acc: click 5건 중 정답 4
        self.assertAlmostEqual(m["cond_index_acc"], 4 / 5, places=4)
        # cond_dir_acc: scroll 2건 중 정답 1
        self.assertAlmostEqual(m["cond_dir_acc"], 1 / 2, places=4)
        # parse_rate = 1.0
        self.assertAlmostEqual(m["parse_rate"], 1.0, places=4)
        # per_type 키 존재
        for t in ("click", "scroll", "navigate_back"):
            self.assertIn(t, m["per_type"])
            self.assertIn("step_acc", m["per_type"][t])
            self.assertIn("count", m["per_type"][t])
        # bounds-related 키 부재
        self.assertNotIn("avg_bounds_iou", m)
        self.assertNotIn("cond_bounds_iou", m)

    def test_parse_failure_zero(self):
        pairs = [
            ({"action_type": "click", "index": "1"}, "this is not json"),
            (
                {"action_type": "click", "index": "2"},
                '{"action_type":"click","index":"2"}',
            ),
        ]
        gt_p, pr_p = _gt_pred_jsonl(pairs)
        try:
            m = evaluate_predictions(str(gt_p), str(pr_p))
        finally:
            gt_p.unlink()
            pr_p.unlink()
        self.assertAlmostEqual(m["step_accuracy"], 0.5, places=4)
        self.assertAlmostEqual(m["parse_rate"], 0.5, places=4)

    def test_codefence_parsing(self):
        pairs = [
            (
                {"action_type": "click", "index": "1"},
                '```json\n{"action_type":"click","index":"1"}\n```',
            ),
        ]
        gt_p, pr_p = _gt_pred_jsonl(pairs)
        try:
            m = evaluate_predictions(str(gt_p), str(pr_p))
        finally:
            gt_p.unlink()
            pr_p.unlink()
        self.assertAlmostEqual(m["step_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(m["parse_rate"], 1.0, places=4)

    def test_thought_action_wrapper_parsing(self):
        # 실제 데이터셋 포맷: <thought>...</thought>\n<action>{...}</action>
        wrapped = (
            "<thought>open the app</thought>\n"
            '<action>{"action_type":"open_app","app_name":"Gmail"}</action>'
        )
        pairs = [({"action_type": "open_app", "app_name": "Gmail"}, wrapped)]
        gt_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        pr_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        gt_f.write(json.dumps({"messages": [{"from": "gpt", "value": wrapped}]}) + "\n")
        pr_f.write(json.dumps({"predict": wrapped}) + "\n")
        gt_f.close()
        pr_f.close()
        try:
            m = evaluate_predictions(gt_f.name, pr_f.name)
        finally:
            Path(gt_f.name).unlink()
            Path(pr_f.name).unlink()
        # gt 도 parse_action 으로 <action> JSON 추출 → 채점 가능
        self.assertAlmostEqual(m["step_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(m["parse_rate"], 1.0, places=4)

    def test_length_mismatch_warns(self):
        # gt 3, pred 2 → 짧은 쪽에 맞춰 자르되 metrics 는 계산되어야 함
        gt_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        pr_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for i in range(3):
            gt_f.write(
                json.dumps(
                    {
                        "messages": [
                            {
                                "from": "gpt",
                                "value": '{"action_type":"click","index":"'
                                + str(i)
                                + '"}',
                            }
                        ]
                    }
                )
                + "\n"
            )
        for i in range(2):
            pr_f.write(
                json.dumps(
                    {"predict": '{"action_type":"click","index":"' + str(i) + '"}'}
                )
                + "\n"
            )
        gt_f.close()
        pr_f.close()
        try:
            m = evaluate_predictions(gt_f.name, pr_f.name)
        finally:
            Path(gt_f.name).unlink()
            Path(pr_f.name).unlink()
        # 2 건만 채점되어야 함
        self.assertEqual(m["total"], 2)

    # ── 추가 pin-down 케이스 ─────────────────────────────────────────────
    def test_unknown_type_lowers_macro(self):
        # click 2건 (정답) + unknown 1건 → macro = (1.0 + 0.0) / 2 = 0.5
        pairs = [
            (
                {"action_type": "click", "index": "1"},
                '{"action_type":"click","index":"1"}',
            ),
            (
                {"action_type": "click", "index": "2"},
                '{"action_type":"click","index":"2"}',
            ),
            (
                {"action_type": "spell_cast", "magic": "fire"},
                '{"action_type":"spell_cast","magic":"fire"}',
            ),
        ]
        gt_p, pr_p = _gt_pred_jsonl(pairs)
        try:
            m = evaluate_predictions(str(gt_p), str(pr_p))
        finally:
            gt_p.unlink()
            pr_p.unlink()
        self.assertEqual(m["total"], 3)
        self.assertAlmostEqual(m["type_accuracy"], 1.0, places=4)
        # micro SA = 2/3
        self.assertAlmostEqual(m["step_accuracy"], 2 / 3, places=4)
        # macro = (click_step_acc + spell_cast_step_acc) / 2 = (1.0 + 0.0) / 2
        self.assertAlmostEqual(m["macro_step_accuracy"], 0.5, places=4)
        self.assertIn("spell_cast", m["per_type"])
        self.assertEqual(m["per_type"]["spell_cast"]["count"], 1)
        self.assertEqual(m["per_type"]["spell_cast"]["step_acc"], 0.0)

    def test_cond_acc_zero_when_no_type(self):
        # navigate_back 1건만 → 모든 cond_*_acc 는 n=0 로 0.0
        pairs = [
            ({"action_type": "navigate_back"}, '{"action_type":"navigate_back"}'),
        ]
        gt_p, pr_p = _gt_pred_jsonl(pairs)
        try:
            m = evaluate_predictions(str(gt_p), str(pr_p))
        finally:
            gt_p.unlink()
            pr_p.unlink()
        self.assertAlmostEqual(m["cond_index_acc"], 0.0, places=4)
        self.assertAlmostEqual(m["cond_dir_acc"], 0.0, places=4)
        self.assertAlmostEqual(m["cond_app_acc"], 0.0, places=4)
        self.assertAlmostEqual(m["cond_text_acc"], 0.0, places=4)

    def test_output_key_fallback(self):
        # pred_entry 에 predict 대신 output 키 — fallback 고정
        gt_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        pr_f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        gt_f.write(
            json.dumps(
                {
                    "messages": [
                        {"from": "gpt", "value": '{"action_type":"click","index":"5"}'}
                    ]
                }
            )
            + "\n"
        )
        pr_f.write(json.dumps({"output": '{"action_type":"click","index":"5"}'}) + "\n")
        gt_f.close()
        pr_f.close()
        try:
            m = evaluate_predictions(gt_f.name, pr_f.name)
        finally:
            Path(gt_f.name).unlink()
            Path(pr_f.name).unlink()
        self.assertAlmostEqual(m["step_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(m["parse_rate"], 1.0, places=4)

    def test_per_type_count_sum_equals_total(self):
        # per_type 전체 count 합 == total 불변식
        pairs = [
            (
                {"action_type": "click", "index": "1"},
                '{"action_type":"click","index":"1"}',
            ),
            (
                {"action_type": "click", "index": "2"},
                '{"action_type":"click","index":"2"}',
            ),
            (
                {"action_type": "click", "index": "3"},
                '{"action_type":"click","index":"X"}',
            ),
            (
                {"action_type": "scroll", "direction": "down"},
                '{"action_type":"scroll","direction":"down"}',
            ),
            (
                {"action_type": "scroll", "direction": "up"},
                '{"action_type":"scroll","direction":"up"}',
            ),
            (
                {"action_type": "finish", "status": "complete"},
                '{"action_type":"finish"}',
            ),
        ]
        gt_p, pr_p = _gt_pred_jsonl(pairs)
        try:
            m = evaluate_predictions(str(gt_p), str(pr_p))
        finally:
            gt_p.unlink()
            pr_p.unlink()
        self.assertEqual(
            sum(v["count"] for v in m["per_type"].values()),
            m["total"],
        )

    def test_finish_status_different_value_still_correct(self):
        # finish 는 type-only 정책 → status 가 달라도 step_correct True
        pairs = [
            (
                {"action_type": "finish", "status": "complete"},
                '{"action_type":"finish","status":"failed"}',
            ),
        ]
        gt_p, pr_p = _gt_pred_jsonl(pairs)
        try:
            m = evaluate_predictions(str(gt_p), str(pr_p))
        finally:
            gt_p.unlink()
            pr_p.unlink()
        self.assertAlmostEqual(m["step_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(m["type_accuracy"], 1.0, places=4)


class ParseAction(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(
            parse_action('{"action_type":"click","index":"1"}'),
            {"action_type": "click", "index": "1"},
        )

    def test_codefence_json(self):
        self.assertEqual(
            parse_action('```json\n{"action_type":"click","index":"2"}\n```'),
            {"action_type": "click", "index": "2"},
        )

    def test_codefence_no_lang(self):
        self.assertEqual(
            parse_action('```\n{"action_type":"click","index":"3"}\n```'),
            {"action_type": "click", "index": "3"},
        )

    def test_garbage(self):
        self.assertIsNone(parse_action("hello world"))

    def test_empty(self):
        self.assertIsNone(parse_action(""))

    def test_inline_extraction_from_garbage(self):
        # 앞뒤 garbage 사이에 단순 객체 — 최종 fallback regex (\{[^{}]*\}) 분기
        self.assertEqual(
            parse_action('blah blah {"action_type":"click","index":"1"} trailing'),
            {"action_type": "click", "index": "1"},
        )

    def test_thought_action_wrapper(self):
        # <action>{...}</action> 안 flat JSON → inline regex fallback 으로 추출
        self.assertEqual(
            parse_action(
                "<thought>do it</thought>\n"
                '<action>{"action_type":"open_app","app_name":"Gmail"}</action>'
            ),
            {"action_type": "open_app", "app_name": "Gmail"},
        )

    def test_codefence_multiline_whitespace(self):
        # 코드펜스 앞뒤 다중 개행 + fence 내부 공백 라인 포함
        self.assertEqual(
            parse_action('\n\n```json\n\n{"action_type":"click","index":"9"}\n\n```\n'),
            {"action_type": "click", "index": "9"},
        )


class IdOodAggregation(unittest.TestCase):
    """evaluate_pairs 로 ID + OOD 통합 집계가 올바른지 검증."""

    def _mk_pairs(self, specs):
        gt_entries, pred_entries = [], []
        for gt, pred_text in specs:
            gt_entries.append({"messages": [{"from": "gpt", "value": json.dumps(gt)}]})
            pred_entries.append({"predict": pred_text})
        return gt_entries, pred_entries

    def test_overall_equals_concat_of_id_and_ood(self):
        id_specs = [
            (
                {"action_type": "click", "index": "3"},
                '{"action_type":"click","index":"3"}',
            ),  # correct
            (
                {"action_type": "click", "index": "4"},
                '{"action_type":"click","index":"9"}',
            ),  # wrong index
        ]
        ood_specs = [
            (
                {"action_type": "scroll", "direction": "down"},
                '{"action_type":"scroll","direction":"up"}',
            ),  # wrong dir
            (
                {"action_type": "navigate_back"},
                '{"action_type":"navigate_back"}',
            ),  # correct
        ]
        gt_id, pr_id = self._mk_pairs(id_specs)
        gt_ood, pr_ood = self._mk_pairs(ood_specs)

        m_id = evaluate_pairs(gt_id, pr_id)
        m_ood = evaluate_pairs(gt_ood, pr_ood)
        m_all = evaluate_pairs(gt_id + gt_ood, pr_id + pr_ood)

        self.assertEqual(m_id["total"], 2)
        self.assertEqual(m_ood["total"], 2)
        self.assertEqual(m_all["total"], 4)
        # overall step_accuracy = (1 + 1) / 4 = 0.5
        self.assertAlmostEqual(m_all["step_accuracy"], 0.5, places=4)
        # in_domain: 1 of 2 correct → 0.5; out_of_domain: 1 of 2 → 0.5
        self.assertAlmostEqual(m_id["step_accuracy"], 0.5, places=4)
        self.assertAlmostEqual(m_ood["step_accuracy"], 0.5, places=4)

    def test_per_type_counts_merge_across_splits(self):
        id_specs = [
            (
                {"action_type": "click", "index": "1"},
                '{"action_type":"click","index":"1"}',
            ),
            (
                {"action_type": "click", "index": "2"},
                '{"action_type":"click","index":"2"}',
            ),
        ]
        ood_specs = [
            (
                {"action_type": "click", "index": "9"},
                '{"action_type":"click","index":"0"}',
            ),
        ]
        gt_id, pr_id = self._mk_pairs(id_specs)
        gt_ood, pr_ood = self._mk_pairs(ood_specs)
        m_all = evaluate_pairs(gt_id + gt_ood, pr_id + pr_ood)
        self.assertEqual(m_all["per_type"]["click"]["count"], 3)
        # 2 correct out of 3
        self.assertAlmostEqual(m_all["per_type"]["click"]["step_acc"], 2 / 3, places=4)


if __name__ == "__main__":
    unittest.main()
