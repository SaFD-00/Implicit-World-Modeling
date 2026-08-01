"""Regression tests for scripts/_compare_site.py (eval_viewer --site 의 사이트 빌더).

여기서 지키는 것은 **조용히 틀릴 수 있는 부분**들이다:

1. 프롬프트 파서 — GT 프롬프트는 두 계열이다.
     A) EXP01~04 : '## Current State' / '## Next State' / '## Action'
     B) EXP05~07 : 'Current UI State:' / 'Next UI State:' / 'Action:' /
                   'Task Instruction:' / 'Action History:'
   한 계열만 보는 파서는 다른 계열에서 빈 문자열을 돌려주고 화면이 통째로 비는데,
   그게 정확히 woa 필터 사고(2026-07-30)의 실패 모드였다.

2. 레이아웃 판정 — stage 번호로 추정하면 안 된다. EXP07 의 stage1 `-action` 은
   `# Mode: NEXT_ACTION_PREDICTION` 이라 two-state 역추론이 아니라 stage2 와 같은
   (instruction + history + current state → thought + action) 이다.

3. Hungarian 채점기 배선 — `_hungarian_eval` 의 bs4/솔버는 지연 로드라서, 초기화 없이
   부르면 내부 except 가 예외를 삼켜 **전 행 0점**이 조용히 나온다.

Run:
    pytest tests/test_compare_site.py -v
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_compare_site = importlib.import_module("_compare_site")
parse_prompt = _compare_site.parse_prompt
detect_layout = _compare_site.detect_layout

# ── 두 계열의 실물 축약본 (2026-08-01 산출물에서 발췌) ────────────────────
FAMILY_A_STATE = """system
# Role
You are a mobile GUI agent.
user

## Current State
<div index="0">
  <button index="1" aria-label="OK"/>
</div>

## Action
{"action_type":"click","index":"13"}
assistant
"""

FAMILY_A_ACTION = """system
# Role
Given two consecutive UI states, infer the action.
user
## Current State
<div index="0"/>

## Next State
<div index="1"/>
assistant
"""

FAMILY_B_STATE = """system
# Role
You are a mobile GUI agent.
user
Current UI State:
<node bounds="[0,0][840,1876]" point="[420,938]"/>

[Screenshot]


Action:
<action>{"action": "click", "coordinate": [663, 282]}</action>
assistant
"""

FAMILY_B_ACTION = """system
# Role
Infer the executed action.
user
Current UI State:
<node bounds="[0,0][10,10]" point="[5,5]"/>

[Current UI Screenshot]


Next UI State:
<node bounds="[0,0][10,10]" point="[5,5]"><p bounds="[1,1][2,2]" point="[1,1]">x</p></node>

[Next UI Screenshot]

assistant
"""

FAMILY_B_AGENT = """system
# Mode: NEXT_ACTION_PREDICTION
# Role
You are a mobile GUI agent.
user
Task Instruction:
Open the Google Fit app and track my running activity.

Action History:
Step 1. [Thought] Opened the app drawer.

Current UI State:
<node bounds="[0,0][840,1876]" point="[420,938]"/>

[Screenshot]

assistant
"""


class TestPromptParser(unittest.TestCase):
    def test_family_a_state(self):
        sec = parse_prompt(FAMILY_A_STATE)
        self.assertIn('<button index="1"', sec["current_state"])
        self.assertEqual(sec["action"], '{"action_type":"click","index":"13"}')
        self.assertNotIn("next_state", sec)

    def test_family_a_action(self):
        sec = parse_prompt(FAMILY_A_ACTION)
        self.assertEqual(sec["current_state"], '<div index="0"/>')
        self.assertEqual(sec["next_state"], '<div index="1"/>')

    def test_family_b_state(self):
        sec = parse_prompt(FAMILY_B_STATE)
        self.assertTrue(sec["current_state"].startswith("<node"))
        self.assertIn('"action": "click"', sec["action"])

    def test_family_b_action(self):
        sec = parse_prompt(FAMILY_B_ACTION)
        self.assertIn("<node", sec["current_state"])
        self.assertIn("<p", sec["next_state"])

    def test_family_b_agent(self):
        sec = parse_prompt(FAMILY_B_AGENT)
        self.assertEqual(
            sec["instruction"], "Open the Google Fit app and track my running activity."
        )
        self.assertIn("Step 1.", sec["history"])
        self.assertIn("<node", sec["current_state"])

    def test_screenshot_markers_are_stripped(self):
        """[Screenshot] / [Current UI Screenshot] 는 상태 XML 에 섞이면 안 된다."""
        for prompt in (FAMILY_B_STATE, FAMILY_B_ACTION, FAMILY_B_AGENT):
            for value in parse_prompt(prompt).values():
                self.assertNotIn("Screenshot]", value)

    def test_required_sections_present_for_each_layout(self):
        cases = [
            (FAMILY_A_STATE, "state"),
            (FAMILY_B_STATE, "state"),
            (FAMILY_A_ACTION, "action"),
            (FAMILY_B_ACTION, "action"),
            (FAMILY_B_AGENT, "stage2"),
        ]
        for prompt, layout in cases:
            sec = parse_prompt(prompt)
            for name in _compare_site.REQUIRED_SECTIONS[layout]:
                self.assertTrue(
                    sec.get(name, "").strip(), f"{layout}: {name} 누락 — {sec.keys()}"
                )

    def test_unknown_prompt_yields_empty_sections(self):
        """모르는 형식은 빈 dict — 호출부가 '파싱 실패'로 집계해 터뜨린다."""
        self.assertEqual(parse_prompt("system\nfoo\nuser\nbar\nassistant\n"), {})


class TestLayoutDetection(unittest.TestCase):
    def test_state_kind_is_always_state(self):
        self.assertEqual(detect_layout(FAMILY_B_STATE, "state"), "state")

    def test_two_state_action(self):
        self.assertEqual(detect_layout(FAMILY_A_ACTION, "action"), "action")
        self.assertEqual(detect_layout(FAMILY_B_ACTION, "action"), "action")

    def test_agent_style_action_is_stage2_layout(self):
        """EXP07 stage1 -action 은 stage 번호가 1 이어도 stage2 레이아웃이다."""
        self.assertEqual(detect_layout(FAMILY_B_AGENT, "action"), "stage2")


class TestScorerWiring(unittest.TestCase):
    """카드 점수는 정본 채점기 값이어야 한다 — 배선이 끊기면 전 행 0점이 된다."""

    def test_state_scorer_wired_both_modes(self):
        for mode in ("index", "pos"):
            _compare_site.assert_state_scorer_wired(mode)

    def test_identical_state_scores_full_marks(self):
        for mode in ("index", "pos"):
            xml = _compare_site._PROBE_XML[mode]
            stats = _compare_site.score_state_row(xml, xml, mode)
            self.assertTrue(stats["exact"])
            self.assertEqual(stats["f1"], 100.0)
            self.assertEqual(stats["ea"], 100.0)

    def test_action_row_scoring_index_mode(self):
        gt = {"action_type": "click", "index": "13"}
        ok = _compare_site.score_action_row(gt, dict(gt), "", "index")
        self.assertTrue(ok["step_correct"] and ok["type_correct"] and ok["parsed"])
        self.assertEqual(ok["field"], "index")

        ng = _compare_site.score_action_row(
            gt, {"action_type": "click", "index": "99"}, "", "index"
        )
        self.assertTrue(ng["type_correct"])
        self.assertFalse(ng["step_correct"])

        unparsed = _compare_site.score_action_row(gt, None, "", "index")
        self.assertFalse(unparsed["parsed"])

    def test_action_row_scoring_xy_mode_uses_bbox(self):
        ui = (
            '<div bounds="[0,0][840,1876]" point="[420,938]">'
            '<button bounds="[100,200][300,280]" point="[200,240]">OK</button></div>'
        )
        gt = {"action": "click", "coordinate": [200, 240]}
        inside = _compare_site.score_action_row(
            gt, {"action": "click", "coordinate": [110, 210]}, ui, "xy"
        )
        outside = _compare_site.score_action_row(
            gt, {"action": "click", "coordinate": [700, 1500]}, ui, "xy"
        )
        self.assertTrue(inside["step_correct"])
        self.assertFalse(outside["step_correct"])
        self.assertEqual(inside["field"], "bbox")


class TestNamingAndOrdering(unittest.TestCase):
    def test_site_dirname(self):
        self.assertEqual(
            _compare_site.site_dirname("AC_EXP02", 1, "state"),
            "on_ac_exp02_stage1_state_compare",
        )
        # stage2 는 kind 가 action 이고 EXP07 은 버전이 이름에 남는다 (v1/v2 분리 필수).
        self.assertEqual(
            _compare_site.site_dirname("AC_EXP07_v2", 2, "action"),
            "on_ac_exp07_v2_stage2_action_compare",
        )

    def test_split_label(self):
        self.assertEqual(_compare_site.split_label("on-AC-state-id"), "ID")
        self.assertEqual(_compare_site.split_label("on-AC-state-ood"), "OOD")
        self.assertEqual(
            _compare_site.split_label("on-AC-state-id-without-open_app"), "ID · woa"
        )
        self.assertEqual(_compare_site.split_label("on-MB"), "MB")

    def test_setting_sort_key_orders_base_then_epochs(self):
        paths = [
            "lora_world-model/epoch-1",
            "base",
            "lora_world-model/epoch-0.25",
            "lora_world-model/epoch-0.5",
        ]
        ordered = sorted(paths, key=lambda v: _compare_site.setting_sort_key("m", v))
        self.assertEqual(
            ordered,
            [
                "base",
                "lora_world-model/epoch-0.25",
                "lora_world-model/epoch-0.5",
                "lora_world-model/epoch-1",
            ],
        )


class TestRegistryCoverage(unittest.TestCase):
    """사이트는 EXP 단위라, 레지스트리에 EXP 가 빠지면 그 EXP 는 통째로 안 나온다."""

    def test_all_on_disk_exps_are_registered(self):
        eval_viewer = importlib.import_module("eval_viewer")
        for exp in ("AC_EXP05", "AC_EXP06", "AC_EXP07_v1", "AC_EXP07_v2"):
            self.assertIn(exp, eval_viewer.DS_DATADIR)
        # stage1 은 EXP06(stage2 전용) 제외, stage2 는 EXP04(데이터 없음) 제외.
        for exp in ("AC_EXP05", "AC_EXP07_v1", "AC_EXP07_v2"):
            self.assertIn(exp, eval_viewer.EVAL_DATASETS[1])
        for exp in ("AC_EXP05", "AC_EXP06", "AC_EXP07_v1", "AC_EXP07_v2"):
            self.assertIn(exp, eval_viewer.EVAL_DATASETS[2])

    def test_exp07_versions_have_distinct_eval_dirs(self):
        """v1/v2 는 data 는 공유하고 eval 디렉토리는 갈린다 — 섞이면 안 된다."""
        eval_viewer = importlib.import_module("eval_viewer")
        v1 = eval_viewer.EVAL_DATASETS[1]["AC_EXP07_v1"]["on-AC-state-id"]
        v2 = eval_viewer.EVAL_DATASETS[1]["AC_EXP07_v2"]["on-AC-state-id"]
        self.assertEqual(v1["dir"], "on-AC_EXP07_v1-state")
        self.assertEqual(v2["dir"], "on-AC_EXP07_v2-state")
        self.assertEqual(v1["test"], v2["test"])

    def test_task_of_classifies_by_metric_keys(self):
        eval_viewer = importlib.import_module("eval_viewer")
        entries = eval_viewer.EVAL_DATASETS[1]["AC_EXP05"]
        self.assertEqual(eval_viewer.task_of(entries["on-AC-state-id"]), "state")
        self.assertEqual(eval_viewer.task_of(entries["on-AC-action-id"]), "action")


if __name__ == "__main__":
    unittest.main()
