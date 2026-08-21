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
_state_diff_eval = importlib.import_module("_state_diff_eval")
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
            # 3번째 인자는 current state — pred == gt == current 는 "변화가 없는
            # 행"이라 state-diff 축이 전부 미정의다 (hungarian 축만 만점).
            stats = _compare_site.score_state_row(xml, xml, xml, mode)
            self.assertTrue(stats["exact"])
            self.assertEqual(stats["f1"], 100.0)
            self.assertEqual(stats["ea"], 100.0)

    def test_state_row_carries_state_diff_axes(self):
        """카드가 Hungarian F1 하나로 판단되면 안 된다 (AGENTS.md 13b).

        `_state_diff_eval._PROBE` 의 `cur`/`gt`/`maxdel` 은 정본이 change 축을
        설계할 때 쓴 fixture 다. `maxdel`(current 를 하나도 재현하지 않는 예측)이
        **바닥이 0 이 아님**을 보인다 — 그 행에서 strict 는 바닥에 진다.
        """
        for mode in ("index", "pos"):
            probe = _state_diff_eval._PROBE[mode]
            copy = _compare_site.score_state_row(
                probe["cur"], probe["gt"], probe["cur"], mode
            )
            # 복사기는 변화를 하나도 주장하지 않는다 → change 축 0.0 (None 이 아니다)
            self.assertEqual(copy["n_change_pred"], 0)
            self.assertEqual(copy["change_f1_strict"], 0.0)
            self.assertLess(copy["addmod_recall"], 1.0)

            maxdel = _compare_site.score_state_row(
                probe["maxdel"], probe["gt"], probe["cur"], mode
            )
            self.assertGreater(maxdel["change_f1_floor"], 0.0, "바닥은 0 이 아니다")
            self.assertLess(
                maxdel["change_f1_strict"],
                maxdel["change_f1_floor"],
                "퇴화 예측은 바닥에 진다 — 0 기준으로 읽으면 오독한다",
            )

    def test_change_audit_reproduces_canonical_numbers(self):
        """요소 단위 감사(화면의 색·리스트)가 정본 수치를 재현해야 한다.

        재현하지 못하면 화면은 멀쩡한데 감사가 거짓이 된다 — 그래서 빌드가 이걸
        행마다 확인하고, 여기서는 그 대조기 자체가 살아 있는지 본다.
        """
        for mode in ("index", "pos"):
            probe = _state_diff_eval._PROBE[mode]
            for pred in (probe["gt"], probe["cur"], probe["maxdel"], ""):
                diff = _state_diff_eval.compute_state_diff(
                    pred, probe["gt"], probe["cur"], mode
                )
                _, derived = _compare_site.state_change_audit(
                    pred, probe["gt"], probe["cur"], mode
                )
                self.assertIsNone(
                    _compare_site._audit_consistency_error(derived, diff),
                    f"{mode}/{pred[:20]!r}",
                )

    def test_change_audit_marks_copied_element(self):
        """복사 편향의 직접 증거 — '바뀌어야 할 자리를 그대로 베꼈다'가 보여야 한다."""
        probe = _state_diff_eval._PROBE["index"]
        audit, _ = _compare_site.state_change_audit(
            probe["cur"], probe["gt"], probe["cur"], "index"
        )
        self.assertEqual(audit["pd"], [], "복사기는 아무 변화도 주장하지 않는다")
        copied = [it for it in audit["gt"] if it.get("w") == "copy"]
        self.assertTrue(copied, f"copy 표식이 없다 — {audit['gt']}")
        # 매칭된 예측 요소와 그 text_sim 이 함께 보여야 "왜 miss 인지"가 읽힌다.
        self.assertIn("m", copied[0])
        self.assertLess(copied[0]["s"], _state_diff_eval.CHANGE_TEXT_SIM_TAU)

    def test_gt_change_marks_split_key_spaces(self):
        """DELETED 는 current 요소, ADDED/MODIFIED 는 GT 요소 — 키 공간이 다르다.

        섞어도 화면은 칠해져서 정상으로 보이므로 여기서 붙잡는다.
        """
        probe = _state_diff_eval._PROBE["index"]
        marks = _compare_site.gt_change_marks(probe["cur"], probe["gt"], "index")
        self.assertEqual(set(marks["cur"].values()), {"mk-del"})
        self.assertTrue(set(marks["gt"].values()) <= {"mk-add", "mk-mod"})
        self.assertTrue(marks["gt"], "GT 쪽 변화 마크가 비었다")
        # `img #4` 는 gt 에서 사라진다 = current 쪽 DELETED 마크의 근거.
        self.assertIn("4", marks["cur"])

    def test_marks_are_dropped_when_key_is_degenerate(self):
        """EXP03/04 의 XML 은 bounds 만 있고 index 속성이 없는데 index 모드로 채점된다.

        그러면 전 요소의 `index` 가 -1 로 같아져, 그 값을 키로 쓰면 **화면 전체가
        칠해진다**. 안 칠하는 것(무해)과 다 칠하는 것(그럴듯하게 틀림)은 같지 않다.
        """
        self.assertEqual(
            _compare_site._el_key({"tag": "div", "text": "", "index": -1}, "index"), ""
        )
        self.assertEqual(
            _compare_site._el_key({"tag": "div", "text": "", "bounds": ""}, "pos"), ""
        )
        cur = '<node><div bounds="[0,0][10,10]" point="[5,5]">a</div></node>'
        gt = '<node><div bounds="[0,0][10,10]" point="[5,5]">b</div></node>'
        marks = _compare_site.gt_change_marks(cur, gt, "index")
        self.assertEqual(marks, {"cur": {}, "gt": {}}, "판별 불가 키가 실렸다")

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
