"""scripts/_hungarian_eval.py — XML 스키마 스위치(`android` / `cerebra`) 테스트.

AC_EXP08 의 XML 은 위치축이 `data-bbox="x1 y1 x2 y2"` 이고 `bounds` 속성이 아예 없다.
android 규약만 읽는 채점기에 넣으면 **에러 없이** 좌표가 전부 파싱 실패로 떨어져 위치
cost 가 0 이 된다 — 요소 수도 f1 도 그럴듯하게 나오므로 아무도 못 본다 (하드 제약 8·13b
와 같은 실패 계열). `--xml-schema cerebra` 가 그 축을 연다.

여기서 고정하는 것
  1. **기본은 android** 이고, android 에서는 cerebra 확장이 하나도 켜지지 않는다
     (기존 실험군의 채점 결과 불변 — AGENTS "채점기를 바꿔야 하면 opt-in 플래그로").
  2. cerebra 는 `data-bbox` 좌표·`alt`/`placeholder`/`value` 텍스트를 실제로 읽는다.
  3. `full` 집합의 **채택 조건은 스키마와 무관하다** — `_state_diff_eval` 의 유도성 축이
     `hungarian_metric_v3` 와 같은 길이·순서를 전제하므로 여기서 갈리면 그 축이 조용히
     죽는다. cerebra 의 채택 확장(aria-label·구조 div)은 legacy 집합에만 걸린다.
  4. 산출 JSON 이 `xml_schema` 를 스탬프하고, 그 값이 정본/state-diff 두 파일에서
     **같다** (전역 전파 검사 — `element_set` 과 같은 규약).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_he = __import__("_hungarian_eval")
_sd = __import__("_state_diff_eval")


def setUpModule():
    _he._lazy_deps()


@contextlib.contextmanager
def xml_schema(name: str):
    """전역이라 반드시 되돌린다 — 안 되돌리면 **다른 테스트 파일**이 android 데이터를
    cerebra 규약으로 채점하게 되고, 그 실패는 이 파일과 무관한 곳에서 튄다."""
    old = _he.XML_SCHEMA
    _he.set_xml_schema(name)
    try:
        yield
    finally:
        _he.set_xml_schema(old)


@contextlib.contextmanager
def element_set(name: str):
    old = _he.ELEMENT_SET
    _he.set_element_set(name)
    try:
        yield
    finally:
        _he.set_element_set(old)


# AC_EXP08 실 데이터의 축약형. 위치축은 data-bbox 뿐이고 index/bounds/description 은 없다.
CEREBRA = (
    '<div data-bbox="0 0 840 1876">'
    '<div aria-label="Tabs" data-bbox="0 0 840 200">'
    '<button data-bbox="0 0 400 200">OK</button>'
    '<img alt="logo" data-bbox="400 0 840 200"/>'
    "</div>"
    "</div>"
)
# 같은 트리를 그대로 옮기되 button 만 다른 자리로 옮긴 것 — 위치축이 살아 있어야만
# "멀어졌다"가 보인다.
CEREBRA_MOVED = (
    '<div data-bbox="0 0 840 1876">'
    '<div aria-label="Tabs" data-bbox="0 0 840 200">'
    '<button data-bbox="0 1600 400 1800">OK</button>'
    '<img alt="logo" data-bbox="400 0 840 200"/>'
    "</div>"
    "</div>"
)


class TestAndroidModeIsUnchanged(unittest.TestCase):
    """기본 모드에서는 cerebra 확장이 **하나도** 켜지면 안 된다."""

    def test_default_is_android(self):
        self.assertEqual(_he.XML_SCHEMA, "android")
        self.assertEqual(_he._default_xml_schema(), "android")

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            _he.set_xml_schema("cerebro")

    def test_space_separated_coords_are_not_parsed(self):
        with xml_schema("android"):
            self.assertIsNone(_he._parse_bounds_center("0 0 840 1876"))

    def test_bracket_coords_still_parse(self):
        for name in ("android", "cerebra"):
            with xml_schema(name):
                self.assertEqual(_he._parse_bounds_center("[0,0][10,20]"), (5.0, 10.0))

    def test_position_signal_is_silently_dead_on_cerebra_xml(self):
        """이 0.0 이 이 플래그의 존재 이유다 — f1 은 멀쩡한데 위치축만 죽는다."""
        with xml_schema("android"), element_set("full"):
            m = _he.compute_hungarian_acc(CEREBRA_MOVED, CEREBRA, "pos")
        self.assertEqual(m["hungarian_pos"], 0.0)
        self.assertGreater(m["hungarian_f1"], 0.9, "지표는 그럴듯하게 나온다")

    def test_extra_text_attrs_are_not_read(self):
        with xml_schema("android"), element_set("full"):
            els = _he.extract_elements('<img alt="logo" data-bbox="0 0 1 1"/>', "pos")
        self.assertEqual(els[0]["text"], "")

    def test_element_dict_has_no_extra_key(self):
        """android 산출물의 element dict 모양이 그대로여야 한다."""
        with xml_schema("android"), element_set("full"):
            els = _he.extract_elements(CEREBRA, "pos")
        self.assertEqual(set(els[0]), {"tag", "text", "bounds"})


class TestCerebraReadsTheSchema(unittest.TestCase):
    def test_space_separated_coords_parse(self):
        with xml_schema("cerebra"):
            self.assertEqual(_he._parse_bounds_center("0 0 840 1876"), (420.0, 938.0))

    def test_bounds_wins_over_data_bbox(self):
        """`_pos_key` 규약 — 두 축이 다 있으면 android 축이 이긴다."""
        with xml_schema("cerebra"):
            el = {"bounds": "[0,0][10,10]", "data-bbox": "800 800 900 900"}
            self.assertEqual(_he._parse_bounds_center(_he._pos_key(el)), (5.0, 5.0))

    def test_position_signal_is_alive(self):
        with xml_schema("cerebra"), element_set("full"):
            same = _he.compute_hungarian_acc(CEREBRA, CEREBRA, "pos")
            moved = _he.compute_hungarian_acc(CEREBRA_MOVED, CEREBRA, "pos")
        self.assertEqual(same["hungarian_pos"], 1.0)
        self.assertLess(moved["hungarian_pos"], 1.0, "옮긴 button 이 위치축에 잡힌다")

    def test_alt_placeholder_value_are_text(self):
        with xml_schema("cerebra"), element_set("full"):
            els = _he.extract_elements(
                '<div data-bbox="0 0 9 9">'
                '<img alt="logo" data-bbox="0 0 1 1"/>'
                '<input placeholder="Search" value="abc" data-bbox="2 2 3 3"/>'
                "</div>",
                "pos",
            )
        texts = [e["text"] for e in els]
        self.assertIn("logo", texts)
        self.assertIn("Search | abc", texts)


class TestAdoptionIsElementSetScoped(unittest.TestCase):
    """cerebra 의 채택 확장(aria-label · 구조 div)은 **legacy 집합에만** 건다.

    full 은 이미 `soup.find_all(True)` 로 전부 채택하고, 그 순회가
    `hungarian_metric_v3.iter_nodes` 와 1:1 이라는 계약 위에 `_state_diff_eval` 의
    유도성 축이 서 있다. full 쪽 채택 조건을 스키마로 갈라 놓으면 그 축이 **에러 없이**
    행 단위로 건너뛰어진다.
    """

    def _n(self, es: str, xs: str) -> int:
        with element_set(es), xml_schema(xs):
            return len(_he.extract_elements(CEREBRA, "pos"))

    def test_full_count_is_schema_independent(self):
        self.assertEqual(self._n("full", "android"), self._n("full", "cerebra"))
        self.assertEqual(self._n("full", "android"), 4)

    def test_legacy_android_drops_the_cerebra_only_elements(self):
        """옛 화이트리스트는 button 하나만 남긴다 — div 둘은 속성이 없고(=android 기준),
        img 는 alt 를 안 읽어 텍스트가 비어 `is_content` 가 False 다."""
        self.assertEqual(self._n("legacy", "android"), 1)

    def test_legacy_cerebra_adopts_aria_and_structure(self):
        self.assertEqual(self._n("legacy", "cerebra"), 4)


# ── CLI 스탬프 & 전역 전파 ──────────────────────────────────────────────
PROMPT = (
    "system\nrole\nuser\n\nCurrent UI State:\n"
    + CEREBRA_MOVED
    + '\n\n[Screenshot]\n\nAction:\n<action>{"action": "click", "coordinate": [10, 20]}'
    "</action>\nassistant\n"
)


class TestCliStamp(unittest.TestCase):
    """스탬프는 **각 채점기가 실제로 읽은 전역**이어야 한다. `_hungarian_eval.py` 를
    스크립트로 돌리면 그 모듈은 `__main__` 이고 `_state_diff_eval` 의 import 는 두 번째
    사본을 만든다 — 한쪽만 설정하면 정본은 cerebra, state-diff 는 android 로 채점된다.
    그래서 in-process 가 아니라 **서브프로세스**로 돌리고, 스탬프뿐 아니라 state-diff 가
    실제로 센 요소 수(`avg_n_gt`)까지 본다."""

    def _run(
        self, tmp: Path, flags: list[str], env: dict | None = None
    ) -> tuple[dict, dict]:
        test_path = tmp / "test.jsonl"
        pred_path = tmp / "pred.jsonl"
        test_path.write_text(
            json.dumps(
                {"messages": [{"value": "sys"}, {"value": PROMPT}, {"value": CEREBRA}]}
            )
            + "\n"
        )
        pred_path.write_text(
            json.dumps({"prompt": PROMPT, "predict": CEREBRA, "label": CEREBRA}) + "\n"
        )
        out = tmp / "hungarian_metrics.json"
        r = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "_hungarian_eval.py"),
                "score",
                "--test",
                str(test_path),
                "--pred",
                str(pred_path),
                "--match-mode",
                "pos",
                "--output",
                str(out),
                *flags,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, **(env or {})},
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        return (
            json.loads(out.read_text()),
            json.loads((tmp / "state_diff_metrics.json").read_text()),
        )

    def test_android_is_the_default_and_both_files_agree(self):
        with tempfile.TemporaryDirectory() as d:
            hung, sd = self._run(Path(d), [])
        self.assertEqual(hung["xml_schema"], "android")
        self.assertEqual(sd["xml_schema"], "android")
        self.assertEqual(hung["avg_hungarian_pos"], 0.0, "위치축이 죽어 있다")

    def test_cerebra_flag_reaches_the_state_diff_scorer(self):
        """스탬프만 보면 안 된다 — 스탬프는 맞고 채점은 android 일 수 있다.
        legacy 집합에서 요소 수가 갈리므로 그것을 증거로 쓴다."""
        with tempfile.TemporaryDirectory() as d:
            hung, sd = self._run(
                Path(d), ["--xml-schema", "cerebra", "--element-set", "legacy"]
            )
        self.assertEqual(hung["xml_schema"], "cerebra")
        self.assertEqual(sd["xml_schema"], "cerebra")
        self.assertEqual(
            sd["avg_n_gt"], 4.0, "cerebra 채택 조건이 state-diff 까지 갔다"
        )

    def test_flags_are_orthogonal(self):
        with tempfile.TemporaryDirectory() as d:
            hung, sd = self._run(Path(d), ["--xml-schema", "cerebra"])
        self.assertEqual((hung["element_set"], hung["xml_schema"]), ("full", "cerebra"))
        self.assertEqual((sd["element_set"], sd["xml_schema"]), ("full", "cerebra"))
        self.assertGreater(hung["avg_hungarian_pos"], 0.0)

    def test_env_var_reaches_both_scorers(self):
        with tempfile.TemporaryDirectory() as d:
            hung, sd = self._run(Path(d), [], {"XML_SCHEMA": "cerebra"})
        self.assertEqual((hung["xml_schema"], sd["xml_schema"]), ("cerebra", "cerebra"))

    def test_explicit_flag_beats_the_env_var(self):
        with tempfile.TemporaryDirectory() as d:
            hung, sd = self._run(
                Path(d), ["--xml-schema", "android"], {"XML_SCHEMA": "cerebra"}
            )
        self.assertEqual((hung["xml_schema"], sd["xml_schema"]), ("android", "android"))

    def test_empty_env_var_is_treated_as_unset(self):
        with unittest.mock.patch.dict(os.environ, {"XML_SCHEMA": ""}):
            self.assertEqual(_he._default_xml_schema(), "android")


class TestStateDiffCli(unittest.TestCase):
    """`_state_diff_eval.py` 자체 CLI(백필 진입점)도 같은 플래그를 받아야 한다 —
    정본과 값이 갈리면 층 분해 항등식이 조용히 깨진다."""

    def test_backfill_entry_point_stamps_the_schema(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "test.jsonl").write_text(
                json.dumps(
                    {
                        "messages": [
                            {"value": "sys"},
                            {"value": PROMPT},
                            {"value": CEREBRA},
                        ]
                    }
                )
                + "\n"
            )
            (tmp / "pred.jsonl").write_text(
                json.dumps({"prompt": PROMPT, "predict": CEREBRA}) + "\n"
            )
            out = tmp / "state_diff_metrics.json"
            r = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPTS / "_state_diff_eval.py"),
                    "score",
                    "--test",
                    str(tmp / "test.jsonl"),
                    "--pred",
                    str(tmp / "pred.jsonl"),
                    "--match-mode",
                    "pos",
                    "--xml-schema",
                    "cerebra",
                    "--element-set",
                    "legacy",
                    "--output",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            sd = json.loads(out.read_text())
        self.assertEqual(sd["xml_schema"], "cerebra")
        self.assertEqual(sd["element_set"], "legacy")
        self.assertEqual(sd["metrics_schema"], _sd.METRICS_SCHEMA)
        self.assertEqual(sd["avg_n_gt"], 4.0)


if __name__ == "__main__":
    unittest.main()
