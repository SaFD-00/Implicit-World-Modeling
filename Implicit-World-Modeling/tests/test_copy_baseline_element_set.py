"""scripts/_copy_baseline_eval.py — element 집합 스위치(`--element-set`)와 스탬프.

형제 채점기 둘(`_hungarian_eval` · `_state_diff_eval`)은 2026-08-21 부터 산출물에
`element_set` 을 박는데 이 채점기만 빠져 있었다. 셋 중 하나만 기준 불명이면 오히려
더 나쁘다 — 나란히 놓고 "어느 파일이 옛 기준인가"를 판정할 수 없게 된다.

여기서 찌르는 것은 셋이다.

  1. **스탬프의 출처.** 인자가 아니라 채점기가 실제로 읽은 전역(`_he.ELEMENT_SET`)이어야
     한다. 전역은 모듈 사본마다 따로 있어 전파가 실제로 끊길 수 있는데, 호출자가 믿는
     값을 적으면 하필 그 경우에 산출물이 거짓을 말한다.
  2. **플래그가 스탬프뿐 아니라 채점에 닿는가.** 스탬프는 맞는데 채점은 딴 집합인
     경우를 잡으려면 실제로 센 요소 수(`avg_n_cur`)를 함께 봐야 한다.
  3. **복사기 불변식이 `full` 에서도 성립하는가.** `assert_copy_baseline_invariants`
     의 닫힌 형태 등식은 element 집합과 무관하게 성립해야 한다 — 복사기는
     `pred_els == cur_els` 라 pred↔gt 매칭과 cur→gt 분류가 **같은 매칭**이기 때문이다.
     집합을 넓혔더니 깨진다면 그건 값 문제가 아니라 그 논증이 틀렸다는 뜻이다.
"""

from __future__ import annotations

import json
import subprocess
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


# ── 픽스처 ───────────────────────────────────────────────────────────────
# 루트 `node` 가 **legacy 화이트리스트 밖**이라 두 집합의 요소 수가 갈린다:
# legacy 3(button/p/img) vs full 4(+node). 이 차이가 "플래그가 채점에 닿았나"의 관측점이다.
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
MODES = ("index", "pos")
# 요소 수는 두 집합의 정의에서 바로 따라 나온다 — 코드를 돌려 얻은 수가 아니다.
N_CUR = {"legacy": 3.0, "full": 4.0}


def _prompt(cur: str, mode: str) -> str:
    """current state 를 담은 렌더 프롬프트. 계열마다 마커가 다르다."""
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


def _write_pair(d: Path, mode: str) -> tuple[Path, Path]:
    """(test jsonl, prediction jsonl). 완벽 예측 · 복사 · 무변화 3행."""
    rows = [
        (CUR[mode], GT[mode], GT[mode]),  # 완벽 예측
        (CUR[mode], GT[mode], CUR[mode]),  # 모델도 복사
        (CUR[mode], CUR[mode], CUR[mode]),  # 변화 없는 행
    ]
    t, p = d / "test.jsonl", d / "generated_predictions.jsonl"
    t.write_text(
        "".join(
            json.dumps(
                {"messages": [{"value": "sys"}, {"value": "u"}, {"value": gt}]},
                ensure_ascii=False,
            )
            + "\n"
            for _, gt, _ in rows
        ),
        encoding="utf-8",
    )
    p.write_text(
        "".join(
            json.dumps({"prompt": _prompt(cur, mode), "predict": pr}, ensure_ascii=False)
            + "\n"
            for cur, _, pr in rows
        ),
        encoding="utf-8",
    )
    return t, p


class TestStampSource(unittest.TestCase):
    """스탬프는 **채점기가 읽은 전역**에서 나와야 한다 (인자가 아니라)."""

    def setUp(self):
        self._saved = _he.ELEMENT_SET
        self.addCleanup(_he.set_element_set, self._saved)

    def test_stamp_follows_the_global_the_scorer_read(self):
        for name in ("legacy", "full"):
            with self.subTest(element_set=name):
                _he.set_element_set(name)
                m = _cb.stamp({}, "pos", None)
                self.assertEqual(m["element_set"], name)

    def test_stamp_keeps_the_other_four_keys(self):
        _he.set_element_set("full")
        m = _cb.stamp({}, "index", "잘림 사유")
        self.assertEqual(
            set(m),
            {
                "copy_baseline_schema",
                "metrics_schema",
                "match_mode",
                "element_set",
                "truncated",
            },
        )
        self.assertEqual(m["metrics_schema"], _sd.METRICS_SCHEMA)
        self.assertEqual(m["truncated"], "잘림 사유")


class TestCliFlag(unittest.TestCase):
    """CLI 를 subprocess 로 돈다. 전역이라 in-process 로 돌리면 **다른 테스트로 샌다.**"""

    def _run(self, tmp: Path, mode: str, flags: list[str]) -> dict:
        t, p = _write_pair(tmp, mode)
        out = tmp / "copy_baseline_metrics.json"
        r = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "_copy_baseline_eval.py"),
                "score",
                "--test",
                str(t),
                "--pred",
                str(p),
                "--match-mode",
                mode,
                "--output",
                str(out),
                *flags,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(out.read_text(encoding="utf-8"))

    def test_default_is_full_and_reaches_the_scoring(self):
        """스탬프만 보면 안 된다 — 스탬프는 맞고 채점은 딴 집합일 수 있다."""
        for mode in MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                m = self._run(Path(d), mode, [])
                self.assertEqual(m["element_set"], "full")
                self.assertEqual(
                    m["overall"]["copy_baseline"]["avg_n_cur"],
                    N_CUR["full"],
                    "full 은 루트 node 까지 센다",
                )

    def test_legacy_flag_reaches_the_scoring(self):
        for mode in MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                m = self._run(Path(d), mode, ["--element-set", "legacy"])
                self.assertEqual(m["element_set"], "legacy")
                self.assertEqual(
                    m["overall"]["copy_baseline"]["avg_n_cur"],
                    N_CUR["legacy"],
                    "화이트리스트가 루트 node 를 버린다",
                )

    def test_unknown_value_is_rejected_by_the_parser(self):
        """오타가 조용히 full 로 떨어지면 안 된다 — 산출물이 거짓을 말하게 된다."""
        with tempfile.TemporaryDirectory() as d:
            t, p = _write_pair(Path(d), "pos")
            r = subprocess.run(
                [
                    sys.executable,
                    str(_SCRIPTS / "_copy_baseline_eval.py"),
                    "score",
                    "--test",
                    str(t),
                    "--pred",
                    str(p),
                    "--output",
                    str(Path(d) / "out.json"),
                    "--element-set",
                    "whitelist",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 2, r.stdout)
            self.assertIn("--element-set", r.stderr)


class TestInvariantsSurviveFull(unittest.TestCase):
    """복사기 불변식은 element 집합과 무관하게 성립해야 한다.

    근거: 복사기는 `pred_els == cur_els` 이므로 hit 을 만드는 매칭(pred↔gt)과 GT 를
    diff 유형으로 나누는 매칭(cur→gt)이 **같은 호출**이다. 이 논증에는 어떤 요소가
    집합에 드는지가 등장하지 않는다 — 그래서 `full` 에서도 그대로 성립해야 하고,
    깨진다면 값 문제가 아니라 논증이 틀렸다는 뜻이라 반드시 드러나야 한다.
    (`build_section` 이 `assert_copy_baseline_invariants` 를 부르므로 위반 시 raise 다.)
    """

    def setUp(self):
        self._saved = _he.ELEMENT_SET
        self.addCleanup(_he.set_element_set, self._saved)

    def _section(self, mode: str, element_set: str) -> dict:
        _he.set_element_set(element_set)
        with tempfile.TemporaryDirectory() as d:
            t, p = _write_pair(Path(d), mode)
            return _cb.evaluate_pairs(
                _he._load_jsonl(str(t)),
                _he._load_jsonl(str(p)),
                mode,
                label=f"{mode}/{element_set}",
            )

    def test_closed_form_invariants_in_both_sets(self):
        for element_set in ("legacy", "full"):
            for mode in MODES:
                with self.subTest(element_set=element_set, mode=mode):
                    cb = self._section(mode, element_set)["copy_baseline"]
                    self.assertEqual(cb["copy_exact_rate"], 1.0)
                    self.assertEqual(cb["avg_copy_rate_pred"], 1.0)
                    self.assertEqual(cb["avg_change_f1_strict"], 0.0)
                    self.assertEqual(cb["avg_modified_recall"], 1.0)
                    self.assertEqual(cb["avg_unchanged_recall"], 1.0)
                    self.assertEqual(cb["avg_added_recall"], 0.0)
                    self.assertEqual(cb["avg_n_cur"], N_CUR[element_set])

    def test_full_scores_more_elements_than_legacy(self):
        """두 집합이 실제로 다른 채점을 한다는 것 자체를 못박는다 — 위 불변식이
        '집합이 안 바뀌어서' 통과하는 퇴화를 배제한다."""
        for mode in MODES:
            with self.subTest(mode=mode):
                legacy = self._section(mode, "legacy")["copy_baseline"]
                full = self._section(mode, "full")["copy_baseline"]
                self.assertGreater(full["avg_n_cur"], legacy["avg_n_cur"])
                self.assertGreater(full["avg_n_gt"], legacy["avg_n_gt"])


if __name__ == "__main__":
    unittest.main()
