"""scripts/_hungarian_eval.py — 채점 대상 element 집합(`full` / `legacy`) 테스트.

옛 화이트리스트(`INTERACTIVE_TAGS`/`CONTENT_TAGS`/`CLICKABLE_ATTRS`)는 실제 요소의
**약 24% 를 조용히 버렸다** — EXP05 test 50문서에서 2,795개 중 687개, `div` 617개와
`img` 11개는 전량이다. 버려진 요소에서 일어난 화면 변화는 채점기가 아예 관측하지
못했다. 2026-08-21 부터 기본은 `full`(파서가 낸 모든 요소)이고, `legacy` 는 옛
산출물과 나란히 놓기 위한 재현 모드다.

여기서 고정하는 것
  1. full 이 구조 요소·텍스트 없는 요소를 실제로 채택한다.
  2. 태그 케이스가 파서 경로(xml / html.parser 폴백)와 무관하게 같다.
  3. 텍스트 규칙 "자체 우선, 비면 자손 흡수"가 index/pos 두 모드에서 같다.
  4. legacy 가 옛 집합과 옛 수치를 그대로 재현한다.
  5. 산출 JSON 이 **자기가 어떤 기준으로 채점됐는지** 스탬프한다 — 그리고 그 스탬프가
     정본/state-diff 두 채점기에서 **같은 값**이다 (전역 전파 검사).
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
_cs = __import__("_compare_site")  # 배선 가드 음성 테스트용 — 읽기만 한다


def setUpModule():
    """bs4/솔버는 지연 로드다. 초기화 없이 `extract_elements` 를 부르면 예외가 나고,
    정본 경로에서는 `compute_hungarian_acc` 의 except 가 그걸 삼켜 **전 행 0점**이 된다."""
    _he._lazy_deps()


@contextlib.contextmanager
def element_set(name: str):
    """전역이라 반드시 되돌린다 — 안 되돌리면 **다른 테스트 파일**이 옛/새 집합을
    섞어 채점하게 되고, 그 실패는 이 파일과 무관한 곳에서 튄다."""
    old = _he.ELEMENT_SET
    _he.set_element_set(name)
    try:
        yield
    finally:
        _he.set_element_set(old)


# 실제 EXP05 XML 에서 옛 화이트리스트가 통째로 버리던 형태들.
STRUCTURAL = (
    '<node bounds="[0,0][10,10]">'
    '<RecyclerView bounds="[0,0][10,5]">'
    '<div bounds="[1,1][4,4]" aria-label="Plan"/>'
    '<img bounds="[5,1][9,4]"/>'
    "</RecyclerView>"
    '<LinearLayoutCompat bounds="[0,5][10,10]"/>'
    "</node>"
)


class TestFullAdoptsEverything(unittest.TestCase):
    def test_structural_and_textless_tags_are_adopted(self):
        for mode in ("index", "pos"):
            with self.subTest(mode=mode), element_set("full"):
                tags = [e["tag"] for e in _he.extract_elements(STRUCTURAL, mode)]
                self.assertEqual(
                    tags,
                    ["node", "recyclerview", "div", "img", "linearlayoutcompat"],
                    "파서가 낸 모든 요소가 채점 대상이어야 한다",
                )

    def test_textless_img_survives(self):
        """옛 조건은 `is_content = (tag in CONTENT_TAGS) and bool(text)` 였다 —
        img 는 텍스트가 없어 **항상** False 였고, 그래서 전량 탈락했다."""
        with element_set("full"):
            els = _he.extract_elements('<img bounds="[0,0][1,1]"/>', "pos")
        self.assertEqual([(e["tag"], e["text"]) for e in els], [("img", "")])

    def test_tag_case_is_parser_independent(self):
        """lxml "xml" 은 케이스를 보존하고 html.parser 폴백은 소문자화한다. 예측만
        마크다운 펜스에 감싸이면 pred 는 폴백, GT 는 xml 경로를 타 태그가 영구
        불일치하는데, 태그 불일치 cost(W_TAG=3.0)는 임계(1.5/1.7)를 넘는 하드 게이트라
        그 문서는 통째로 0점이 된다."""
        fenced = f"```html\n{STRUCTURAL}\n```"
        with element_set("full"):
            plain_tags = [e["tag"] for e in _he.extract_elements(STRUCTURAL, "pos")]
            fenced_tags = [e["tag"] for e in _he.extract_elements(fenced, "pos")]
        self.assertEqual(plain_tags, fenced_tags)
        self.assertIn("recyclerview", fenced_tags)

    def test_fence_fallback_still_finds_elements(self):
        """펜스/프리앰블 폴백은 화이트리스트 제거와 무관하게 살아 있어야 한다."""
        with element_set("full"):
            for wrapped in (
                f"```html\n{STRUCTURAL}\n```",
                f"## Predicted Next State\n{STRUCTURAL}",
            ):
                with self.subTest(wrapped=wrapped[:12]):
                    self.assertEqual(len(_he.extract_elements(wrapped, "pos")), 5)


class TestFullTextRule(unittest.TestCase):
    """"자체 우선, 비면 자손 흡수" — index/pos 공통 규칙."""

    def _text(self, xml, mode="pos", tag=None):
        with element_set("full"):
            els = _he.extract_elements(xml, mode)
        return next(e["text"] for e in els if tag is None or e["tag"] == tag)

    def test_self_text_wins(self):
        """기존에 포함되던 요소에는 사실상 무변화 — 전후 모두 "Saves"."""
        xml = '<button aria-label="Saves"><p>Saves</p></button>'
        self.assertEqual(self._text(xml, tag="button"), "Saves")

    def test_container_absorbs_descendants_when_self_is_empty(self):
        self.assertEqual(self._text("<div><p>X</p></div>", tag="div"), "X")

    def test_self_attribute_stops_absorption(self):
        """자체가 있는데 자손까지 삼키면 상위 컨테이너 토큰이 형제 구분을 지운다."""
        self.assertEqual(
            self._text('<div aria-label="Nav"><p>X</p></div>', tag="div"), "Nav"
        )

    def test_descendant_aria_label_is_absorbed(self):
        """EXP05 계열은 description 이 0건이고 aria-label 만 쓴다. 자손에서 그것을
        안 걷으면 nav 컨테이너가 텍스트 0 이 되어 아래 퇴화를 그대로 맞는다."""
        self.assertEqual(
            self._text('<div><button aria-label="OK"/></div>', tag="div"), "OK"
        )

    def test_absorbed_text_is_token_separated(self):
        """separator 없이 `get_text` 를 부르면 "ab" 라는 없는 토큰이 만들어진다."""
        self.assertEqual(self._text("<div><p>a</p><p>b</p></div>", tag="div"), "a b")

    def test_index_and_pos_share_the_rule(self):
        """옛 집합에는 pos 모드만 자손 흡수를 안 하는 무주석 divergence 가 있었다."""
        xml = '<div><button aria-label="OK"/><p>hello</p></div>'
        self.assertEqual(self._text(xml, "index", "div"), self._text(xml, "pos", "div"))

    def test_absorption_keeps_containers_distinguishable(self):
        """흡수가 없으면 컨테이너 텍스트가 전부 "" 이 되고, `_text_sim("","")==1.0`
        이라 **태그만 같으면 아무 컨테이너끼리나 cost 0 으로 붙어** 완벽한 UNCHANGED 로
        채점된다. 흡수는 그 퇴화를 막으려고 있다."""
        a = self._text("<div><p>alpha</p></div>", tag="div")
        b = self._text("<div><p>omega</p></div>", tag="div")
        self.assertLess(_he._text_sim(a, b), 1.0)


class TestLegacyReproducesTheOldSet(unittest.TestCase):
    def test_structural_tags_are_dropped(self):
        for mode in ("index", "pos"):
            with self.subTest(mode=mode), element_set("legacy"):
                tags = [e["tag"] for e in _he.extract_elements(STRUCTURAL, mode)]
                self.assertEqual(tags, [], "화이트리스트 밖 요소는 전량 탈락한다")

    def test_legacy_keeps_the_old_text_collection(self):
        """pos 는 자손 흡수 없음, index 는 흡수 있음 — 옛 divergence 그대로."""
        xml = '<button aria-label="Saves"><p>deep</p></button>'
        with element_set("legacy"):
            pos_text = _he.extract_elements(xml, "pos")[0]["text"]
            idx_text = _he.extract_elements(xml, "index")[0]["text"]
        self.assertEqual(pos_text, "Saves", "pos: 자체 속성만")
        self.assertEqual(idx_text, "deep", "index: 자손 흡수, aria-label 은 안 봄")

    def test_include_aria_still_opens_the_pos_set(self):
        """`include_aria` 는 legacy 에서만 의미가 있다 (full 은 모두 채택한다)."""
        xml = '<div bounds="[0,0][1,1]" aria-label="Home"/>'
        with element_set("legacy"):
            self.assertEqual(_he.extract_elements(xml, "pos"), [])
            self.assertEqual(len(_he.extract_elements(xml, "pos", True)), 1)
        with element_set("full"):
            self.assertEqual(len(_he.extract_elements(xml, "pos")), 1)

    def test_old_three_way_split_counts(self):
        """`tests/test_state_diff_eval.py` 의 픽스처가 옛 집합에서 내던 1/1/1.
        지금 그 파일은 root `<node>` 가 요소로 들어온 full 기준 값을 고정한다."""
        cur = (
            '<node index="0"><button index="1" aria-label="OK"/>'
            '<p index="2">unread inbox 3</p></node>'
        )
        gt = (
            '<node index="0"><button index="1" aria-label="OK"/>'
            '<p index="2">unread inbox 7</p>'
            '<p index="3">brand new banner</p></node>'
        )
        with element_set("legacy"):
            counts = _sd.summarize_diff(_sd.classify_diff(cur, gt, "index"))
        self.assertEqual(
            (counts["UNCHANGED"], counts["MODIFIED"], counts["ADDED"]), (1, 1, 1)
        )

    def test_old_hand_checked_floor(self):
        """옛 손 검산값 — cur 는 button/p 2요소(root `node` 는 안 뽑힌다),
        바닥 hits=1/n_pred=2/n_gt=2 → 0.5, 최대삭제 예측은 n_pred=3 → 0.4.
        full 기준 값(3 / 0.5714 / 5 / 0.4444)은 `test_state_diff_eval.py` 에 있다."""
        cur = '<node index="0"><button index="1"/><p index="2">old row</p></node>'
        gt = (
            '<node index="0"><button index="1"/>'
            '<span index="3">fresh banner</span></node>'
        )
        maxdel = '<node index="900"><select index="901"/></node>'
        with element_set("legacy"):
            empty = _sd.compute_state_diff("", gt, cur, "index")
            degen = _sd.compute_state_diff(maxdel, gt, cur, "index")
        self.assertEqual(empty["n_cur"], 2)
        self.assertEqual(empty["n_change_gt"], 2)
        self.assertEqual(empty["change_f1_floor"], 0.5)
        self.assertEqual(degen["n_change_pred"], 3)
        self.assertEqual(degen["change_f1_strict"], 0.4)


class TestSwitch(unittest.TestCase):
    def test_default_is_full(self):
        self.assertEqual(_he.ELEMENT_SET, "full")

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            _he.set_element_set("whitelist")
        self.assertEqual(_he.ELEMENT_SET, "full", "실패해도 상태는 안 바뀐다")


# ── CLI 스탬프 & 전역 전파 ──────────────────────────────────────────────
CUR = (
    '<node index="0"><button index="1" aria-label="OK"/>'
    '<p index="2">unread inbox 3</p></node>'
)
GT = (
    '<node index="0"><button index="1" aria-label="OK"/>'
    '<p index="2">unread inbox 7</p>'
    '<p index="3">brand new banner</p></node>'
)
PROMPT = (
    "system\nrole\nuser\n\n## Current State\n" + CUR + "\n\n## Action\n"
    '{"action_type":"click","index":"1"}\nassistant\n'
)


class TestCliStamp(unittest.TestCase):
    """산출물이 자기 채점 기준을 말해야 한다 — 기존 산출물은 재채점하지 않기로 했으므로
    (2026-08-21) 스탬프 없는 파일이 곧 legacy 기준이다.

    그리고 이 테스트는 **전역 전파**를 잡는다. `_hungarian_eval.py` 를 스크립트로
    돌리면 그 모듈은 `__main__` 이고 `_state_diff_eval` 의 `import _hungarian_eval` 은
    두 번째 사본을 만든다 — 한쪽만 설정하면 정본은 새 집합, state-diff 는 옛 집합으로
    채점된다. 그래서 in-process 가 아니라 **서브프로세스**로 돌리고, 스탬프뿐 아니라
    state-diff 가 실제로 센 요소 수(`avg_n_gt`)까지 본다.
    """

    def _run(
        self, tmp: Path, flags: list[str], env: dict | None = None
    ) -> tuple[dict, dict]:
        test_path = tmp / "test.jsonl"
        pred_path = tmp / "pred.jsonl"
        test_path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"value": "sys"},
                        {"value": PROMPT},
                        {"value": GT},
                    ]
                }
            )
            + "\n"
        )
        pred_path.write_text(
            json.dumps({"prompt": PROMPT, "predict": GT, "label": GT}) + "\n"
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

    def test_full_is_the_default_and_both_files_agree(self):
        with tempfile.TemporaryDirectory() as d:
            hung, sd = self._run(Path(d), [])
        self.assertEqual(hung["element_set"], "full")
        self.assertEqual(sd["element_set"], "full")
        self.assertEqual(sd["avg_n_gt"], 4.0, "root node 를 포함해 4요소")

    def test_legacy_flag_reaches_the_state_diff_scorer(self):
        """스탬프만 보면 안 된다 — 스탬프는 맞고 채점은 새 집합일 수 있다.
        그래서 요소 수를 함께 본다 (legacy 는 root node 를 안 뽑아 3요소)."""
        with tempfile.TemporaryDirectory() as d:
            hung, sd = self._run(Path(d), ["--element-set", "legacy"])
        self.assertEqual(hung["element_set"], "legacy")
        self.assertEqual(sd["element_set"], "legacy")
        self.assertEqual(sd["avg_n_gt"], 3.0, "화이트리스트가 root node 를 버린다")

    def test_metrics_schema_stamp_survives(self):
        with tempfile.TemporaryDirectory() as d:
            _, sd = self._run(Path(d), [])
        self.assertEqual(sd["metrics_schema"], _sd.METRICS_SCHEMA)


class TestEnvDefault(unittest.TestCase):
    """셸 파이프라인은 `--element-set` 을 넘기지 않는다 (`stage1_eval.sh` ·
    `rebuild_*.sh` · `probe_forget_eval.sh`). 환경변수 기본값이 없으면 **정상 경로로는
    legacy 를 요청할 방법이 없고**, 그러면 플래그가 `--include-aria` 처럼 죽는다."""

    def test_env_var_sets_the_default(self):
        self.assertEqual(_he._default_element_set(), "full")
        with unittest.mock.patch.dict(os.environ, {"ELEMENT_SET": "legacy"}):
            self.assertEqual(_he._default_element_set(), "legacy")

    def test_empty_env_var_is_treated_as_unset(self):
        """셸에서 `ELEMENT_SET="${ELEMENT_SET:-}"` 로 빈 값을 export 하는 일이 흔하다."""
        with unittest.mock.patch.dict(os.environ, {"ELEMENT_SET": ""}):
            self.assertEqual(_he._default_element_set(), "full")

    def test_env_var_reaches_both_scorers_end_to_end(self):
        stamp = TestCliStamp._run
        with tempfile.TemporaryDirectory() as d:
            hung, sd = stamp(self, Path(d), [], {"ELEMENT_SET": "legacy"})
        self.assertEqual((hung["element_set"], sd["element_set"]), ("legacy", "legacy"))
        self.assertEqual(sd["avg_n_gt"], 3.0, "실제로 옛 집합으로 채점됐다")

    def test_explicit_flag_beats_the_env_var(self):
        stamp = TestCliStamp._run
        with tempfile.TemporaryDirectory() as d:
            hung, sd = stamp(
                self, Path(d), ["--element-set", "full"], {"ELEMENT_SET": "legacy"}
            )
        self.assertEqual((hung["element_set"], sd["element_set"]), ("full", "full"))
        self.assertEqual(sd["avg_n_gt"], 4.0)


class TestWiringGuardStillFails(unittest.TestCase):
    """배선 가드가 **여전히 원래 실패를 잡는지** 음성으로 고정한다.

    가드의 존재 이유는 "bs4/scipy 미초기화 → `compute_hungarian_acc` 의 except 가
    예외를 삼켜 전 행 조용히 0점"이다 (2026-08-01 실측: 표본 f1 0.0 vs aggregate 0.71).
    element 집합이 `full` 이 되며 추출 경로가 바뀌었으니, 가드가 아직 **정상 종료하지
    않는다**는 것을 확인해야 한다. `_compare_site` 는 다른 워커 소유라 읽기만 한다.
    """

    @contextlib.contextmanager
    def _broken(self, **attrs):
        """`_lazy_deps` 를 무력화해 의존성 복구를 막고 지정한 전역을 부순다."""
        saved = {k: getattr(_he, k) for k in (*attrs, "_lazy_deps")}
        _he._lazy_deps = lambda: None
        for k, v in attrs.items():
            setattr(_he, k, v)
        try:
            yield
        finally:
            for k, v in saved.items():
                setattr(_he, k, v)
            _he._lazy_deps()

    def test_empty_extraction_raises_on_both_guards(self):
        with self._broken(extract_elements=lambda *a, **k: []):
            for mode in ("index", "pos"):
                with self.subTest(mode=mode):
                    with self.assertRaises(_cs.SiteBuildError):
                        _cs.assert_state_scorer_wired(mode)
                    with self.assertRaises(_sd.StateDiffError):
                        _sd.assert_scorer_wired(mode)

    def test_missing_bs4_never_passes_the_guard(self):
        """이때 `compute_hungarian_acc` 는 **조용히 0점을 돌려준다** — 가드가 없으면
        그 표가 그대로 저장된다. 가드는 정상 종료하면 안 된다."""
        with self._broken(BeautifulSoup=None):
            self.assertEqual(
                _he.compute_hungarian_acc("<p>a</p>", "<p>a</p>", "pos")[
                    "hungarian_f1"
                ],
                0.0,
                "가드가 막으려는 조용한 실패가 실제로 존재한다",
            )
            # 예외 종류는 의존성이 어디서 터지느냐에 달렸다 (`SiteBuildError` 는
            # SystemExit 계열, bs4 부재는 `_parse_soup` 의 TypeError). 고정할 것은
            # **정상 종료하지 않는다**는 성질이다.
            with self.assertRaises(BaseException):
                _cs.assert_state_scorer_wired("pos")
            with self.assertRaises(BaseException):
                _sd.assert_scorer_wired("pos")

    def test_missing_solver_never_passes_the_guard(self):
        with self._broken(_solve=None):
            with self.assertRaises(BaseException):
                _cs.assert_state_scorer_wired("pos")
            with self.assertRaises(BaseException):
                _sd.assert_scorer_wired("pos")

    def test_guards_pass_again_after_restore(self):
        """위 세 테스트가 전역을 되돌리는지 — 안 되돌리면 다른 파일이 전 행 0점을 받는다."""
        _cs.assert_state_scorer_wired("pos")
        _sd.assert_scorer_wired("pos")


if __name__ == "__main__":
    unittest.main()
