"""Regression tests for scripts/derivability_site.py (유도성 분류 감사 사이트 빌더).

여기서 지키는 것은 **조용히 틀릴 수 있는 부분**들이다:

1. 요소 ↔ 라벨 1:1 정렬. 뷰어가 XML 을 다시 파싱해 요소 집합이 분류기와 갈리면
   색이 엉뚱한 요소에 칠해지고, 그 화면을 보고 내린 판단이 전부 틀린다. 이건 화면이
   멀쩡해 보이기 때문에 눈으로 못 잡는다.
2. `derivable` 2분류가 분류기(`is_derivable`)에서 왔는가. JS 가 라벨 문자열로 직접
   계산하면 라벨이 추가·이동될 때 화면과 지표가 갈린다.
3. payload 의 `</` 이스케이프. 빠지면 `</div>` 하나가 <script> 를 조기 종료시켜
   **빈 페이지**가 나오는데, 파일은 정상 크기로 남는다.
4. 프롬프트 두 계열(EXP01~04 `## Action` / EXP05~07 `Action:` + `<action>`) 모두에서
   action 이 실제로 읽히는가. 한 계열만 보는 파서가 다른 계열에서 조용히 빈 값을
   돌려주던 것이 woa 필터 사고(2026-07-30)의 실패 모드였다.
5. 필수 섹션 누락은 **크게 터진다**. 빈 화면을 정상 산출물처럼 내면 안 된다.
6. `action_target_undecidable_no_bounds` 는 라벨이 NON_DERIVABLE 이지만 뜻이
   **"판정 못 함"** 이다. 진짜 유도 불가능과 같은 색/필터로 묶이면 오분류 사냥의
   모집단이 오염된다. 분류기가 규칙 문자열을 바꾸면 이 구분이 조용히 사라지므로,
   실동작으로 트리거해서 고정한다 (상수 비교가 아니라 실제 판정 결과로).

Run:
    pytest tests/test_derivability_site.py -v
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "diff_loss"))

ds = importlib.import_module("derivability_site")
H = importlib.import_module("hungarian_diff_v3")

# ── 두 계열의 실물 축약본 ────────────────────────────────────────────────
# B 계열 (EXP05~07): bounds/point + `Action:` + <action> JSON
CUR_POS = """<div bounds="[0,0][840,1800]" point="[420,900]">
  <p bounds="[10,10][300,60]" point="[155,35]">Trips</p>
  <button bounds="[10,100][300,160]" point="[155,130]">Create a Trip</button>
</div>"""
GT_POS = """<div bounds="[0,0][840,1800]" point="[420,900]">
  <p bounds="[10,10][300,60]" point="[155,35]">Trips</p>
  <p bounds="[10,200][300,260]" point="[155,230]">Bengaluru to Paris</p>
  <input bounds="[10,300][300,360]" point="[155,330]" value="hello world"/>
</div>"""
ROW_POS = {
    "messages": [
        {"from": "system", "value": "# Role\nYou are a mobile GUI agent."},
        {
            "from": "human",
            "value": f"Current UI State:\n{CUR_POS}\n\n[Screenshot]\n<image>\n\n"
            'Action:\n<action>{"action": "type", "text": "hello world"}</action>',
        },
        {"from": "gpt", "value": GT_POS},
    ]
}

# A 계열 (EXP01~04): index 속성만, bounds 없음. `## Action` + 맨 JSON
CUR_IDX = """<div index="0">
  <p index="1">Trips</p>
  <button index="2">Create a Trip</button>
</div>"""
GT_IDX = """<div index="0">
  <p index="1">Trips</p>
  <p index="2">Bengaluru to Paris</p>
</div>"""
ROW_IDX = {
    "messages": [
        {"from": "system", "value": "Given the current UI ..."},
        {
            "from": "human",
            "value": f"<image>\n## Current State\n{CUR_IDX}\n\n"
            '## Action\n{"action_type":"input_text","index":"2","text":"hello world"}',
        },
        {"from": "gpt", "value": GT_IDX},
    ]
}


def _load_real_rows(exp: str, n: int) -> list[tuple[int, dict]]:
    """실데이터 앞 n 행. 없으면 빈 리스트 (데이터 없는 환경에서 스킵하기 위해)."""
    p = REPO / "data" / f"AndroidControl_{exp}" / "stage1_test_id_state.jsonl"
    if not p.is_file():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                out.append((i, json.loads(line)))
    return out


def _write(rows: list[dict]) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "test.jsonl"
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return tmp


class ElementAlignment(unittest.TestCase):
    """요소 집합·라벨·2분류가 전부 분류기에서 오는가 (설계 원칙 1~3)."""

    def test_elements_match_classifier_one_to_one(self) -> None:
        for row in (ROW_POS, ROW_IDX):
            s = ds.build_sample(0, row)
            human = row["messages"][1]["value"]
            gt = row["messages"][2]["value"]
            sections = ds.parse_prompt(human)
            deriv = H.classify_derivability(
                sections["current_state"], gt, sections["action"]
            )
            self.assertEqual(len(s["gt"]["elements"]), len(deriv))
            for el, d in zip(s["gt"]["elements"], deriv, strict=True):
                self.assertEqual(el["seq"], d["future_seq_idx"])
                self.assertEqual(el["label"], d["derivability"])
                self.assertEqual(el["own"], d["element"]["own_text"])

    def test_derivable_flag_comes_from_classifier(self) -> None:
        """2분류를 JS 가 라벨 문자열로 재계산하지 않도록, 값이 실려 있어야 한다."""
        s = ds.build_sample(0, ROW_POS)
        for el in s["gt"]["elements"]:
            self.assertIn("derivable", el)
            self.assertEqual(el["derivable"], H.is_derivable(el["label"]))

    def test_counts_agree_with_summarize(self) -> None:
        s = ds.build_sample(0, ROW_POS)
        self.assertEqual(sum(s["counts"].values()), len(s["gt"]["elements"]))
        # own 열은 자체 텍스트가 있는 요소만 — STRUCTURE 는 정의상 0 이어야 한다.
        self.assertEqual(s["counts_own"]["STRUCTURE"], 0)
        self.assertEqual(
            sum(s["counts_own"].values()),
            sum(1 for e in s["gt"]["elements"] if e["own"]),
        )

    def test_reason_is_carried(self) -> None:
        """판정 근거 없이 색만 있으면 오분류를 판별할 수 없다."""
        s = ds.build_sample(0, ROW_POS)
        for el in s["gt"]["elements"]:
            self.assertIn("rule", el["reason"])
            self.assertTrue(el["reason"]["rule"])


class PromptFamilies(unittest.TestCase):
    """두 계열 모두에서 action 이 실제로 읽히는가."""

    def test_both_families_parse_action(self) -> None:
        self.assertEqual(
            ds.build_sample(0, ROW_POS)["action"],
            {"action": "type", "text": "hello world"},
        )
        self.assertEqual(
            ds.build_sample(0, ROW_IDX)["action"],
            {"action_type": "input_text", "index": "2", "text": "hello world"},
        )

    def test_action_that_cannot_be_parsed_fails_loudly(self) -> None:
        """Action 섹션이 있는데 dict 가 안 나오면 ACTION_* 라벨이 통째로 사라진다."""
        broken = {
            "messages": [
                {
                    "from": "human",
                    "value": f"Current UI State:\n{CUR_POS}\n\nAction:\nnot json",
                },
                {"from": "gpt", "value": GT_POS},
            ]
        }
        with self.assertRaises(ds.BuildError):
            ds.build_sample(0, broken)

    def test_missing_section_fails_loudly(self) -> None:
        broken = {
            "messages": [
                {"from": "human", "value": "Current UI State:\n<div/>"},  # action 없음
                {"from": "gpt", "value": GT_POS},
            ]
        }
        with self.assertRaises(ds.BuildError):
            ds.build_sample(0, broken)

    def test_slot_key_comes_from_classifier(self) -> None:
        """자리 키를 뷰어가 직접 조립하면 라벨을 잇는 축이 분류기와 갈린다."""
        for row in (ROW_POS, ROW_IDX):
            s = ds.build_sample(0, row)
            deriv = H.classify_derivability(
                ds.parse_prompt(row["messages"][1]["value"])["current_state"],
                row["messages"][2]["value"],
                H.extract_action(row["messages"][1]["value"]),
            )
            for el, d in zip(s["gt"]["elements"], deriv, strict=True):
                self.assertEqual(el["slot"], H.slot_key(d["element"]))
        # 두 계열이 서로 다른 축을 쓴다 — 접두로 구분된다.
        self.assertTrue(
            all(
                e["slot"].startswith("b:")
                for e in ds.build_sample(0, ROW_POS)["gt"]["elements"]
                if e["slot"]
            )
        )
        self.assertTrue(
            all(
                e["slot"].startswith("i:")
                for e in ds.build_sample(0, ROW_IDX)["gt"]["elements"]
                if e["slot"]
            )
        )

    def test_index_family_has_no_bounds(self) -> None:
        """bounds 없는 계열은 좌표 와이어프레임을 그릴 수 없다 → screen 이 None."""
        s = ds.build_sample(0, ROW_IDX)
        self.assertIsNone(s["gt"]["screen"])
        self.assertTrue(all("index" in e for e in s["gt"]["elements"]))
        # pos 계열은 반대로 screen 이 잡혀야 한다.
        self.assertEqual(ds.build_sample(0, ROW_POS)["gt"]["screen"], [840, 1800])


class Undecidable(unittest.TestCase):
    """ "판정 못 함" 과 "유도 불가능" 이 화면에서 섞이지 않는가."""

    # action 이 좌표를 주는데(pos 규약) GT 요소에 bounds 가 없는 혼합 케이스.
    # 이때 ACTION_TARGET 포함 판정 자체가 불가능하다.
    ROW_MIXED = {
        "messages": [
            {"from": "system", "value": "# Role"},
            {
                "from": "human",
                "value": 'Current UI State:\n<div bounds="[0,0][840,1800]"/>\n\n'
                'Action:\n<action>{"action": "click", "coordinate": [100, 200]}</action>',
            },
            {"from": "gpt", "value": "<div><p>brand new server text</p></div>"},
        ]
    }

    def test_rule_string_still_triggers(self) -> None:
        """상수만 비교하면 분류기 쪽 리네임을 못 잡는다 — 실제로 발동시켜 고정한다."""
        s = ds.build_sample(0, self.ROW_MIXED)
        rules = {e["reason"]["rule"] for e in s["gt"]["elements"]}
        self.assertIn(
            ds.UNDECIDABLE_RULE,
            rules,
            "분류기가 판정 불가 규칙 이름을 바꿨다 — 뷰어의 3분류가 무력화된다",
        )

    def test_flagged_and_counted_separately(self) -> None:
        s = ds.build_sample(0, self.ROW_MIXED)
        undec = [e for e in s["gt"]["elements"] if e.get("undecidable")]
        self.assertTrue(undec)
        for e in undec:
            # 라벨 자체는 분류기 그대로여야 한다 (뷰어가 라벨을 바꾸면 지표와 갈린다).
            self.assertEqual(e["label"], "NON_DERIVABLE")
            self.assertFalse(e["derivable"])
        self.assertEqual(s["undecidable"], len(undec))
        # NON_DERIVABLE 의 부분집합이다 — 합계에 이중으로 더하면 안 된다.
        self.assertEqual(sum(s["counts"].values()), len(s["gt"]["elements"]))
        self.assertLessEqual(s["undecidable"], s["counts"]["NON_DERIVABLE"])

    def test_has_its_own_color(self) -> None:
        css = ds._label_css()
        self.assertIn(".l7-UNDECIDABLE", css)
        self.assertIn(".d2-undec", css)
        self.assertNotEqual(
            ds.LABEL_COLOR["UNDECIDABLE"], ds.LABEL_COLOR["NON_DERIVABLE"]
        )


class ImePanelFlag(unittest.TestCase):
    """IME 제안 스트립 표식(`in_ime_panel`)이 뷰어까지 전달되는가.

    제안은 라벨 축과 독립이다 — 타이핑 접두 완성은 ACTION_PAYLOAD, IME 사전이
    만든 단어는 NON_DERIVABLE 로 떨어진다. 그래서 라벨 필터로는 이 부류를 훑을 수
    없고 별도 표식이 필요하다. 분류기가 `reason.in_ime_panel` 을 주는데 뷰어가
    그것을 버리면 감사 대상이 화면에서 사라진다.
    """

    def test_flag_is_forwarded_when_classifier_sets_it(self) -> None:
        rows = _load_real_rows("EXP05", 300)
        if not rows:
            self.skipTest("실데이터 없음 — 배선 검사만 하는 환경")
        seen = False
        for i, rec in rows:
            s = ds.build_sample(i, rec)
            deriv_flags = [e.get("ime") for e in s["gt"]["elements"]]
            if any(deriv_flags):
                seen = True
                for e in s["gt"]["elements"]:
                    self.assertEqual(
                        bool(e.get("ime")), bool(e["reason"].get("in_ime_panel"))
                    )
                break
        self.assertTrue(seen, "300행 안에 IME 제안 스트립 표본이 하나도 없다")


class HtmlOutput(unittest.TestCase):
    def test_payload_is_valid_json_and_script_safe(self) -> None:
        src = _write([ROW_POS, ROW_IDX])
        out = src.parent / "site" / "index.html"
        ds.build(src, out, samples=2, seed=1)
        html = out.read_text(encoding="utf-8")
        # <script> 는 한 쌍만 — payload 의 `</` 가 살아 있으면 조기 종료된다.
        self.assertEqual(html.count("<script>"), 1)
        self.assertEqual(html.count("</script>"), 1)
        m = re.search(r"const DATA = (\{.*\});\n", html)
        self.assertIsNotNone(m)
        raw = m.group(1)
        self.assertNotIn("</", raw)  # 전부 <\/ 로 이스케이프돼 있어야 한다
        data = json.loads(raw.replace("<\\/", "</"))
        self.assertEqual(len(data["rows"]), 2)
        self.assertEqual(data["labels"], list(H.DERIVABILITY_LABELS))
        self.assertEqual(data["derivable_labels"], sorted(H.DERIVABLE_LABELS))

    def test_every_label_has_a_color(self) -> None:
        """라벨이 추가됐는데 색이 없으면 화면에서 조용히 무색으로 그려진다."""
        css = ds._label_css()
        for lbl in H.DERIVABILITY_LABELS:
            self.assertIn(lbl, ds.LABEL_COLOR, f"{lbl} 색 미정의")
            self.assertIn(f".chip-{lbl}", css)
            self.assertIn(f".l7-{lbl}", css)

    def test_image_placeholder_stripped_from_display_only(self) -> None:
        """`<image>` 는 프롬프트의 이미지 placeholder 라 UI 요소가 아니다 —
        표시에서만 지우고 분류 입력은 원문 그대로여야 한다."""
        s = ds.build_sample(0, ROW_POS)
        self.assertNotIn("<image>", s["current"]["xml"])
        self.assertIn("Create a Trip", s["current"]["xml"])


if __name__ == "__main__":
    unittest.main()
