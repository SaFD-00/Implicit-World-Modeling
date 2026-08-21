"""diff loss v3 — 액션 유도성 축 회귀 테스트.

대상 (v1/v2 는 재현성 때문에 불가침이라 건드리지 않는다):
- ``hungarian_metric_v3``      : element 화이트리스트 제거 + own/absorbed 텍스트 분리
- ``hungarian_diff_v3``        : ``classify_derivability`` 7 라벨 + 판정 근거
- ``token_weight_builder_v3``  : (diff_type × derivability) 2축 가중치, innermost 병합
- ``preprocess_dataset_v2``    : ``--metric-version v3`` 런타임 훅

고정하려는 실패 모드:
  · 화이트리스트가 되살아나 컨테이너가 element 집합/ span 에서 빠지는 것
  · 유도성 판정에 흡수 텍스트가 섞여 루트가 NON_DERIVABLE 이 되는 것
  · SYSTEM_UI 가 앱 폼을 삼키는 것 (키캡 밀도 + bbox 상단 조건)
  · 조상(루트) span 이 max 병합으로 자손의 감쇠를 덮어써 NON_DERIVABLE 감쇠가
    0건이 되는 것  ← v3 의 핵심 수정
  · baseline 하드코딩 / action 샘플 uniform 분기 소실 (AGENTS.md 하드 제약 10)

Run:
    pytest tests/test_diff_loss_v3.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DIFF_LOSS = Path(__file__).resolve().parent.parent / "scripts" / "diff_loss"
sys.path.insert(0, str(DIFF_LOSS))

import hungarian_diff_v3 as hd3  # noqa: E402
import preprocess_dataset_v2 as pp  # noqa: E402
import token_weight_builder_v3 as tw3  # noqa: E402
from hungarian_metric_v3 import extract_elements, iter_nodes, parse_soup  # noqa: E402


class FakeTokenizer:
    """고정 폭 청크 토크나이저 — 토큰 경계를 정확히 아는 상태로 병합 규칙을 검사한다.

    (tests/test_diff_loss_v2.py 의 것과 같은 도구다. v2 테스트와 서로 import 하지 않고
    각자 들고 있는다 — 한쪽 픽스처를 고치다 다른 쪽을 조용히 깨뜨리지 않기 위해서다.)
    """

    def __init__(self, chunk: int = 4):
        self.chunk = chunk

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        for i, start in enumerate(range(0, len(text), self.chunk)):
            end = min(start + self.chunk, len(text))
            ids.append(i)
            offsets.append((start, end))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


# ── 픽스처 ────────────────────────────────────────────────────────────────

CURRENT = """<div bounds="[0,0][840,1876]">
  <p bounds="[10,10][200,60]">Inbox</p>
  <p bounds="[10,80][400,130]">Scroll me</p>
  <button bounds="[10,200][300,260]" aria-label="Compose"/>
</div>"""

FUTURE = """<div bounds="[0,0][840,1876]">
  <p bounds="[10,10][200,60]">Inbox</p>
  <div bounds="[0,150][840,600]">
    <p bounds="[10,300][400,350]">Scroll me</p>
    <button bounds="[10,200][300,260]">Sent</button>
    <p bounds="[10,400][500,450]">Galaxy S24 $999</p>
    <p bounds="[10,460][500,510]">hello world</p>
  </div>
</div>"""

ACTION = {"action": "click", "coordinate": [150, 230], "text": "hello"}


def _labels(current=CURRENT, future=FUTURE, action=ACTION):
    """own_text → 라벨 조회표 (텍스트 없는 요소는 tag/bounds 로 구분)."""
    out = {}
    for r in hd3.classify_derivability(current, future, action):
        el = r["element"]
        key = el["own_text"] or f"{el['tag']}{el['bounds']}"
        out[key] = r["derivability"]
    return out


# ── element 집합 (metric v3) ──────────────────────────────────────────────


def test_no_whitelist_every_node_is_an_element():
    """화이트리스트 제거 — 노드 수와 element 수가 정확히 같아야 한다.

    v2 는 aria-label 만 가진 div/컨테이너를 통째로 버려서, 화면 변화가 관측조차
    되지 않고 가중치 빌더에서도 baseline 으로 방치됐다.
    """
    nodes = iter_nodes(parse_soup(FUTURE))
    els = extract_elements(FUTURE)
    assert len(els) == len(nodes) > 0
    assert [e["seq_idx"] for e in els] == list(range(len(nodes)))
    assert all(e["tag"] == e["tag"].lower() for e in els)


def test_own_text_and_absorbed_text_are_separate():
    """컨테이너는 매칭용 text 로 자손을 흡수하되 own_text 는 비어 있어야 한다.

    이 분리가 깨지면 유도성 판정이 흡수 텍스트를 보게 되어 루트가 화면 전체 텍스트로
    NON_DERIVABLE 이 된다 (프로토타입에서 실제로 겪은 오분류).
    """
    els = extract_elements(FUTURE)
    root = els[0]
    assert root["own_text"] == ""
    assert "Inbox" in root["text"] and "Galaxy S24 $999" in root["text"]
    leaf = next(e for e in els if e["own_text"] == "Inbox")
    assert leaf["text"] == "Inbox"


def test_parser_falls_back_when_xml_yields_empty_tree():
    """코드펜스/프리앰블이 붙어 well-formed XML 이 아니어도 요소를 뽑아야 한다."""
    fenced = "## Predicted Next State\n```html\n" + FUTURE + "\n```"
    assert len(extract_elements(fenced)) == len(extract_elements(FUTURE))


# ── 유도성 라벨 ───────────────────────────────────────────────────────────


def test_copy_same_slot_same_text():
    lab = _labels()
    assert lab["Inbox"] == hd3.COPY


def test_reflow_text_moved_elsewhere():
    lab = _labels()
    assert lab["Scroll me"] == hd3.REFLOW


def test_action_payload_from_action_text():
    lab = _labels()
    assert lab["hello world"] == hd3.ACTION_PAYLOAD


def test_action_payload_from_open_app_name():
    future = '<div bounds="[0,0][840,1876]"><p bounds="[1,1][2,2]">AccuWeather</p></div>'
    res = hd3.classify_derivability(
        CURRENT, future, {"action": "open", "app_name": "AccuWeather"}
    )
    entry = next(r for r in res if r["element"]["own_text"] == "AccuWeather")
    assert entry["derivability"] == hd3.ACTION_PAYLOAD
    assert entry["reason"]["rule"] == "action_app_name_in_text"


def test_action_target_click_inside_bounds():
    lab = _labels()
    assert lab["Sent"] == hd3.ACTION_TARGET


def test_structure_when_no_own_text():
    lab = _labels()
    assert lab["div[0,0][840,1876]"] == hd3.STRUCTURE
    assert lab["div[0,150][840,600]"] == hd3.STRUCTURE


def test_non_derivable_server_content():
    """현재 화면에도 없고 action 이 준 것도 아닌 새 콘텐츠 = 유도 불가능."""
    lab = _labels()
    assert lab["Galaxy S24 $999"] == hd3.NON_DERIVABLE


def test_reason_carries_evidence():
    """뷰어가 라벨을 사람이 검증할 수 있게 근거를 실어야 한다."""
    res = hd3.classify_derivability(CURRENT, FUTURE, ACTION)
    copy = next(r for r in res if r["element"]["own_text"] == "Inbox")
    assert copy["reason"]["rule"] == "same_slot_same_text"
    assert copy["reason"]["matched_current"]["own_text"] == "Inbox"
    assert copy["reason"]["text_sim"] == 1.0
    target = next(r for r in res if r["element"]["own_text"] == "Sent")
    assert target["reason"]["coordinate"] == [150, 230]


def test_action_accepts_dict_json_or_user_message():
    """action 은 dict / JSON 문자열 / <action> 을 품은 user 메시지 모두 받는다."""
    as_json = '{"action": "click", "coordinate": [150, 230], "text": "hello"}'
    as_msg = f"Current UI State:\n{CURRENT}\n\nAction:\n<action>{as_json}</action>"
    assert _labels(action=as_json)["Sent"] == hd3.ACTION_TARGET
    assert _labels(action=as_msg)["Sent"] == hd3.ACTION_TARGET
    # action 이 없으면 ACTION_* 만 사라지고 나머지 판정은 유지된다
    no_act = _labels(action=None)
    assert no_act["Sent"] == hd3.NON_DERIVABLE
    assert no_act["Inbox"] == hd3.COPY


def test_derivability_aligns_with_diff_by_seq_idx():
    """두 축은 seq_idx 로 결합된다 — 길이·순서가 어긋나면 가중치가 조용히 뒤섞인다."""
    diff = hd3.classify_diff(CURRENT, FUTURE)
    deriv = hd3.classify_derivability(CURRENT, FUTURE, ACTION)
    assert len(diff) == len(deriv) == len(extract_elements(FUTURE))
    assert [d["future_seq_idx"] for d in diff] == [
        d["future_seq_idx"] for d in deriv
    ]
    assert all(
        a["element"]["bounds"] == b["element"]["bounds"] for a, b in zip(diff, deriv)
    )


def test_summarize_and_lookup():
    deriv = hd3.classify_derivability(CURRENT, FUTURE, ACTION)
    counts = hd3.summarize_derivability(deriv)
    assert set(counts) == set(hd3.DERIVABILITY_LABELS)  # 키가 항상 다 있다
    assert sum(counts.values()) == len(deriv)
    lookup = hd3.derivability_lookup(deriv)
    assert lookup[("p", hd3.slot_key({"bounds": "[10,400][500,450]"}))] == hd3.NON_DERIVABLE
    assert hd3.is_derivable(hd3.COPY) and not hd3.is_derivable(hd3.NON_DERIVABLE)


def test_slot_key_falls_back_to_index_when_bounds_absent():
    """EXP01~04·MobiBench HTML 에는 bounds 가 없고 index 만 있다.

    bounds 단독으로 "같은 자리" 를 판정하면 그 데이터셋에서 COPY 가 **0** 이 되고
    전부 REFLOW 로 흘러내린다. 조회표 키도 (tag, "") 로 붕괴한다.
    """
    cur = '<div index="0"><p index="1">Inbox</p><p index="2">Draft</p></div>'
    fut = '<div index="0"><p index="1">Inbox</p><p index="2">Sent mail</p></div>'
    res = hd3.classify_derivability(cur, fut, None)
    by_text = {r["element"]["own_text"]: r for r in res}
    assert by_text["Inbox"]["derivability"] == hd3.COPY
    assert by_text["Inbox"]["reason"]["rule"] == "same_slot_same_text"
    # 같은 슬롯의 새 콘텐츠라는 근거가 남아야 뷰어에서 눈으로 가른다
    assert by_text["Sent mail"]["reason"]["rule"] == "same_slot_new_content"
    assert by_text["Sent mail"]["reason"]["matched_current"]["index"] == "2"
    lookup = hd3.derivability_lookup(res)
    assert lookup[("p", "i:2")] == hd3.NON_DERIVABLE


def test_action_target_undecidable_without_bounds_is_flagged():
    """좌표는 있는데 bounds 가 없으면 '판정 불가' 를 근거에 남긴다.

    조용히 False 로 흘리면 NON_DERIVABLE 모집단이 데이터셋마다 다르게 부풀어
    분포 비교가 망가진다.
    """
    cur = '<div index="0"><p index="1">Inbox</p></div>'
    fut = '<div index="0"><p index="1">Inbox</p><p index="2">Novel</p></div>'
    res = hd3.classify_derivability(cur, fut, {"action": "click", "coordinate": [1, 2]})
    novel = next(r for r in res if r["element"]["own_text"] == "Novel")
    assert novel["derivability"] == hd3.NON_DERIVABLE
    assert novel["reason"]["rule"] == "action_target_undecidable_no_bounds"


# ── SYSTEM_UI (키캡 밀도) ─────────────────────────────────────────────────

_KEYS = "qwertyuiopasdfghjklzxcvbnm"


def _keyboard(top: int = 1200, n: int = 26) -> str:
    rows = "".join(
        f'<button bounds="[{i * 30},{top}][{i * 30 + 28},{top + 100}]" '
        f'aria-label="{_KEYS[i % 26]}"/>'
        for i in range(n)
    )
    return f'<div bounds="[0,{top}][840,1876]">{rows}</div>'


def _screen_with_keyboard(top: int = 1200, n: int = 26) -> str:
    """앱 폼(위) + 키보드 패널(아래) 한 화면. 실제 EXP05 레이아웃의 축소판."""
    return (
        '<div bounds="[0,0][840,1876]">'
        '  <div bounds="[0,0][840,1100]">'
        '    <p bounds="[10,100][400,150]">New contact</p>'
        '    <button bounds="[10,300][200,360]">Cancel</button>'
        '    <button bounds="[600,300][830,360]">More options</button>'
        '    <p bounds="[10,400][200,450]">ABC</p>'
        "  </div>"
        f"  {_keyboard(top, n)}"
        "</div>"
    )


def test_system_ui_labels_keyboard_panel():
    lab = _labels(current=CURRENT, future=_screen_with_keyboard(), action=None)
    assert lab["q"] == hd3.SYSTEM_UI
    assert lab["m"] == hd3.SYSTEM_UI


def test_system_ui_does_not_swallow_app_form():
    """★ 앱 콘텐츠를 SYSTEM_UI(유도 가능)로 넣으면 모델 실패가 지표에서 숨는다.

    키캡 밀도만 보면 루트도 26/30 으로 통과하지만 bbox 상단 조건이 막는다.
    (LCA + y임계 방식이 실측 doc#1·#5 에서 폼을 통째로 삼켰던 실패의 회귀 테스트.)
    """
    lab = _labels(current=CURRENT, future=_screen_with_keyboard(), action=None)
    assert lab["Cancel"] != hd3.SYSTEM_UI
    assert lab["More options"] != hd3.SYSTEM_UI
    assert lab["New contact"] != hd3.SYSTEM_UI
    assert lab["ABC"] != hd3.SYSTEM_UI  # 키캡 모양이지만 폼 안이라 패널 밖이다


def test_system_ui_needs_enough_keycaps():
    """키캡이 KEYCAP_MIN_COUNT 미만이면 아예 SYSTEM_UI 를 만들지 않는다 (과소 라벨 우선)."""
    lab = _labels(current=CURRENT, future=_screen_with_keyboard(n=10), action=None)
    assert hd3.SYSTEM_UI not in lab.values()


def test_system_ui_rejects_top_of_screen_panel():
    """화면 상단에 걸친 키 격자는 키보드가 아니다 — 앱 콘텐츠일 수 있다."""
    lab = _labels(current=CURRENT, future=_screen_with_keyboard(top=100), action=None)
    assert hd3.SYSTEM_UI not in lab.values()


# ── 2축 가중치 표 ─────────────────────────────────────────────────────────


def test_weight_map_covers_every_combination():
    wmap = tw3.make_weight_map()
    for diff_type in tw3.DIFF_TYPES:
        for label in hd3.DERIVABILITY_LABELS:
            assert (diff_type, label) in wmap


def test_non_derivable_add_mod_is_attenuated_only_there():
    wmap = tw3.make_weight_map(
        w_added=1.0, w_modified=1.0, w_unchanged=0.25, w_non_derivable=0.5
    )
    assert wmap[("ADDED", hd3.NON_DERIVABLE)] == 0.5
    assert wmap[("MODIFIED", hd3.NON_DERIVABLE)] == 0.5
    # 유도 가능한 변화는 full weight — 모델이 맞혀야 하는 것들이다
    for label in hd3.DERIVABLE_LABELS:
        assert wmap[("ADDED", label)] == 1.0
        assert wmap[("MODIFIED", label)] == 1.0
    # UNCHANGED 행은 유도성과 무관하게 baseline
    for label in hd3.DERIVABILITY_LABELS:
        assert wmap[("UNCHANGED", label)] == 0.25


def test_baseline_is_derived_from_wmap_not_hardcoded():
    """하드 제약 10 — baseline 을 하드코딩하면 diff 토큰이 baseline 에 방치된다."""
    assert tw3.baseline_weight(tw3.make_weight_map(w_unchanged=0.3)) == 0.3
    assert tw3.baseline_weight(tw3.make_weight_map(w_unchanged=1.0)) == 1.0


# ── char span / 토큰 가중치 ───────────────────────────────────────────────


def test_every_element_gets_a_span_even_with_duplicate_bounds():
    """v2 는 (tag, bounds) 첫 등장만 잡아 동일 bounds 중첩 컨테이너를 뭉갰다."""
    html = (
        '<div bounds="[0,0][840,1876]"><div bounds="[0,0][840,1876]">'
        '<p bounds="[0,0][840,1876]">x</p></div></div>'
    )
    cov = tw3.span_coverage(html)
    assert cov["unresolved"] == 0 and cov["spans"] == 3
    spans = tw3.get_element_char_spans(html)
    widths = [ce - cs for cs, ce, _ in spans]
    assert widths == sorted(widths, reverse=True)  # 바깥이 더 넓다 = 서로 다른 span


def test_span_covers_self_closing_and_unclosed_tags():
    html = '<div bounds="[0,0][1,1]"><button bounds="[2,2][3,3]"/>tail</div>'
    assert tw3.span_coverage(html)["unresolved"] == 0


def _weights(html, diff, deriv, wmap, chunk=4, prefix="P"):
    tok = FakeTokenizer(chunk=chunk)
    w = tw3.build_token_weights(
        tokenizer=tok,
        system="",
        user="",
        future_html=html,
        diff_result=diff,
        prefix_text=prefix,
        weight_map=wmap,
        deriv_result=deriv,
    )
    return w[len(tok(prefix)["input_ids"]) :]


def _fake_axes(html, diff_types, labels):
    """seq_idx → (diff_type, derivability) 를 손으로 지정한 축 결과."""
    els = extract_elements(html)
    diff = [
        {"element": el, "future_seq_idx": i, "diff_type": diff_types[i]}
        for i, el in enumerate(els)
    ]
    deriv = [
        {"element": el, "future_seq_idx": i, "derivability": labels[i]}
        for i, el in enumerate(els)
    ]
    return diff, deriv


NESTED = '<div bounds="[0,0][840,1876]"><p bounds="[10,400][500,450]">Galaxy S24</p></div>'


def test_innermost_span_wins_over_ancestor():
    """★ v3 의 핵심. 조상(루트) span 이 max 병합으로 자손 감쇠를 덮어쓰면 안 된다.

    EXP05 실측: future 루트는 current 루트와 태그가 다르거나(<node> vs <div>) 자손
    변화로 텍스트가 흔들려 30문서 중 28건이 ADDED/MODIFIED 로 잡힌다. 루트 span 은
    문서 전체라, max 병합이면 전 토큰이 full weight 가 되고 NON_DERIVABLE 감쇠가
    **0건**이 된다 (30문서 실측: max 병합 시 0.5 가중치 토큰 0개 / 89% 가 1.0,
    innermost 병합 시 25.0% 가 0.5).
    """
    wmap = tw3.make_weight_map()
    diff, deriv = _fake_axes(
        NESTED, ["MODIFIED", "ADDED"], [hd3.STRUCTURE, hd3.NON_DERIVABLE]
    )
    w = _weights(NESTED, diff, deriv, wmap)

    content_tok = NESTED.index("Galaxy") // 4
    assert w[content_tok] == 0.5  # 자손의 감쇠가 살아남는다
    assert w[0] == 1.0  # 루트 여는 태그는 루트의 가중치
    assert 0.5 in w and 1.0 in w


def test_uncovered_tokens_get_baseline():
    """어떤 요소에도 안 걸린 토큰은 wmap 에서 유도한 baseline 을 받는다."""
    wmap = tw3.make_weight_map(w_unchanged=0.3)
    html = "zzzz" + NESTED
    diff, deriv = _fake_axes(
        html, ["MODIFIED", "ADDED"], [hd3.STRUCTURE, hd3.NON_DERIVABLE]
    )
    w = _weights(html, diff, deriv, wmap)
    assert w[0] == 0.3  # 'zzzz' — 루트 밖


def test_left_straddling_token_gets_weight():
    """interval overlap 유지 — v1 의 '토큰 시작점만' 비대칭 버그를 되살리지 않는다."""
    wmap = tw3.make_weight_map()
    html = "zz" + NESTED
    diff, deriv = _fake_axes(
        html, ["MODIFIED", "ADDED"], [hd3.STRUCTURE, hd3.NON_DERIVABLE]
    )
    w = _weights(html, diff, deriv, wmap)
    assert w[0] == 1.0  # 'zz<d' — 루트 왼쪽 경계를 걸친 토큰도 가중치를 받는다


def test_without_deriv_result_falls_back_to_one_axis():
    """deriv_result 가 없으면 v2 와 같은 1축 거동 (감쇠 없음) — 안전한 방향."""
    wmap = tw3.make_weight_map()
    diff, _ = _fake_axes(
        NESTED, ["MODIFIED", "ADDED"], [hd3.STRUCTURE, hd3.NON_DERIVABLE]
    )
    w = _weights(NESTED, diff, None, wmap)
    assert 0.5 not in w


def test_end_to_end_attenuates_non_derivable_content():
    """실제 두 축을 돌려 붙였을 때도 감쇠가 살아남는가 (손으로 만든 축이 아니라)."""
    wmap = tw3.make_weight_map()
    diff = hd3.classify_diff(CURRENT, FUTURE)
    deriv = hd3.classify_derivability(CURRENT, FUTURE, ACTION)
    w = _weights(FUTURE, diff, deriv, wmap)
    galaxy_tok = FUTURE.index("Galaxy") // 4
    assert w[galaxy_tok] == 0.5


# ── preprocess 런타임 훅 ──────────────────────────────────────────────────


def test_load_metric_binds_diff_and_builder_together():
    """v3 는 weight_map 키 형태가 달라 빌더도 함께 갈려야 한다."""
    pp._load_metric("v3")
    assert pp._hd.__name__ == "hungarian_diff_v3"
    assert pp._tw.__name__ == "token_weight_builder_v3"
    pp._load_metric("v2")
    assert pp._hd.__name__ == "hungarian_diff_v2"
    assert pp._tw.__name__ == "token_weight_builder_v2"


def test_action_sample_stays_uniform_under_v3():
    """하드 제약 10 — action 샘플의 uniform 분기가 빠지면 '전부 최저 가중치' 가 된다."""
    pp._load_metric("v3")
    sample = {
        "messages": [
            {"from": "human", "value": f"Current UI State:\n{CURRENT}"},
            {"from": "gpt", "value": '<action>{"action": "click"}</action>'},
        ],
        "images": ["a.png", "b.png"],  # 2장 = action 샘플
    }
    out, status = pp.process_sample(
        sample, FakeTokenizer(), lambda s, u: "P", tw3.make_weight_map(), 0, "fail"
    )
    assert status == "action"
    assert set(out["token_weights"]) == {1.0}


def test_state_sample_uses_two_axis_weights():
    pp._load_metric("v3")
    user = (
        f"Current UI State:\n{CURRENT}\n\n[Screenshot]\n<image>\n\nAction:\n"
        '<action>{"action": "click", "coordinate": [150, 230], "text": "hello"}</action>'
    )
    sample = {
        "messages": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": FUTURE},
        ],
        "images": ["a.png"],  # 1장 = state 샘플
    }
    out, status = pp.process_sample(
        sample, FakeTokenizer(), lambda s, u: "P", tw3.make_weight_map(), 0, "fail"
    )
    assert status == "ok"
    assert out["_deriv_counts"][hd3.NON_DERIVABLE] >= 1
    assert 0.5 in out["token_weights"]  # 유도 불가능 콘텐츠가 감쇠됐다
    assert 1.0 in out["token_weights"]  # 유도 가능한 변화는 full weight


@pytest.fixture(autouse=True)
def _restore_metric():
    """모듈 전역(_hd/_tw)을 만지는 테스트가 다른 테스트에 새지 않게 되돌린다."""
    yield
    pp._load_metric("v2")


# ── index 계열(EXP01~04) 폴백 회귀 ────────────────────────────────────────
# bounds 가 없고 index 만 있는 XML 에서 COPY / ACTION_TARGET 이 조용히 0 이 되던
# 퇴화를 고정한다. 실측(EXP01 120문서): 수정 전 COPY 0.0% / REFLOW 16.2%,
# 수정 후 COPY 7.6% / REFLOW 8.6% — COPY 로 갈 요소가 전부 REFLOW 로 흘러내렸었다.
_IDX_CURRENT = (
    '<node index="0">'
    '<button index="13" description="Save">Save</button>'
    '<p index="14">Inbox</p>'
    "</node>"
)
_IDX_FUTURE = (
    '<node index="0">'
    '<button index="13" description="Save">Save</button>'
    '<p index="14">Inbox</p>'
    '<p index="15">Rachit Kumar, Lucknow</p>'
    "</node>"
)
_IDX_ACTION = {"action_type": "click", "index": "13"}


def test_index_fallback_yields_copy_and_action_target():
    deriv = hd3.classify_derivability(_IDX_CURRENT, _IDX_FUTURE, _IDX_ACTION)
    by_idx = {d["element"]["index"]: d for d in deriv}
    # index 13 은 액션 대상이자 내용이 그대로다 — COPY 가 먼저 걸린다(더 강한 근거).
    assert by_idx["13"]["derivability"] == hd3.COPY
    assert by_idx["13"]["reason"]["rule"] == "same_slot_same_text"
    # index 14 도 같은 자리·같은 내용
    assert by_idx["14"]["derivability"] == hd3.COPY
    # index 15 는 current 에 근거가 없는 새 콘텐츠
    assert by_idx["15"]["derivability"] == hd3.NON_DERIVABLE


def test_index_action_target_when_content_changed():
    """액션 대상이면서 내용이 바뀐 요소는 ACTION_TARGET 이다 (COPY 로 안 걸린다)."""
    future = (
        '<node index="0">'
        '<button index="13" description="Saved">Saved</button>'
        "</node>"
    )
    deriv = hd3.classify_derivability(_IDX_CURRENT, future, _IDX_ACTION)
    tgt = next(d for d in deriv if d["element"]["index"] == "13")
    assert tgt["derivability"] == hd3.ACTION_TARGET
    assert tgt["reason"]["rule"] == "action_index_matches_element"


def test_slot_key_prefixes_do_not_collide():
    """bounds 축과 index 축이 같은 키 공간을 공유하면 안 된다."""
    assert hd3.slot_key({"bounds": "[0,0][1,1]"}) != hd3.slot_key({"index": "[0,0][1,1]"})
    assert hd3.slot_key({"bounds": "", "index": "3"}) == hd3.slot_key({"index": "3"})
    assert hd3.slot_key({"bounds": "", "index": ""}) == ""


def test_extract_action_handles_untagged_format():
    """EXP01~04 프롬프트는 `<action>` 태그가 없다 — 그래도 파싱돼야 한다."""
    tagged = 'Action:\n<action>{"action": "click", "coordinate": [1, 2]}</action>'
    untagged = "## Action\n" + '{"action_type":"click","index":"13"}'
    assert hd3.action_type(hd3.extract_action(tagged)) == "click"
    a = hd3.extract_action(untagged)
    assert a == {"action_type": "click", "index": "13"}
    assert hd3.action_type(a) == "click"
    # action space 설명의 예시 JSON 을 액션으로 오인하면 안 된다
    noisy = (
        '1. {"action": "click", "coordinate": [x, y]}\n'
        '2. {"action": "type", "text": "<text>"}\n'
        "## Action\n" + '{"action_type":"scroll","direction":"down"}'
    )
    assert hd3.extract_action(noisy) == {"action_type": "scroll", "direction": "down"}


# ── IME 제안 스트립이 SYSTEM_UI 로 새어들지 않는다 ────────────────────────
# 실측(EXP05+EXP07 300문서): SYSTEM_UI 3,129건 중 86건(2.7%)이 한 문서에만 나타나는
# 텍스트였고 전부 IME 사전 출력이었다 — "Vegas"/"Dresses"/emoji/자동완성 이메일.
# 제안을 SYSTEM_UI 로 두면 사용자·앱 콘텐츠가 "유도 가능"이 되어 학습에서 full weight
# 를 받고 채점에서는 실패가 숨는다.
def _kbd(extra_rows: str = "") -> str:
    keys = "".join(
        f'<button bounds="[{i*40},1400][{i*40+38},1450]" aria-label="{c}"/>'
        for i, c in enumerate("qwertyuiopasdfghjklzxcvbnm")
    )
    return (
        '<div bounds="[0,1300][1040,1876]">'
        f"{extra_rows}"
        '<button bounds="[0,1500][100,1550]" aria-label="Shift"/>'
        '<button bounds="[100,1500][200,1550]" aria-label="Delete"/>'
        f"{keys}</div>"
    )


def _screen(panel: str) -> str:
    return f'<node bounds="[0,0][1040,1876]">{panel}</node>'


def test_ime_suggestion_is_not_system_ui():
    current = _screen(_kbd('<p bounds="[0,1330][300,1370]">Vega</p>'))
    future = _screen(
        _kbd(
            '<p bounds="[0,1330][300,1370]">Vega</p>'
            '<button bounds="[300,1330][600,1370]" aria-label="Dresses"/>'
        )
    )
    deriv = hd3.classify_derivability(current, future, {"action": "type", "text": "Vega"})
    by_text = {d["element"]["own_text"]: d for d in deriv}
    # 유지된 크롬은 그대로 SYSTEM_UI
    assert by_text["Shift"]["derivability"] == hd3.SYSTEM_UI
    assert by_text["q"]["derivability"] == hd3.SYSTEM_UI
    # 패널 안에서 새로 생긴 텍스트는 SYSTEM_UI 가 아니다
    sug = by_text["Dresses"]
    assert sug["derivability"] != hd3.SYSTEM_UI
    assert sug["reason"].get("in_ime_panel") is True


def test_ime_suggestion_that_completes_the_payload_is_action_payload():
    """타이핑한 문자열의 접두 완성은 유도 가능하다 — 규칙 사슬이 제자리를 찾아준다."""
    current = _screen(_kbd())
    future = _screen(
        _kbd('<button bounds="[300,1330][600,1370]" aria-label="Vega city"/>')
    )
    # current 에 패널이 있으므로 규칙이 켜진다
    deriv = hd3.classify_derivability(current, future, {"action": "type", "text": "Vega"})
    sug = next(d for d in deriv if d["element"]["own_text"] == "Vega city")
    assert sug["derivability"] == hd3.ACTION_PAYLOAD


def test_freshly_opened_keyboard_stays_system_ui():
    """키보드가 이번에 처음 열렸으면 패널 전체가 신규다 — 규칙을 켜면 안 된다."""
    current = _screen('<p bounds="[0,100][300,140]">Inbox</p>')
    future = _screen(_kbd('<p bounds="[0,1330][300,1370]">Hi</p>'))
    deriv = hd3.classify_derivability(current, future, {"action": "click", "coordinate": [1, 1]})
    by_text = {d["element"]["own_text"]: d for d in deriv}
    assert by_text["Shift"]["derivability"] == hd3.SYSTEM_UI
    assert by_text["Hi"]["derivability"] == hd3.SYSTEM_UI
