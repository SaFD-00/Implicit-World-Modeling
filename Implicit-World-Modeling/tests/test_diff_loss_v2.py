"""diff loss v2 회귀 테스트 — S2-05 / S2-03 / S2-09 수정 고정.

대상 (v1 은 EXP02 재현성 때문에 불가침이라 테스트하지 않는다):
- ``token_weight_builder_v2.build_token_weights``
  · S2-05: 토큰 시작점만 보던 비대칭 경계 → **interval overlap** 으로 교정.
    element 왼쪽 경계를 걸친 토큰(앞 텍스트와 합쳐 토크나이즈된 것)도 가중치를 받는다.
  · 중첩 span 에 동시에 걸친 토큰은 **더 큰 가중치**를 채택 (순서 비의존).
- ``preprocess_dataset_v2.preprocess``
  · S2-09: ``--input == --output`` 거부 + 실패 시 부분 산출물(.tmp) 미잔류 (원자 교체)
  · S2-03: fail-closed 기본 + fallback 을 성공으로 집계하지 않음 + sidecar 기록

Run:
    pytest tests/test_diff_loss_v2.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DIFF_LOSS = Path(__file__).resolve().parent.parent / "scripts" / "diff_loss"
sys.path.insert(0, str(DIFF_LOSS))

import preprocess_dataset_v2 as pp  # noqa: E402
from token_weight_builder_v2 import build_token_weights  # noqa: E402

WMAP = {"ADDED": 1.0, "MODIFIED": 1.0, "UNCHANGED": 0.25}
BASE = WMAP["UNCHANGED"]


class FakeTokenizer:
    """고정 폭 청크 토크나이저 — 경계 케이스를 정확히 구성하기 위한 것.

    chunk 크기 n 으로 문자열을 자르고 offset_mapping 을 그대로 돌려준다.
    zero_at 에 지정한 토큰 인덱스는 zero-length offset (special token 모사).
    """

    def __init__(self, chunk: int = 4, zero_at: set[int] | None = None):
        self.chunk = chunk
        self.zero_at = zero_at or set()

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        for i, start in enumerate(range(0, len(text), self.chunk)):
            end = min(start + self.chunk, len(text))
            ids.append(i)
            offsets.append((0, 0) if i in self.zero_at else (start, end))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def _weights(html, diff_result, tok, prefix="P"):
    """assistant 구간 weight 만 잘라서 반환."""
    w = build_token_weights(
        tokenizer=tok,
        system="",
        user="",
        future_html=html,
        diff_result=diff_result,
        prefix_text=prefix,
        weight_map=WMAP,
    )
    n_prefix = len(tok(prefix)["input_ids"])
    return w[n_prefix:]


# ── S2-05: 왼쪽 경계를 걸친 토큰 ──────────────────────────────────────────


def test_left_straddling_token_gets_weight():
    # 'xx' 뒤에 element 가 시작한다 → char_start=2.
    # chunk=4 라 첫 토큰 offset (0,4) 는 char_start(2) 를 **걸친다**.
    # 시작점만 보던 구버전은 tok_cs=0 < 2 라 이 토큰을 놓쳤다.
    html = 'xx<p bounds="[1,2][3,4]">Hi</p>'
    tok = FakeTokenizer(chunk=4)
    diff = [{"element": {"tag": "p", "index": -1, "bounds": "[1,2][3,4]", "text": "Hi"}, "diff_type": "MODIFIED"}]

    w = _weights(html, diff, tok)
    offsets = tok(html, return_offsets_mapping=True)["offset_mapping"]

    char_start = html.index("<p")
    assert offsets[0] == (0, 4) and offsets[0][0] < char_start < offsets[0][1], "전제: 첫 토큰이 왼쪽 경계를 걸쳐야 한다"
    assert w[0] == 1.0, "왼쪽 경계를 걸친 토큰이 가중치를 받아야 한다 (S2-05)"
    # element 안쪽 토큰도 당연히 가중치
    assert all(x == 1.0 for x in w[1:])


def test_no_overlap_token_keeps_baseline():
    # element 가 뒤쪽에만 있고, 앞 토큰들은 전혀 겹치지 않는다 → baseline 유지.
    html = 'xxxxxxxx<p bounds="[1,2][3,4]">Hi</p>'
    tok = FakeTokenizer(chunk=4)
    diff = [{"element": {"tag": "p", "index": -1, "bounds": "[1,2][3,4]", "text": "Hi"}, "diff_type": "MODIFIED"}]

    w = _weights(html, diff, tok)
    assert w[0] == BASE and w[1] == BASE, "겹치지 않는 토큰은 baseline"
    assert w[2] == 1.0, "element 가 시작되는 토큰부터 가중치"


def test_zero_length_offsets_skipped():
    html = 'xx<p bounds="[1,2][3,4]">Hi</p>'
    tok = FakeTokenizer(chunk=4, zero_at={1})  # 두 번째 토큰이 (0,0)
    diff = [{"element": {"tag": "p", "index": -1, "bounds": "[1,2][3,4]", "text": "Hi"}, "diff_type": "MODIFIED"}]

    w = _weights(html, diff, tok)
    assert w[1] == BASE, "zero-length offset 토큰은 어디에도 겹치지 않으므로 baseline"


def test_unchanged_element_stays_baseline():
    html = '<p bounds="[1,2][3,4]">Hi</p>'
    tok = FakeTokenizer(chunk=4)
    diff = [{"element": {"tag": "p", "index": -1, "bounds": "[1,2][3,4]", "text": "Hi"}, "diff_type": "UNCHANGED"}]

    w = _weights(html, diff, tok)
    assert set(w) == {BASE}, "UNCHANGED 는 baseline 유지"


def test_nested_spans_take_max_weight():
    # 바깥 div(UNCHANGED=baseline) 안에 button(ADDED=1.0). 겹치는 토큰은 max 를 채택.
    html = '<div bounds="[0,0][9,9]"><button bounds="[1,1][2,2]">X</button></div>'
    tok = FakeTokenizer(chunk=8)
    diff = [
        {"element": {"tag": "div", "index": -1, "bounds": "[0,0][9,9]", "text": ""}, "diff_type": "UNCHANGED"},
        {"element": {"tag": "button", "index": -1, "bounds": "[1,1][2,2]", "text": "X"}, "diff_type": "ADDED"},
    ]
    w = _weights(html, diff, tok)
    b_start = html.index("<button")
    b_end = html.index("</button>") + len("</button>")
    offsets = tok(html, return_offsets_mapping=True)["offset_mapping"]
    for i, (cs, ce) in enumerate(offsets):
        overlaps_button = cs < b_end and ce > b_start
        if overlaps_button:
            assert w[i] == 1.0, f"button 과 겹친 토큰 {i} 은 max 가중치 1.0"


# ── S2-09 / S2-03: preprocess 파이프라인 ───────────────────────────────────


def _sample(images: int, asst: str) -> dict:
    return {
        "messages": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "Current UI State:\n<p bounds=\"[1,2][3,4]\">A</p>\n[Screenshot]"},
            {"from": "gpt", "value": asst},
        ],
        "images": ["a.jpg"] * images,
    }


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(pp.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: FakeTokenizer()))
    return pp


def test_input_equals_output_rejected(tmp_path, patched):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps(_sample(2, "<action>{}</action>")) + "\n")
    with pytest.raises(ValueError, match="같은 경로"):
        patched.preprocess(str(p), str(p), "Qwen/fake")


def test_atomic_no_partial_output_on_failure(tmp_path, patched, monkeypatch):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    src.write_text(
        json.dumps(_sample(1, '<p bounds="[1,2][3,4]">B</p>')) + "\n"
        + json.dumps(_sample(1, '<p bounds="[1,2][3,4]">C</p>')) + "\n"
    )

    # 두 번째 레코드에서 diff 실패를 강제
    calls = {"n": 0}
    real_load = patched._load_metric

    def boom_load(v):
        real_load(v)

        def classify(cur, fut):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return []

        patched._hd.classify_diff = classify

    monkeypatch.setattr(patched, "_load_metric", boom_load)

    with pytest.raises(patched.SampleFailure):
        patched.preprocess(str(src), str(out), "Qwen/fake", on_error="fail")

    assert not out.exists(), "실패 시 출력 파일이 생기면 안 된다 (원자 교체)"
    assert not (tmp_path / "out.jsonl.tmp").exists(), "부분 산출물(.tmp)이 남으면 안 된다"


def test_action_sample_uniform_and_counted_separately(tmp_path, patched):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    src.write_text(json.dumps(_sample(2, "<action>{}</action>")) + "\n")

    meta = patched.preprocess(str(src), str(out), "Qwen/fake")
    rec = json.loads(out.read_text().strip())

    assert set(rec["token_weights"]) == {1.0}, "action 샘플은 uniform 1.0"
    assert meta["counts"]["action"] == 1
    assert meta["counts"]["ok"] == 0, "action 은 diff 성공으로 집계하지 않는다"
    assert Path(str(out) + ".meta.json").is_file(), "sidecar 메타데이터 기록"


def test_uniform_fallback_not_counted_as_ok(tmp_path, patched, monkeypatch):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    src.write_text(json.dumps(_sample(1, '<p bounds="[1,2][3,4]">B</p>')) + "\n")

    real_load = patched._load_metric

    def boom_load(v):
        real_load(v)
        patched._hd.classify_diff = lambda c, f: (_ for _ in ()).throw(RuntimeError("boom"))

    monkeypatch.setattr(patched, "_load_metric", boom_load)

    meta = patched.preprocess(str(src), str(out), "Qwen/fake", on_error="uniform")
    assert meta["counts"]["diff_fail"] == 1
    assert meta["counts"]["ok"] == 0, "fallback 을 성공으로 집계하면 안 된다 (S2-03)"
    assert meta["counts"]["written"] == 1
    assert set(json.loads(out.read_text().strip())["token_weights"]) == {1.0}


def test_skip_mode_drops_failed_records(tmp_path, patched, monkeypatch):
    src = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    src.write_text(json.dumps(_sample(1, '<p bounds="[1,2][3,4]">B</p>')) + "\n")

    real_load = patched._load_metric

    def boom_load(v):
        real_load(v)
        patched._hd.classify_diff = lambda c, f: (_ for _ in ()).throw(RuntimeError("boom"))

    monkeypatch.setattr(patched, "_load_metric", boom_load)

    meta = patched.preprocess(str(src), str(out), "Qwen/fake", on_error="skip")
    assert meta["counts"]["skipped"] == 1
    assert meta["counts"]["written"] == 0
    assert out.read_text() == ""
