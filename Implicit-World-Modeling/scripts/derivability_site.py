"""유도성(derivability) 라벨을 실제 데이터에서 **눈으로 감사**하는 단독 사이트 빌더.

왜 필요한가
───────────
`hungarian_diff_v3.classify_derivability` 는 GT next-state 의 요소마다
"현재 state + action 만으로 유도 가능한가" 를 라벨링한다. 이 라벨이 채점(유도 가능/
불가능 분리 지표)과 학습 가중치를 동시에 좌우하는데, **분포표만으로는 오분류를 못
잡는다.** 프로토타입 단계에서 실제로 잡힌 오분류들 — "키보드 전체가 NON_DERIVABLE",
"연락처 폼의 Cancel/More options 가 SYSTEM_UI 로 삼켜짐" — 은 전부 표본을 눈으로
훑어서 발견됐다. 이 스크립트는 그 감사 루프를 재현 가능하게 만든다.

예측이 필요 없다. 입력은 test jsonl 하나뿐이다 — current state · action · GT next
state 가 모두 그 안에 있고, 유도성은 그 셋만으로 정의된다.

사용법
──────
    python scripts/derivability_site.py \
        --test data/AndroidControl_EXP05/stage1_test_id_state.jsonl \
        --samples 40 --seed 42 \
        --out outputs/_compare/derivability_ac_exp05_id/index.html

설계 원칙
─────────
1. **판정 규칙을 여기서 다시 구현하지 않는다.** 정본은 `hungarian_diff_v3` 하나다.
   뷰어가 자체 휴리스틱을 가지면 화면과 지표가 갈린다.
2. **요소 집합도 분류기와 같은 것을 쓴다.** XML 을 JS 정규식으로 다시 파싱하면
   (`_compare_site.parseNodes` 가 그렇다: bounds 없는 노드를 버리고 420개에서 자른다)
   색칠 대상 집합이 라벨 집합과 달라진다. 그래서 파싱·요소화는 전부 Python 에서
   `iter_nodes(parse_soup(...))` 로 하고, JS 는 그 배열만 그린다. depth·index 속성은
   **표시용 메타**라 규칙 재구현이 아니다.
3. **유도 가능/불가능 2분류를 JS 가 계산하지 않는다.** 요소마다 분류기의
   `is_derivable` 결과를 그대로 실어 보낸다 — 라벨 문자열 집합을 JS 에 복제하는
   순간 그게 갈림길이 된다.
4. 프롬프트 파서는 `_prompt_sections.parse_prompt` **유일본**을 쓰고, 필수 섹션을
   못 읽으면 조용히 빈 화면을 내지 않고 SystemExit 한다 (woa 필터 사고 2026-07-30 의
   실패 모드가 "파싱 실패인데 산출물은 정상 모양" 이었다).
5. CSS/JS 는 `_compare_site` 에서 import 하지 않고 이 파일 안에 둔다. 이유는
   재사용 가치가 없어서가 아니라 **폭발 반경** 이다: `import _compare_site` 는
   `_action_eval`·`_hungarian_eval`·`_state_diff_eval`·`thought_eval` 을 함께 끌고
   오는데 전부 다른 소유자가 동시 편집 중이고, `SITE_JS` 는 애초에 DATA.task·
   splits·match_mode·annotations 에 결합돼 있어 구조적으로 재사용 불가다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
for _p in (_SCRIPTS, _SCRIPTS / "diff_loss"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _prompt_sections import REQUIRED_SECTIONS, parse_prompt  # noqa: E402
from hungarian_diff_v3 import (  # noqa: E402
    DERIVABILITY_LABELS,
    DERIVABLE_LABELS,
    action_type,
    classify_derivability,
    extract_action,
    slot_key,
    summarize_derivability,
)
from hungarian_metric_v3 import (  # noqa: E402
    build_element_records,
    iter_nodes,
    parse_bounds,
    parse_soup,
)

# bs4 가 XML 을 html.parser 로 읽을 때 행마다 내는 경고. 판정과 무관한 노이즈라
# 진짜 경고를 덮지 않도록 여기서만 끈다 (`_compare_site` 와 같은 처리).
try:
    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # bs4 없는 환경
    pass

#: 라벨별 색. 7색 모드의 정본이고 CSS 는 이 dict 에서 생성한다 — 색을 CSS 문자열에
#: 직접 쓰면 라벨이 추가됐을 때 조용히 무색으로 그려진다.
LABEL_COLOR: dict[str, str] = {
    "COPY": "#2563eb",
    "REFLOW": "#0891b2",
    "ACTION_PAYLOAD": "#7c3aed",
    "ACTION_TARGET": "#d97706",
    "SYSTEM_UI": "#64748b",
    "STRUCTURE": "#94a3b8",
    "NON_DERIVABLE": "#dc2626",
    # 라벨은 아니지만 화면에서는 일곱 라벨과 같은 자격으로 구분돼야 한다.
    "UNDECIDABLE": "#78716c",
}

#: 2분류 모드 색. 동료가 1차적으로 보려는 것이 이 두 색이다.
#: `undec` 는 셋째 색이다 — 2분류의 어느 쪽도 아니라 회색으로 뺀다.
D2_COLOR = {"yes": "#059669", "no": "#dc2626", "undec": "#78716c"}

#: "유도 불가능" 이 아니라 **"판정 못 함"** 인 요소의 규칙 이름.
#: `classify_derivability` 는 이 경우에도 라벨을 NON_DERIVABLE 로 두지만, 뜻이 다르다 —
#: action 이 좌표를 줬는데 요소에 bounds 가 없어 포함 판정 자체가 불가능한 것이다.
#: 진짜 NON_DERIVABLE 과 같은 색으로 칠하면 오분류 사냥의 모집단이 오염된다.
#: 분류기가 이 문자열을 바꾸면 구분이 조용히 사라지므로 테스트가 실동작으로 고정한다.
UNDECIDABLE_RULE = "action_target_undecidable_no_bounds"

#: 흡수 텍스트(`element["text"]`)는 루트 컨테이너에서 화면 전체 텍스트가 되므로
#: 그대로 실으면 payload 가 요소 수 × 화면 길이로 부푼다. 표시용으로만 자른다.
TEXT_CLIP = 160


class BuildError(SystemExit):
    """빌드 중단 — 값이 아니라 배선이 깨졌을 때만 낸다."""


# ── 행 → payload ─────────────────────────────────────────────────────────


def _depths(nodes: list) -> list[int]:
    """노드별 트리 깊이 (표시 전용). bounds 없는 계열의 트리 뷰 들여쓰기에 쓴다."""
    out = []
    for n in nodes:
        d = 0
        p = n.parent
        while p is not None and getattr(p, "name", None) not in (None, "[document]"):
            d += 1
            p = p.parent
        out.append(d)
    return out


def _screen(elements: list[dict]) -> list[int] | None:
    """bounds 로 추정한 화면 크기. 하나도 없으면 None (= 트리 뷰로 그린다)."""
    mx = my = 0
    for e in elements:
        b = e.get("bounds")
        if b:
            mx = max(mx, b[2])
            my = max(my, b[3])
    return [mx, my] if mx > 0 and my > 0 else None


def _pack(nodes: list, records: list[dict]) -> list[dict]:
    """분류기가 쓰는 element record + 표시용 메타(depth).

    `index` 와 `slot` 은 손으로 만들지 않는다 — `index` 는 element record 가 이미
    싣고 있고, `slot` 은 분류기의 `slot_key()` 가 정본이다. 뷰어가 자체 키 규칙을
    가지면 (bounds 문자열을 직접 조립하는 등) 라벨을 잇는 축이 분류기와 갈린다.
    """
    depths = _depths(nodes)
    out = []
    for rec, depth in zip(records, depths, strict=True):
        b = parse_bounds(rec["bounds"])
        item = {
            "seq": rec["seq_idx"],
            "tag": rec["tag"],
            "own": rec["own_text"],
            "depth": depth,
            "slot": slot_key(rec),
        }
        if b:
            item["bounds"] = list(b)
        if rec.get("index"):
            item["index"] = str(rec["index"])
        # 자체 텍스트가 없는 요소만 흡수 텍스트를 보여준다 — own 이 있으면 흡수
        # 텍스트는 own 과 같아서(build_element_records) 중복이다.
        if not rec["own_text"] and rec["text"]:
            item["absorbed"] = rec["text"][:TEXT_CLIP]
        # bounds 문자열은 있는데 파싱이 안 되면(형식 깨짐) 원문을 근거로 남긴다.
        if rec["bounds"] and not b:
            item["bounds_raw"] = rec["bounds"]
        out.append(item)
    return out


def build_sample(row_idx: int, rec: dict) -> dict:
    """jsonl 한 행 → 사이트 표본 하나. 필수 섹션이 없으면 BuildError."""
    msgs = rec.get("messages") or []
    human = next((m["value"] for m in msgs if m.get("from") == "human"), "")
    gt_xml = next((m["value"] for m in msgs if m.get("from") == "gpt"), "")
    sections = parse_prompt(human)
    missing = [k for k in REQUIRED_SECTIONS["state"] if not sections.get(k)]
    if missing or not gt_xml:
        raise BuildError(
            f"[derivability_site] row {row_idx}: 필수 섹션 누락 {missing}"
            f"{' + gpt 응답 없음' if not gt_xml else ''} — 프롬프트 계열이 늘었거나"
            " 데이터가 깨졌다. 파서(_prompt_sections)를 먼저 확인하라."
        )

    cur_xml = sections["current_state"]
    action_raw = sections["action"]
    # action 파싱은 분류기의 공개 진입점 하나만 쓴다. `extract_action` 이 두 규약
    # (`<action>{...}</action>` / 태그 없는 `## Action`) 을 모두 흡수하므로 뷰어가
    # 계열 분기를 갖지 않는다 — 화면의 action 과 판정의 action 이 갈리면 안 된다.
    action = extract_action(human)
    if action_raw and not action:
        raise BuildError(
            f"[derivability_site] row {row_idx}: Action 섹션은 있는데 "
            f"extract_action 이 빈 dict 를 냈다 ({action_raw[:80]!r}) — 프롬프트 "
            "규약이 하나 더 생겼다. 조용히 넘기면 ACTION_* 라벨이 통째로 사라진다."
        )

    # 판정에도 **같은 dict** 를 넘긴다 (문자열을 다시 넘기면 파싱 경로가 갈린다).
    deriv = classify_derivability(cur_xml, gt_xml, action)

    gt_nodes = iter_nodes(parse_soup(gt_xml))
    if len(gt_nodes) != len(deriv):
        raise BuildError(
            f"[derivability_site] row {row_idx}: 요소 정렬 계약 위반 "
            f"nodes={len(gt_nodes)} deriv={len(deriv)} — classify_derivability 가 "
            "iter_nodes 와 1:1 이 아니면 색이 엉뚱한 요소에 칠해진다."
        )
    gt_elements = _pack(gt_nodes, [d["element"] for d in deriv])
    for item, d in zip(gt_elements, deriv, strict=True):
        item["label"] = d["derivability"]
        item["derivable"] = bool(d["derivable"])  # 분류기의 is_derivable 결과 그대로
        r = d["reason"]
        reason = {"rule": r.get("rule", ""), "sim": r.get("text_sim", 0.0)}
        if r.get("matched_current"):
            mc = r["matched_current"]
            reason["matched"] = {
                "tag": mc["tag"],
                "own": mc["own_text"],
                # index 계열은 bounds 가 없어 근거 요소를 화면에서 짚을 수 없다 —
                # index 가 그 자리를 대신한다.
                "bounds": mc.get("bounds", ""),
                "index": str(mc.get("index") or ""),
                "seq": mc["seq_idx"],
            }
        # 라벨은 NON_DERIVABLE 이지만 뜻은 "판정 못 함" 이다. 2분류/색/필터가
        # 이것을 진짜 유도 불가능과 섞지 않도록 플래그를 실어 보낸다.
        if reason["rule"] == UNDECIDABLE_RULE:
            item["undecidable"] = True
        for k in ("payload", "coordinate"):
            if k in r:
                reason[k] = r[k]
        item["reason"] = reason

    cur_nodes = iter_nodes(parse_soup(cur_xml))
    cur_elements = _pack(cur_nodes, build_element_records(cur_nodes))

    counts = summarize_derivability(deriv)
    counts_own = {lbl: 0 for lbl in DERIVABILITY_LABELS}
    undec = undec_own = 0
    for d in deriv:
        if d["element"]["own_text"]:
            counts_own[d["derivability"]] += 1
        if d["reason"].get("rule") == UNDECIDABLE_RULE:
            undec += 1
            if d["element"]["own_text"]:
                undec_own += 1

    return {
        "row": row_idx,
        "action": action,
        "action_type": action_type(action),
        "action_raw": action_raw,
        "current": {
            "xml": _display_xml(cur_xml),
            "elements": cur_elements,
            "screen": _screen(cur_elements),
        },
        "gt": {
            "xml": _display_xml(gt_xml),
            "elements": gt_elements,
            "screen": _screen(gt_elements),
        },
        "counts": counts,
        "counts_own": counts_own,
        # NON_DERIVABLE 의 부분집합이다 (합계에 이중으로 더하지 마라).
        "undecidable": undec,
        "undecidable_own": undec_own,
    }


def _display_xml(xml: str) -> str:
    """원문 뷰용 정리. `<image>` 는 프롬프트의 이미지 placeholder 라 UI 요소가 아닌데
    원문에 남으면 감사자가 화면 요소로 오인한다. **표시에서만** 지운다 —
    분류기에는 원문 그대로 넘긴다 (정본 채점 경로와 같은 입력을 유지하기 위해).
    """
    return "\n".join(ln for ln in xml.split("\n") if ln.strip() != "<image>").strip()


# ── 사이트 조립 ──────────────────────────────────────────────────────────


def read_rows(path: Path, samples: int, seed: int) -> list[tuple[int, dict]]:
    """행을 문자열로만 읽고 뽑힌 행만 파싱한다 (test jsonl 은 수십 MB)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    idxs = [i for i, ln in enumerate(lines) if ln.strip()]
    if not idxs:
        raise BuildError(f"[derivability_site] {path}: 비어 있다")
    n = min(samples, len(idxs))
    chosen = sorted(random.Random(seed).sample(idxs, n))
    return [(i, json.loads(lines[i])) for i in chosen]


def build(test: Path, out: Path, samples: int, seed: int) -> dict:
    rows = read_rows(test, samples, seed)
    built = [build_sample(i, rec) for i, rec in rows]

    totals = {lbl: 0 for lbl in DERIVABILITY_LABELS}
    totals_own = {lbl: 0 for lbl in DERIVABILITY_LABELS}
    undec = undec_own = 0
    for s in built:
        for lbl in DERIVABILITY_LABELS:
            totals[lbl] += s["counts"][lbl]
            totals_own[lbl] += s["counts_own"][lbl]
        undec += s["undecidable"]
        undec_own += s["undecidable_own"]

    data = {
        "title": test.parent.name + " · " + test.stem,
        "test": str(test),
        "seed": seed,
        "samples": len(built),
        "labels": list(DERIVABILITY_LABELS),
        # 유도 가능 라벨 목록도 분류기에서 온다 — 범례가 지표와 갈리지 않게.
        "derivable_labels": sorted(DERIVABLE_LABELS),
        "totals": totals,
        "totals_own": totals_own,
        # NON_DERIVABLE 의 부분집합 — 표에서 별도 행으로만 보여준다.
        "undecidable": undec,
        "undecidable_own": undec_own,
        "undecidable_rule": UNDECIDABLE_RULE,
        "rows": built,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(data), encoding="utf-8")
    return data


def render_html(data: dict) -> str:
    # `</` 이스케이프는 필수다 — payload 안의 `</div>` 가 <script> 를 조기 종료시킨다.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{data["title"]} · 유도성 분류 감사</title>
<style>{SITE_CSS}{_label_css()}</style>
</head>
<body>
<header class="topbar"><div class="topbar-inner">
  <div class="brand"><h1>유도성 분류 감사 · {data["title"]}</h1>
    <p>GT next-state 요소를 <b>현재 state + action 으로 유도 가능한가</b> 로 색칠한다 ·
       판정 정본: <code>scripts/diff_loss/hungarian_diff_v3.py</code></p></div>
  <div class="toolbar">
    <div class="seg" id="modeSeg">
      <button data-mode="2" class="active">2분류</button>
      <button data-mode="7">라벨 7색</button>
    </div>
    <button class="btn" id="prevBtn" title="이전 샘플 (&larr;)">&larr;</button>
    <select id="sampleSelect" aria-label="샘플 선택"></select>
    <button class="btn" id="nextBtn" title="다음 샘플 (&rarr;)">&rarr;</button>
  </div>
</div></header>
<main>
  <section class="hero">
    <div class="panel intro">
      <h2>유도 가능한 요소와 그렇지 않은 요소</h2>
      <p>왼쪽이 현재 화면, 오른쪽이 GT 다음 화면이다. 오른쪽 요소를 클릭하면
         <b>어떤 규칙으로 그 라벨이 붙었는지</b>와 근거가 된 현재 요소(왼쪽에 파란 테두리로
         함께 표시)를 볼 수 있다. 근거 없이 색만 보면 오분류를 판별할 수 없다.</p>
      <div class="seed">
        <span class="pill">데이터 <strong>{data["test"]}</strong></span>
        <span class="pill">표본 <strong>{data["samples"]}</strong></span>
        <span class="pill">시드 <strong>{data["seed"]}</strong></span>
        <span class="pill">키보드 <span class="kbd">&larr;</span> <span class="kbd">&rarr;</span></span>
      </div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="panel counts">
      <h3>라벨 분포</h3>
      <p class="hint">전체 요소 / 자체 텍스트(own_text)가 있는 요소만.
         후자가 실제로 "콘텐츠를 유도할 수 있나" 다 — 전체에는 STRUCTURE(컨테이너)가
         대부분이라 비율이 낙관적으로 보인다.</p>
      <div id="countTable"></div>
    </div>
  </section>
  <div class="navrow">
    <div class="sample-id" id="sampleId"></div>
    <div class="filters">
      <span>필터</span>
      <select id="filterSelect"></select>
      <label class="chk"><input type="checkbox" id="hideStructure" checked /> STRUCTURE 숨기기</label>
      <label class="chk"><input type="checkbox" id="onlyOwn" /> own_text 있는 것만</label>
    </div>
    <div class="progress"><div id="progressBar"></div></div>
  </div>
  <section class="panel actionbar">
    <strong>액션</strong><span class="action-chip" id="actionChip"></span>
    <span class="hint" id="actionHint"></span>
  </section>
  <section class="grid3">
    <div class="cell">
      <div class="section-title"><span>Current UI State</span>
        <span class="tools"><button class="mini active" data-view="wire" data-pane="A">와이어프레임</button><button class="mini" data-view="raw" data-pane="A">원문</button></span></div>
      <div class="view" id="paneA"></div>
    </div>
    <div class="cell">
      <div class="section-title"><span>GT Next State · 유도성 색칠</span>
        <span class="tools"><button class="mini active" data-view="wire" data-pane="B">와이어프레임</button><button class="mini" data-view="raw" data-pane="B">원문</button></span></div>
      <div class="view" id="paneB"></div>
    </div>
    <div class="cell">
      <div class="section-title"><span>요소 목록 · 판정 근거</span><span class="hint" id="listCount"></span></div>
      <div class="view list" id="elemList"></div>
    </div>
  </section>
  <section class="panel detail" id="detail"><p class="hint">요소를 클릭하면 판정 근거가 여기 표시된다.</p></section>
</main>
<script>
const DATA = {payload};
{SITE_JS}
</script>
</body>
</html>
"""


def _label_css() -> str:
    """라벨 색 CSS 를 LABEL_COLOR 에서 생성. 색 정의를 한 곳에 묶는다."""
    out = []
    for lbl, color in LABEL_COLOR.items():
        out.append(
            f".l7-{lbl}{{border-color:{color};background:{color}22}}"
            f".chip-{lbl}{{background:{color};color:#fff}}"
        )
    for k, color in D2_COLOR.items():
        out.append(f".d2-{k}{{border-color:{color};background:{color}22}}")
    return "\n".join(out)


SITE_CSS = """
:root{--bg:#f4f6f9;--surface:#fff;--surface-2:#f8fafc;--ink:#172033;--muted:#667085;
 --line:#d9e0ea;--accent:#315efb;--code:#111827;--code-ink:#e5e7eb;
 --shadow:0 8px 28px rgba(15,23,42,.07);--radius:14px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
 font-family:Inter,ui-sans-serif,system-ui,"Noto Sans KR",Segoe UI,sans-serif}
button,select,input{font:inherit}code{background:#eef2f8;padding:1px 5px;border-radius:5px;font-size:11px}
.topbar{position:sticky;top:0;z-index:50;background:rgba(244,246,249,.94);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
.topbar-inner{width:min(2360px,calc(100vw - 24px));margin:auto;padding:12px 18px;display:flex;gap:16px;align-items:center;justify-content:space-between}
.brand h1{font-size:17px;margin:0}.brand p{font-size:12px;color:var(--muted);margin:3px 0 0}
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.btn,.seg button,select{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:9px;padding:7px 10px;cursor:pointer}
.btn:hover,.seg button:hover{border-color:#aeb9c8}.seg{display:flex;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.seg button{border:0;border-radius:0;border-right:1px solid var(--line)}.seg button:last-child{border-right:0}
.seg button.active{background:var(--accent);color:#fff}
.kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:1px 5px;background:#fff;color:var(--muted)}
main{width:min(2360px,calc(100vw - 24px));margin:auto;padding:18px 0}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:14px;margin-bottom:14px}
.intro{padding:18px}.intro h2{margin:0 0 7px;font-size:21px}.intro p{margin:0;color:var(--muted);line-height:1.6;font-size:13px}
.seed{display:flex;gap:9px;align-items:center;margin-top:12px;flex-wrap:wrap}
.pill{display:inline-flex;gap:6px;align-items:center;padding:4px 9px;border-radius:999px;background:var(--surface-2);border:1px solid var(--line);font-size:11px;color:var(--muted)}
.pill strong{color:var(--ink);font-weight:700}
.legend{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
.legend b{font-size:10px;font-weight:800;border-radius:999px;padding:3px 9px}
.counts{padding:14px;overflow:auto}.counts h3{font-size:13px;margin:0 0 6px;color:var(--muted)}
.hint{font-size:11px;color:var(--muted);line-height:1.5;margin:0 0 8px}
table.cnt{width:100%;border-collapse:collapse;font-size:12px}
table.cnt th,table.cnt td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
table.cnt th:first-child,table.cnt td:first-child{text-align:left}
table.cnt tr.sum td{font-weight:800;border-top:2px solid var(--line)}
.navrow{display:flex;align-items:center;gap:14px;margin:12px 0;flex-wrap:wrap}
.sample-id{font-weight:750;font-size:13px}.sample-id small{font-weight:400;color:var(--muted);margin-left:7px}
.filters{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--muted)}
.chk{display:inline-flex;gap:5px;align-items:center;cursor:pointer}
.progress{height:5px;background:#e6eaf0;border-radius:99px;overflow:hidden;flex:1;min-width:120px}
.progress>div{height:100%;background:var(--accent);transition:width .2s}
.actionbar{padding:11px 14px;display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.actionbar strong{font-size:12px}
.action-chip{font:12px ui-monospace,Menlo,monospace;background:#1f2937;color:#fff;border-radius:8px;padding:6px 10px;overflow-wrap:anywhere}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;align-items:start}
.cell{min-width:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.section-title{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line);font-size:12px;font-weight:750}
.section-title .hint{margin:0}
.tools{display:flex;gap:4px}.mini{font-size:11px;padding:4px 7px;border:1px solid var(--line);border-radius:7px;background:#fff;cursor:pointer}
.mini.active{background:#1f2937;color:#fff;border-color:#1f2937}
.view{height:620px;overflow:auto;background:#202938;position:relative}
.view.list{background:var(--surface)}
pre.raw{margin:0;padding:11px;font:10.5px/1.5 ui-monospace,Menlo,monospace;color:var(--code-ink);white-space:pre;min-height:100%}
.wireframe{min-height:100%;padding:10px;display:flex;align-items:flex-start;justify-content:center}
.phone{position:relative;height:600px;width:auto;max-width:100%;background:#f8fafc;border:4px solid #0b1220;border-radius:16px;overflow:hidden}
.node{position:absolute;border:1px solid rgba(100,116,139,.35);background:rgba(148,163,184,.06);overflow:hidden;color:#172033;font:7px/1.1 ui-monospace,monospace;padding:1px;cursor:pointer}
.node-label{pointer-events:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.node.sel{outline:2px solid #111827;outline-offset:-2px;z-index:20}
.node.matched{border:2px dashed #2563eb;background:rgba(37,99,235,.18);z-index:15}
.tap{position:absolute;border-radius:50%;border:2px solid #f59e0b;background:rgba(245,158,11,.35);z-index:30;pointer-events:none}
.tree{min-height:100%;padding:9px;background:#f8fafc}
.tnode{min-height:20px;margin-top:2px;padding:3px 6px;border:1px solid rgba(100,116,139,.35);border-radius:5px;background:rgba(148,163,184,.08);
 color:#172033;font:10px/1.3 ui-monospace,monospace;display:flex;gap:7px;align-items:center;overflow:hidden;cursor:pointer}
.tnode strong{flex:none;color:#475569;font-weight:700}.tnode span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tnode.sel{outline:2px solid #111827}.tnode.matched{border:2px dashed #2563eb;background:rgba(37,99,235,.14)}
.tree-note{position:sticky;top:0;z-index:2;margin:-9px -9px 8px;padding:6px 9px;background:#e9efff;color:#315efb;border-bottom:1px solid #9bb0ff;font-size:10px;font-weight:750}
.row{display:flex;gap:7px;align-items:baseline;padding:6px 9px;border-bottom:1px solid var(--line);cursor:pointer;font-size:11.5px}
.row:hover{background:var(--surface-2)}.row.sel{background:#fff7ed;box-shadow:inset 3px 0 0 #111827}
.row .chip{font-size:9px;font-weight:800;border-radius:999px;padding:2px 6px;white-space:nowrap;flex:none}
.row .tg{color:var(--muted);font:10px ui-monospace,monospace;flex:none}
.row .tx{overflow-wrap:anywhere;flex:1;min-width:0}
.row .rl{color:var(--muted);font:9.5px ui-monospace,monospace;flex:none;white-space:nowrap}
.detail{margin-top:14px;padding:14px}
.detail h3{margin:0 0 9px;font-size:14px;display:flex;gap:8px;align-items:center}
table.kv{width:100%;border-collapse:collapse;font:11.5px/1.55 ui-monospace,Menlo,monospace}
table.kv td{padding:5px 8px;border-top:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}
table.kv td:first-child{color:var(--muted);width:170px}
.empty{padding:18px;color:var(--muted);font-size:12px}
@media(max-width:1400px){.grid3{grid-template-columns:1fr 1fr}.hero{grid-template-columns:1fr}}
"""


SITE_JS = r"""
const LABELS = DATA.labels;
const st = {pos:0, mode:'2', filter:'ALL', hideStructure:true, onlyOwn:false,
            sel:null, view:{A:'wire', B:'wire'}};
const $ = id => document.getElementById(id);
const row = () => DATA.rows[st.pos];

/* 색 클래스. 2분류 모드는 분류기가 준 el.derivable 을 그대로 쓴다 — 라벨 집합을
   JS 에 복제하면 그 순간 화면과 지표가 갈린다.
   `el.undecidable` 은 라벨이 NON_DERIVABLE 이면서 뜻이 "판정 못 함" 인 요소다.
   유도 불가능과 같은 색으로 칠하면 오분류 사냥의 모집단이 오염되므로 셋째 색으로 뺀다. */
function cls(el){
  if(el.undecidable) return st.mode === '2' ? 'd2-undec' : 'l7-UNDECIDABLE';
  return st.mode === '2' ? (el.derivable ? 'd2-yes' : 'd2-no') : 'l7-' + el.label;
}
function chipCls(el){ return 'chip-' + (el.undecidable ? 'UNDECIDABLE' : el.label); }
function visible(el){
  if(st.hideStructure && el.label === 'STRUCTURE') return false;
  if(st.onlyOwn && !el.own) return false;
  if(st.filter === 'ALL') return true;
  if(st.filter === '__ND') return !el.derivable && !el.undecidable;
  if(st.filter === '__UNDEC') return !!el.undecidable;
  if(st.filter === '__D') return el.derivable;
  return el.label === st.filter;
}
function labelOf(el){ return el.own || el.absorbed || ''; }

/* ── 헤더 ───────────────────────────────────────────────────────── */
function renderLegend(){
  const box = $('legend'); box.replaceChildren();
  const items = st.mode === '2'
    ? [['유도 가능','d2-yes'], ['유도 불가능 (NON_DERIVABLE)','d2-no'],
       ['판정 불가 (' + DATA.undecidable_rule + ')','d2-undec']]
    : LABELS.map(l => [l, 'l7-' + l]).concat([['UNDECIDABLE','l7-UNDECIDABLE']]);
  for(const [text, c] of items){
    const b = document.createElement('b');
    b.className = st.mode === '2' ? c : 'chip-' + text;
    if(st.mode === '2'){ b.style.border = '1px solid'; }
    b.textContent = text; box.appendChild(b);
  }
}
function countTable(){
  const cur = row();
  const t = document.createElement('table'); t.className = 'cnt';
  const head = document.createElement('tr');
  for(const h of ['라벨','이 샘플','이 샘플(own)','전체','전체(own)']){
    const th = document.createElement('th'); th.textContent = h; head.appendChild(th);
  }
  t.appendChild(head);
  const sum = k => LABELS.reduce((a,l) => a + k[l], 0);
  const tot = {a:sum(cur.counts), b:sum(cur.counts_own),
               c:sum(DATA.totals), d:sum(DATA.totals_own)};
  for(const l of LABELS){
    const tr = document.createElement('tr');
    const c0 = document.createElement('td');
    const chip = document.createElement('b');
    chip.className = 'chip-' + l; chip.style.fontSize = '9px';
    chip.style.borderRadius = '999px'; chip.style.padding = '2px 7px';
    chip.textContent = l; c0.appendChild(chip); tr.appendChild(c0);
    const vals = [[cur.counts[l], tot.a], [cur.counts_own[l], tot.b],
                  [DATA.totals[l], tot.c], [DATA.totals_own[l], tot.d]];
    for(const [v, d] of vals){
      const td = document.createElement('td');
      td.textContent = d ? `${v} (${(100*v/d).toFixed(1)}%)` : `${v}`;
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
  // 판정 불가 — NON_DERIVABLE 의 **부분집합**이라 합계에 이중으로 더하지 않는다.
  const ud = document.createElement('tr');
  const c1 = document.createElement('td');
  const uchip = document.createElement('b');
  uchip.className = 'chip-UNDECIDABLE'; uchip.style.fontSize = '9px';
  uchip.style.borderRadius = '999px'; uchip.style.padding = '2px 7px';
  uchip.textContent = '└ 판정 불가'; c1.appendChild(uchip); ud.appendChild(c1);
  for(const v of [cur.undecidable, cur.undecidable_own,
                  DATA.undecidable, DATA.undecidable_own]){
    const td = document.createElement('td');
    td.textContent = String(v); td.style.color = '#78716c'; ud.appendChild(td);
  }
  t.appendChild(ud);
  const tr = document.createElement('tr'); tr.className = 'sum';
  for(const v of ['합계', tot.a, tot.b, tot.c, tot.d]){
    const td = document.createElement('td'); td.textContent = v; tr.appendChild(td);
  }
  t.appendChild(tr);
  $('countTable').replaceChildren(t);
}

/* ── 와이어프레임 / 트리 ────────────────────────────────────────── */
function wire(pane, colored){
  const screen = pane.screen;
  if(!screen) return tree(pane, colored);
  const shell = document.createElement('div'); shell.className = 'wireframe';
  const phone = document.createElement('div'); phone.className = 'phone';
  phone.style.aspectRatio = `${screen[0]} / ${screen[1]}`;
  const els = pane.elements.filter(e => e.bounds).slice();
  els.sort((a,b) => area(b) - area(a));
  let shown = 0;
  for(const e of els){
    if(colored && !visible(e)) continue;
    const [x1,y1,x2,y2] = e.bounds;
    const d = document.createElement('div');
    d.className = 'node' + (colored ? ' ' + cls(e) : '');
    if(colored && st.sel === e.seq) d.className += ' sel';
    if(!colored && matchedSeq() === e.seq) d.className += ' matched';
    d.style.left = `${x1/screen[0]*100}%`; d.style.top = `${y1/screen[1]*100}%`;
    d.style.width = `${(x2-x1)/screen[0]*100}%`;
    d.style.height = `${(y2-y1)/screen[1]*100}%`;
    d.title = `#${e.seq} <${e.tag}> ${labelOf(e)}` + (colored ? ` — ${e.label} (${e.reason.rule})` : '');
    const txt = labelOf(e);
    if(txt && ((x2-x1) > screen[0]*.12 && (y2-y1) > screen[1]*.014)){
      const l = document.createElement('div'); l.className = 'node-label'; l.textContent = txt;
      d.appendChild(l);
    }
    if(colored) d.addEventListener('click', ev => { ev.stopPropagation(); select(e.seq); });
    phone.appendChild(d); shown++;
  }
  const tap = row().action && row().action.coordinate;
  if(tap && tap.length >= 2){
    const r = Math.max(screen[0], screen[1]) * 0.014;
    const m = document.createElement('div'); m.className = 'tap';
    m.style.left = `${(tap[0]-r)/screen[0]*100}%`; m.style.top = `${(tap[1]-r)/screen[1]*100}%`;
    m.style.width = `${2*r/screen[0]*100}%`; m.style.height = `${2*r/screen[1]*100}%`;
    phone.appendChild(m);
  }
  if(!shown){
    const e = document.createElement('div'); e.className = 'empty';
    e.textContent = '필터에 걸리는 요소가 없습니다.'; shell.appendChild(e); return shell;
  }
  shell.appendChild(phone); return shell;
}
function area(e){ return (e.bounds[2]-e.bounds[0]) * (e.bounds[3]-e.bounds[1]); }

/* bounds 가 없는 계열(EXP01~04)은 위치를 그릴 수 없다 — 트리 들여쓰기로 그린다. */
function tree(pane, colored){
  const shell = document.createElement('div'); shell.className = 'tree';
  const note = document.createElement('div'); note.className = 'tree-note';
  note.textContent = 'bounds 없음 · XML 계층 구조 기반 트리 뷰';
  shell.appendChild(note);
  let shown = 0;
  for(const e of pane.elements){
    if(colored && !visible(e)) continue;
    if(shown >= 700) break;
    const d = document.createElement('div');
    d.className = 'tnode' + (colored ? ' ' + cls(e) : '');
    if(colored && st.sel === e.seq) d.className += ' sel';
    if(!colored && matchedSeq() === e.seq) d.className += ' matched';
    d.style.marginLeft = `${Math.min(16, e.depth) * 14}px`;
    const nm = document.createElement('strong');
    nm.textContent = `<${e.tag}${e.index !== undefined ? ' #' + e.index : ''}>`;
    d.appendChild(nm);
    const txt = labelOf(e);
    if(txt){ const s = document.createElement('span'); s.textContent = txt; d.appendChild(s); }
    d.title = `#${e.seq}` + (colored ? ` — ${e.label} (${e.reason.rule})` : '');
    if(colored) d.addEventListener('click', ev => { ev.stopPropagation(); select(e.seq); });
    shell.appendChild(d); shown++;
  }
  if(!shown){
    const e = document.createElement('div'); e.className = 'empty';
    e.textContent = '필터에 걸리는 요소가 없습니다.'; shell.appendChild(e);
  }
  return shell;
}
function rawView(xml){
  const p = document.createElement('pre'); p.className = 'raw'; p.textContent = xml; return p;
}
function matchedSeq(){
  if(st.sel === null) return -1;
  const e = row().gt.elements.find(x => x.seq === st.sel);
  return e && e.reason.matched ? e.reason.matched.seq : -1;
}

/* ── 요소 목록 ──────────────────────────────────────────────────── */
function renderList(){
  const box = $('elemList'); box.replaceChildren();
  const els = row().gt.elements.filter(visible);
  $('listCount').textContent = `${els.length} / ${row().gt.elements.length} 요소`;
  if(!els.length){
    const e = document.createElement('div'); e.className = 'empty';
    e.textContent = '필터에 걸리는 요소가 없습니다.'; box.appendChild(e); return;
  }
  for(const e of els.slice(0, 900)){
    const d = document.createElement('div');
    d.className = 'row' + (st.sel === e.seq ? ' sel' : '');
    const chip = document.createElement('span');
    chip.className = 'chip ' + chipCls(e);
    chip.textContent = e.undecidable ? '판정불가'
      : (st.mode === '2' ? (e.derivable ? '가능' : '불가') : e.label);
    d.appendChild(chip);
    const tg = document.createElement('span'); tg.className = 'tg';
    tg.textContent = `#${e.seq} <${e.tag}>`; d.appendChild(tg);
    const tx = document.createElement('span'); tx.className = 'tx';
    tx.textContent = labelOf(e) || '(텍스트 없음)'; d.appendChild(tx);
    const rl = document.createElement('span'); rl.className = 'rl';
    rl.textContent = e.reason.rule; d.appendChild(rl);
    d.addEventListener('click', () => select(e.seq));
    box.appendChild(d);
  }
}

/* ── 판정 근거 ──────────────────────────────────────────────────── */
function renderDetail(){
  const box = $('detail'); box.replaceChildren();
  const e = st.sel === null ? null : row().gt.elements.find(x => x.seq === st.sel);
  if(!e){
    const p = document.createElement('p'); p.className = 'hint';
    p.textContent = '요소를 클릭하면 판정 근거가 여기 표시된다.';
    box.appendChild(p); return;
  }
  const h = document.createElement('h3');
  const chip = document.createElement('b');
  chip.className = chipCls(e);
  chip.style.cssText = 'font-size:11px;border-radius:999px;padding:3px 10px';
  chip.textContent = e.undecidable
    ? e.label + ' · 판정 불가 (유도 불가능이 아니다)'
    : e.label + (e.derivable ? ' · 유도 가능' : ' · 유도 불가능');
  h.appendChild(chip);
  const t2 = document.createElement('span');
  t2.style.cssText = 'font:12px ui-monospace,monospace;color:#667085';
  t2.textContent = `#${e.seq} <${e.tag}>`;
  h.appendChild(t2); box.appendChild(h);

  const rows = [
    ['판정 규칙 (reason.rule)', e.reason.rule],
    ['own_text', e.own || '(없음)'],
  ];
  if(e.absorbed) rows.push(['흡수 텍스트 (판정에는 미사용)', e.absorbed]);
  rows.push(['bounds', e.bounds ? `[${e.bounds[0]},${e.bounds[1]}][${e.bounds[2]},${e.bounds[3]}]`
                                : (e.bounds_raw || '(없음)')]);
  if(e.index !== undefined) rows.push(['index 속성', e.index]);
  rows.push(['slot (분류기 slot_key)', e.slot || '(없음)']);
  const m = e.reason.matched;
  rows.push(['근거 current 요소 (reason.matched_current)',
             m ? `#${m.seq} <${m.tag}> ${m.bounds || (m.index ? 'index=' + m.index : '(자리 정보 없음)')}`
                 + ` — ${m.own || '(텍스트 없음)'}`
               : '(없음)']);
  rows.push(['text_sim', String(e.reason.sim)]);
  if(e.reason.payload !== undefined) rows.push(['action payload', e.reason.payload]);
  if(e.reason.coordinate !== undefined) rows.push(['action coordinate', JSON.stringify(e.reason.coordinate)]);
  rows.push(['action (전체)', JSON.stringify(row().action)]);
  if(e.undecidable){
    rows.push(['⚠ 판정 불가',
      'action 이 좌표를 줬는데 이 요소에 bounds 가 없어 ACTION_TARGET 포함 판정을 '
      + '할 수 없었다. 라벨은 NON_DERIVABLE 이지만 "유도 불가능" 이라는 뜻이 아니다 — '
      + '유도 불가능 통계에 이것을 섞어 읽지 마라.']);
  }
  const t = document.createElement('table'); t.className = 'kv';
  for(const [k, v] of rows){
    const tr = document.createElement('tr');
    const a = document.createElement('td'); a.textContent = k;
    const b = document.createElement('td'); b.textContent = v;
    tr.appendChild(a); tr.appendChild(b); t.appendChild(tr);
  }
  box.appendChild(t);
}

/* ── 렌더 / 이벤트 ──────────────────────────────────────────────── */
function select(seq){ st.sel = (st.sel === seq ? null : seq); render(); }
function renderPanes(){
  const r = row();
  $('paneA').replaceChildren(st.view.A === 'raw' ? rawView(r.current.xml) : wire(r.current, false));
  $('paneB').replaceChildren(st.view.B === 'raw' ? rawView(r.gt.xml) : wire(r.gt, true));
}
function render(){
  const r = row();
  $('sampleId').innerHTML = '';
  const idText = document.createElement('span');
  idText.textContent = `표본 ${st.pos + 1} / ${DATA.rows.length}`;
  const small = document.createElement('small');
  small.textContent = `jsonl 행 ${r.row}`;
  $('sampleId').appendChild(idText); $('sampleId').appendChild(small);
  $('progressBar').style.width = `${(st.pos + 1) / DATA.rows.length * 100}%`;
  $('actionChip').textContent = r.action_raw.replace(/\s+/g, ' ');
  const hasCoord = r.action && r.action.coordinate;
  $('actionHint').textContent = hasCoord
    ? '주황 원 = 클릭 좌표 (ACTION_TARGET 판정의 입력)'
    : 'coordinate 없음 — ACTION_TARGET 은 이 계열에서 절대 발화하지 않는다';
  $('sampleSelect').value = String(st.pos);
  renderLegend(); countTable(); renderPanes(); renderList(); renderDetail();
}
function init(){
  const sel = $('sampleSelect');
  DATA.rows.forEach((r, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    const nd = r.counts.NON_DERIVABLE;
    o.textContent = `#${i + 1} · 행 ${r.row} · ND ${nd}`;
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => { st.pos = Number(sel.value); st.sel = null; render(); });
  const fs = $('filterSelect');
  const opts = [['ALL','전체'],
                ['__ND','유도 불가능만 (판정 불가 제외)'],
                ['__UNDEC','판정 불가만 (' + DATA.undecidable_rule + ')'],
                ['__D','유도 가능만']]
    .concat(LABELS.map(l => [l, l + ' 만 (라벨 그대로)']));
  for(const [v, t] of opts){
    const o = document.createElement('option'); o.value = v; o.textContent = t; fs.appendChild(o);
  }
  fs.addEventListener('change', () => { st.filter = fs.value; render(); });
  $('hideStructure').addEventListener('change', e => { st.hideStructure = e.target.checked; render(); });
  $('onlyOwn').addEventListener('change', e => { st.onlyOwn = e.target.checked; render(); });
  $('prevBtn').addEventListener('click', () => move(-1));
  $('nextBtn').addEventListener('click', () => move(1));
  for(const b of document.querySelectorAll('#modeSeg button')){
    b.addEventListener('click', () => {
      st.mode = b.dataset.mode;
      for(const o of document.querySelectorAll('#modeSeg button')) o.classList.toggle('active', o === b);
      render();
    });
  }
  for(const b of document.querySelectorAll('[data-view]')){
    b.addEventListener('click', () => {
      st.view[b.dataset.pane] = b.dataset.view;
      for(const o of document.querySelectorAll(`[data-pane="${b.dataset.pane}"]`))
        o.classList.toggle('active', o === b);
      render();
    });
  }
  document.addEventListener('keydown', ev => {
    if(ev.target.tagName === 'SELECT' || ev.target.tagName === 'INPUT') return;
    if(ev.key === 'ArrowLeft') move(-1);
    if(ev.key === 'ArrowRight') move(1);
  });
  render();
}
function move(d){
  st.pos = (st.pos + d + DATA.rows.length) % DATA.rows.length;
  st.sel = null; render();
}
init();
"""


# ── CLI ──────────────────────────────────────────────────────────────────


def _print_distribution(data: dict) -> None:
    """stdout 분포표 — 전체 / own_text 있는 요소만 두 열.

    전체 열은 STRUCTURE(자체 텍스트 없는 컨테이너)가 대부분이라 "유도 가능 90%" 처럼
    낙관적으로 보인다. 실제로 콘텐츠 유도성이 걸린 것은 own 열이다.
    """
    tot = sum(data["totals"].values())
    tot_own = sum(data["totals_own"].values())
    print(f"\n{data['test']}  (표본 {data['samples']}, 시드 {data['seed']})")
    print(f"{'label':<16}{'all':>10}{'':>10}{'own_text':>12}{'':>10}")
    for lbl in data["labels"]:
        a, b = data["totals"][lbl], data["totals_own"][lbl]
        pa = f"{100 * a / tot:.1f}%" if tot else "-"
        pb = f"{100 * b / tot_own:.1f}%" if tot_own else "-"
        print(f"{lbl:<16}{a:>10}{pa:>10}{b:>12}{pb:>10}")
    if data["undecidable"]:
        # NON_DERIVABLE 안에 섞여 있는 "판정 못 함" — 유도 불가능으로 읽으면 안 된다.
        print(
            f"{'└ 판정 불가':<15}{data['undecidable']:>10}{'':>10}"
            f"{data['undecidable_own']:>12}{'':>10}   ({data['undecidable_rule']})"
        )
    der = sum(data["totals"][x] for x in data["derivable_labels"])
    der_own = sum(data["totals_own"][x] for x in data["derivable_labels"])
    print(f"{'-' * 48}")
    print(
        f"{'유도 가능':<16}{der:>10}{f'{100 * der / tot:.1f}%' if tot else '-':>10}"
        f"{der_own:>12}{f'{100 * der_own / tot_own:.1f}%' if tot_own else '-':>10}"
    )
    print(f"{'합계':<16}{tot:>10}{'':>10}{tot_own:>12}{'':>10}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="유도성 분류를 실제 데이터에서 눈으로 감사하는 사이트 빌더"
    )
    ap.add_argument("--test", required=True, type=Path, help="stage1 test jsonl")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path, help="출력 index.html 경로")
    args = ap.parse_args(argv)

    if not args.test.is_file():
        raise BuildError(f"[derivability_site] 입력 없음: {args.test}")
    data = build(args.test, args.out, args.samples, args.seed)
    _print_distribution(data)
    size = args.out.stat().st_size / 1024
    print(f"\n→ {args.out}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
