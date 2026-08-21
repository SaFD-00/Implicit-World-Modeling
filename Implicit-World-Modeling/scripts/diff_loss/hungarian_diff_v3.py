"""
hungarian_diff_v3.py
────────────────────────────────────────────────────────────────
v2 의 `classify_diff` (ADDED / MODIFIED / UNCHANGED) 에 **두 번째 축**을 더한다:

    classify_derivability(current_html, future_html, action) -> list[dict]

축의 정의 — **액션 유도성**: *현재 state + action 만으로 그 요소의 내용이 결정되는가.*

왜 필요한가
───────────
GT next-state 에는 모델이 원리적으로 알 수 없는 콘텐츠가 섞여 있다 — 서버가 내려준
검색 결과, 상품명, 가격, 리뷰어 이름, 도시명. 그것을 못 맞힌 것과 "버튼을 눌렀으니
이 패널이 열린다" 를 못 맞힌 것이 한 숫자에 섞이면 지표가 모델 능력을 말하지 않는다.
학습에서도 유도 불가능한 콘텐츠를 full weight 로 외우게 하면 **hallucination 을
학습시킨다.** 그래서 채점과 가중치 양쪽에서 이 축이 필요하다.

라벨 (앞의 6개 = 유도 가능, 마지막 1개 = 유도 불가능)
───────────────────────────────────────────────────
  COPY            현재 화면의 **같은 자리**에 같은 텍스트가 이미 있다
  REFLOW          현재 화면 **어딘가**에 같은 텍스트가 있다 (스크롤/재배치)
  ACTION_PAYLOAD  action 이 준 문자열(type 의 text, open 의 app_name)이 내용에 들어있다
  ACTION_TARGET   action 좌표가 이 요소 안이다 (누른 대상의 상태 변화)
  STRUCTURE       자체 텍스트가 없다 (컨테이너·아이콘) — 예측할 콘텐츠가 없다
  SYSTEM_UI       IME 키보드/키패드 패널 — 앱이 아니라 OS 가 그리는 것이라 결정적이다
  NON_DERIVABLE   위 어디에도 안 걸림 = 새 콘텐츠

설계상 배운 것 (프로토타입 5회 반복에서 확정 — 되돌리지 마라)
──────────────────────────────────────────────────────────
1. **컨테이너를 "자손 텍스트 흡수" 로 판정하면 안 된다.** 루트 div 가 화면 전체
   텍스트를 갖게 되어 NON_DERIVABLE 로 오분류된다. 그래서 이 모듈은 판정에
   `own_text` **만** 쓰고, 자체 텍스트가 없는 요소는 `STRUCTURE` 로 닫는다.
   콘텐츠 유도성은 실제로 텍스트를 가진 요소에서만 따진다.
2. **SYSTEM_UI 를 어휘 목록(q,w,e,…)으로 잡으면 안 된다** — 언어/레이아웃에 묶인다.
   **키캡 밀도**로 잡는다. 그리고 키 묶음의 LCA + y임계로 위로 확장하는 방식은
   실패한다: 다이얼패드 키가 앱 폼 안에 흩어져 있으면 LCA 가 루트로 올라가 연락처
   폼의 Cancel/More options 까지 SYSTEM_UI 로 삼킨다(실측 doc#1·#5). 밀도 조건이
   그 실패를 막는다.
3. **precision 우선**이 맞다. 앱 콘텐츠를 SYSTEM_UI(유도 가능) 로 잘못 넣으면 모델의
   실패가 지표에서 숨는다. 반대로 과소 라벨링(연락처 알파벳 인덱스가 NON_DERIVABLE
   로 남는 등) 은 뷰어에서 눈에 보이므로 회복 가능하다.

각 항목은 **판정 근거**(`reason`)를 함께 담는다 — 뷰어가 "왜 이 라벨인가" 를
보여줄 수 있어야 라벨을 사람이 검증할 수 있다.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hungarian_metric_v3 import (
    _hungarian_match,
    _text_sim,
    build_element_records,
    extract_elements,
    iter_nodes,
    parse_bounds,
    parse_soup,
)

UNCHANGED_COST_THRESHOLD = 0.05

# ── 유도성 라벨 ────────────────────────────────────────────────────────────
COPY = "COPY"
REFLOW = "REFLOW"
ACTION_PAYLOAD = "ACTION_PAYLOAD"
ACTION_TARGET = "ACTION_TARGET"
STRUCTURE = "STRUCTURE"
SYSTEM_UI = "SYSTEM_UI"
NON_DERIVABLE = "NON_DERIVABLE"

#: 판정 우선순위 순서 = 아래 `classify_derivability` 의 if 사슬 순서.
DERIVABILITY_LABELS: tuple[str, ...] = (
    SYSTEM_UI,
    STRUCTURE,
    COPY,
    REFLOW,
    ACTION_PAYLOAD,
    ACTION_TARGET,
    NON_DERIVABLE,
)

#: 유도 **가능** 라벨 집합. 채점기/빌더는 이 집합만 보고 2분할하면 된다.
DERIVABLE_LABELS: frozenset[str] = frozenset(
    {SYSTEM_UI, STRUCTURE, COPY, REFLOW, ACTION_PAYLOAD, ACTION_TARGET}
)


def is_derivable(label: str) -> bool:
    """라벨이 유도 가능 쪽인가. 문자열 비교를 호출부에 흩뿌리지 않기 위한 단일 진입점."""
    return label in DERIVABLE_LABELS


# ── SYSTEM_UI (IME/키패드) 판정 파라미터 ───────────────────────────────────
# 값의 근거는 EXP05 stage1 test id 200문서 실측이다. 완화하면 앱 콘텐츠를 삼키고,
# 조이면 키보드를 놓친다 — 놓치는 쪽이 안전하다(precision 우선, 위 3번).
KEYCAP_MAX_LEN = 4  # 키캡 라벨 최대 길이 (qwerty 1글자 ~ 다이얼패드 'ABC'/'GHI')
KEYCAP_MIN_COUNT = 15  # 이보다 적으면 키보드가 아니라 우연히 짧은 라벨들이다
KEYCAP_MIN_RATIO = 0.40  # 서브트리 leaf 중 키캡 비율. 앱 화면은 이 밑으로 떨어진다
KEYBOARD_MIN_TOP_RATIO = 0.35  # 패널 상단이 화면 상단부에 걸치면 키보드가 아니다
DEFAULT_SCREEN_H = 1876  # bounds 가 하나도 없을 때의 화면 높이 (EXP05 budget 기준)

_KEYCAP_RE = re.compile(r"[A-Za-z0-9+*#]+")
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S)
#: 태그 없는 포맷용 — 한 줄짜리 flat JSON 오브젝트. 중첩은 이 데이터에 없다.
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")

#: ACTION_PAYLOAD 후보 키 — action 이 문자열을 직접 준 것들.
#: `text` 는 type 액션, `app_name` 은 open 액션. 둘 다 "모델이 이미 알고 있는 문자열"이라
#: 다음 화면에 그대로 나타나는 것은 유도 가능하다.
PAYLOAD_KEYS = ("text", "app_name")


def extract_action(user_content: str) -> dict:
    """user 메시지에서 action JSON 을 파싱. 실패하면 빈 dict.

    **두 포맷을 모두 흡수해야 한다.** 실험군마다 프롬프트 규약이 다르다:

      EXP05·EXP07  `<action>{"action": "click", "coordinate": [x, y]}</action>`
      EXP01~04     `## Action\n{"action_type": "click", "index": "13"}`  (태그 없음)

    `<action>` 만 보면 EXP01~04 에서 **항상 빈 dict** 가 되고, 그러면 ACTION_PAYLOAD·
    ACTION_TARGET 이 영원히 안 나온다 — 라벨이 조용히 사라지는 형태라 분포를 나란히
    놓기 전엔 안 드러난다 (실측: EXP01 두 라벨 모두 0.0%).

    타입 키 이름도 갈린다(`action` vs `action_type`) — `_hungarian_eval._gt_action_type`
    의 `_TYPE_KEYS` 규약과 같은 문제이고, 여기서는 `action_type()` 이 그걸 흡수한다.
    """
    if not user_content:
        return {}
    m = _ACTION_RE.search(user_content)
    if m:
        blob = m.group(1)
    else:
        # 태그가 없는 포맷 — "Action" 헤더 뒤의 **마지막** JSON 오브젝트를 집는다.
        # 마지막인 이유: 프롬프트 앞부분의 action space 설명에도 `{...}` 예시가 있어서
        # 처음부터 찾으면 스키마 예시를 액션으로 오인한다.
        head = user_content.rfind("Action")
        if head < 0:
            return {}
        cand = _JSON_OBJ_RE.findall(user_content[head:])
        if not cand:
            return {}
        blob = cand[0]
    try:
        a = json.loads(blob)
    except (ValueError, TypeError):
        return {}
    return a if isinstance(a, dict) else {}


#: 액션 타입 키 — 실험군마다 다르다. `_hungarian_eval._gt_action_type._TYPE_KEYS` 와 같은 규약.
_TYPE_KEYS = ("action", "action_type", "type")


def action_type(action: dict) -> str:
    """액션 타입 문자열 (없으면 "")."""
    for k in _TYPE_KEYS:
        v = action.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _coerce_action(action: Any) -> dict:
    """dict / JSON 문자열 / None 을 모두 받아 dict 로 정규화."""
    if isinstance(action, dict):
        return action
    if isinstance(action, str):
        s = action.strip()
        if "<action>" in s:
            return extract_action(s)
        try:
            a = json.loads(s)
        except (ValueError, TypeError):
            return {}
        return a if isinstance(a, dict) else {}
    return {}


# ── diff 축 (v2 와 동일 로직, v3 element 집합 위에서) ──────────────────────


def classify_diff(current_html: str, future_html: str) -> list[dict]:
    """future 요소별 ADDED / MODIFIED / UNCHANGED.

    반환 항목의 `future_seq_idx` 는 `extract_elements(future_html)` 의 인덱스이고,
    v3 에서는 그것이 곧 문서 순서 노드 인덱스다 (화이트리스트가 없으므로).
    `classify_derivability` 의 반환도 같은 인덱스를 쓴다 — 두 축을 결합할 때
    이 정렬이 계약이다.
    """
    current_els = extract_elements(current_html)
    future_els = extract_elements(future_html)

    if not future_els:
        return []

    if not current_els:
        return [
            {
                "element": el,
                "future_seq_idx": i,
                "diff_type": "ADDED",
                "change_detail": {"text_sim": 0.0, "match_cost": -1.0},
            }
            for i, el in enumerate(future_els)
        ]

    pairs, _ = _hungarian_match(current_els, future_els)

    future_to_match: dict[int, tuple[int, float]] = {
        j: (i, cost) for i, j, cost in pairs
    }

    result: list[dict] = []
    for fut_seq_idx, fut_el in enumerate(future_els):
        if fut_seq_idx not in future_to_match:
            entry = {
                "element": fut_el,
                "future_seq_idx": fut_seq_idx,
                "diff_type": "ADDED",
                "change_detail": {"text_sim": 0.0, "match_cost": -1.0},
            }
        else:
            cur_idx, cost = future_to_match[fut_seq_idx]
            cur_el = current_els[cur_idx]
            text_sim = _text_sim(cur_el["text"], fut_el["text"])

            if cost <= UNCHANGED_COST_THRESHOLD:
                diff_type = "UNCHANGED"
            else:
                diff_type = "MODIFIED"

            entry = {
                "element": fut_el,
                "future_seq_idx": fut_seq_idx,
                "diff_type": diff_type,
                "change_detail": {
                    "text_sim": round(text_sim, 4),
                    "match_cost": round(cost, 5),
                },
            }

        result.append(entry)

    return result


def summarize_diff(diff_result: list[dict]) -> dict[str, int]:
    counts = {"ADDED": 0, "MODIFIED": 0, "UNCHANGED": 0}
    for d in diff_result:
        counts[d["diff_type"]] += 1
    return counts


# ── 유도성 축 ──────────────────────────────────────────────────────────────


def _is_keycap(node: Any) -> bool:
    """키캡 판정 — 어휘가 아니라 **형태**로 본다.

    qwerty(한 글자 aria-label)와 다이얼패드(ABC/DEF 같은 짧은 영숫자)를 함께 잡는다.
    어휘 목록(q,w,e,…)으로 잡으면 언어/키보드 레이아웃이 바뀌는 순간 무너진다.
    """
    lab = (node.get("aria-label") or "").strip()
    if not lab:
        lab = "".join(c for c in node.contents if isinstance(c, str)).strip()
    if not lab or len(lab) > KEYCAP_MAX_LEN:
        return False
    return _KEYCAP_RE.fullmatch(lab) is not None


def _screen_height(nodes: list) -> int:
    ys = []
    for n in nodes:
        b = parse_bounds(n.get("bounds", "") or "")
        if b:
            ys.append(b[3])
    return max(ys) if ys else DEFAULT_SCREEN_H


def _system_ui_node_ids(nodes: list) -> set[int]:
    """IME/키패드 패널 서브트리의 노드 id 집합.

    "키캡을 가장 많이 담으면서 leaf 의 KEYCAP_MIN_RATIO 이상이 키캡인 서브트리" 를
    고른다. 키보드 패널은 대부분이 키라 밀도가 높고 앱 화면은 낮다 — 이 밀도 조건이
    LCA 확장 방식이 앱 폼을 통째로 삼키던 실패를 막는다(모듈 docstring 2번).
    조건을 만족하는 노드가 없으면 시스템 UI 가 없는 화면으로 본다.

    leaf/keycap 집계는 문서 순서 **역순** 누적 한 번으로 끝낸다 (노드마다 find_all 을
    부르면 O(n²)).
    """
    if not nodes:
        return set()

    n_leaf: dict[int, int] = {}
    n_key: dict[int, int] = {}
    for node in reversed(nodes):
        children = [c for c in node.children if getattr(c, "name", None)]
        if not children:
            n_leaf[id(node)] = 1
            n_key[id(node)] = 1 if _is_keycap(node) else 0
        else:
            n_leaf[id(node)] = sum(n_leaf.get(id(c), 0) for c in children)
            n_key[id(node)] = sum(n_key.get(id(c), 0) for c in children)

    height = _screen_height(nodes)
    best = None
    best_n = 0
    for node in nodes:
        leaves = n_leaf[id(node)]
        keys = n_key[id(node)]
        if leaves < KEYCAP_MIN_COUNT or keys < KEYCAP_MIN_COUNT:
            continue
        if keys / leaves < KEYCAP_MIN_RATIO:
            continue
        b = parse_bounds(node.get("bounds", "") or "")
        if b is not None and b[1] < KEYBOARD_MIN_TOP_RATIO * height:
            continue  # 화면 상단까지 걸치면 키보드 패널이 아니다
        # 키를 더 많이 담는 쪽. 동률이면 문서 순서상 먼저인 쪽(=바깥쪽)이 남는다 —
        # 일부러 그렇게 둔다. 동률 시 안쪽을 고르도록 바꿔봤더니(2026-08-21 실측,
        # EXP05 200문서에서 SYSTEM_UI 1935→1560) 잘려 나간 375개가 전부 IME 크롬이었다:
        # "Open features menu"·"Sticker Keyboard"·"GIF Keyboard"·"Voice input"·
        # "Symbol keyboard" 와 툴바/하단 행 컨테이너. 키보드 패널은 키 격자만이 아니라
        # 툴바까지 한 덩어리이므로 바깥 컨테이너를 잡는 것이 맞다. 앱 UI 를 삼키는
        # 문제는 이 동률 규칙이 아니라 **bbox 상단 조건**이 막는다.
        if keys > best_n:
            best, best_n = node, keys

    if best is None:
        return set()
    return {id(best)} | {id(d) for d in best.find_all(True)}


def slot_key(el: dict) -> str:
    """요소의 "화면상 같은 자리" 키 — `bounds` 우선, 없으면 `index` 폴백.

    두 축을 섞어 쓰지 않도록 접두를 붙인다. 한 문서 안에서는 어느 한 축만
    존재하므로 실제로 섞이지는 않지만, 접두가 없으면 `"3"`(index)과 우연히 같은
    문자열의 bounds 가 충돌할 여지가 남는다.
    """
    b = el.get("bounds") or ""
    if b:
        return f"b:{b}"
    i = el.get("index") or ""
    return f"i:{i}" if i else ""


def _coordinate(action: dict) -> tuple[int, int] | None:
    """action 의 클릭 좌표. swipe 의 coordinate1/2 는 '누른 대상' 이 아니므로 제외."""
    c = action.get("coordinate")
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        try:
            return int(c[0]), int(c[1])
        except (TypeError, ValueError):
            return None
    return None


def classify_derivability(
    current_html: str, future_html: str, action: Any = None
) -> list[dict]:
    """future 요소별 유도성 라벨 + 판정 근거.

    Args:
        current_html : 현재 state HTML
        future_html  : GT next state HTML
        action       : action dict / JSON 문자열 / `<action>...</action>` 을 포함한
                       user 메시지 문자열 / None

    Returns:
        `extract_elements(future_html)` 와 **같은 길이·같은 순서**의 리스트.
        각 항목:
          {
            "element": {tag, text, own_text, bounds, seq_idx},
            "future_seq_idx": int,          # == element["seq_idx"]
            "derivability": str,            # DERIVABILITY_LABELS 중 하나
            "derivable": bool,
            "reason": {
                "rule": str,                # 어떤 규칙에 걸렸는지
                "matched_current": dict|None,  # 근거가 된 current 요소
                "text_sim": float,          # matched_current 와의 텍스트 유사도
                ...                         # 규칙별 추가 근거(payload / coordinate)
            },
          }
    """
    act = _coerce_action(action)

    cur_els = extract_elements(current_html)
    # 판정은 **own_text 로만** 한다 (모듈 docstring 1번). 흡수 텍스트를 섞으면
    # 컨테이너가 화면 전체 텍스트를 갖고 오분류된다.
    # "같은 자리" 의 키는 **bounds 우선, 없으면 index 폴백**이다.
    # EXP01~04·MobiBench HTML 에는 `bounds` 가 없고 `index` 만 있다 (EXP01 실측
    # `index=` 307,354건 / `bounds=` 0건). bounds 만 보면 그 데이터셋에서 COPY 와
    # ACTION_TARGET 이 **정확히 0** 이 되고, COPY 로 갈 요소가 전부 REFLOW 로
    # 흘러내린다 (실측: EXP05 COPY 14.0%/REFLOW 4.7% ↔ EXP01 COPY 0%/REFLOW 16.2%).
    # 조용히 퇴화하므로 분포를 나란히 놓기 전엔 드러나지 않는다.
    cur_by_slot: dict[str, list[dict]] = {}
    cur_own_lower: dict[str, dict] = {}
    for el in cur_els:
        o = el["own_text"]
        if not o:
            continue
        k = slot_key(el)
        if k:
            cur_by_slot.setdefault(k, []).append(el)
        cur_own_lower.setdefault(o.lower(), el)

    fut_soup = parse_soup(future_html)
    fut_nodes = iter_nodes(fut_soup)
    fut_els = build_element_records(fut_nodes)
    sysui = _system_ui_node_ids(fut_nodes)

    payloads = []
    for k in PAYLOAD_KEYS:
        v = act.get(k)
        if isinstance(v, str) and v.strip():
            payloads.append((k, v.strip().lower()))
    coord = _coordinate(act)
    # index 계열 액션의 대상 요소 index (문자열로 온다). 좌표 규약이 없는 데이터셋에서
    # ACTION_TARGET 을 판정하는 유일한 축이다.
    _ai = act.get("index")
    act_index = str(_ai).strip() if _ai not in (None, "") else ""

    def _ref(el: dict | None) -> dict | None:
        """근거로 실을 current 요소의 최소 표현 (뷰어가 그대로 찍을 수 있게)."""
        if el is None:
            return None
        return {
            "tag": el["tag"],
            "own_text": el["own_text"],
            "bounds": el["bounds"],
            # index 계열 데이터셋(EXP01~04)은 bounds 가 비어 있어 뷰어가 근거 요소를
            # 화면에서 못 짚는다. 슬롯 키의 폴백 축을 그대로 실어 준다.
            "index": el.get("index", ""),
            "seq_idx": el["seq_idx"],
        }

    result: list[dict] = []
    for node, el in zip(fut_nodes, fut_els):
        own = el["own_text"]
        own_l = own.lower()
        b = parse_bounds(el["bounds"])
        label = NON_DERIVABLE
        rule = "no_rule_matched"
        matched: dict | None = None
        extra: dict = {}

        # 같은 자리(bounds)의 current 요소 — COPY 판정에도, NON_DERIVABLE 의
        # "이 자리에 전에는 무엇이 있었나" 근거에도 쓴다.
        slot = slot_key(el)
        same_slot = cur_by_slot.get(slot, []) if slot else []

        if id(node) in sysui:
            label = SYSTEM_UI
            rule = "keycap_density_panel"
        elif not own:
            # 자체 텍스트가 없는 요소 — 컨테이너든 아이콘이든 예측할 콘텐츠가 없다.
            label = STRUCTURE
            rule = "no_own_text"
        elif any(c["own_text"].lower() == own_l for c in same_slot):
            label = COPY
            rule = "same_slot_same_text"
            matched = next(c for c in same_slot if c["own_text"].lower() == own_l)
        elif own_l in cur_own_lower:
            label = REFLOW
            rule = "text_present_elsewhere"
            matched = cur_own_lower[own_l]
        elif any(p in own_l for _, p in payloads):
            label = ACTION_PAYLOAD
            key, p = next((k, p) for k, p in payloads if p in own_l)
            rule = f"action_{key}_in_text"
            extra = {"payload": p}
        elif coord and b and b[0] <= coord[0] <= b[2] and b[1] <= coord[1] <= b[3]:
            label = ACTION_TARGET
            rule = "action_coordinate_inside_bounds"
            extra = {"coordinate": list(coord)}
        elif act_index and el.get("index") == act_index:
            # index 계열(EXP01~04)의 액션은 좌표가 아니라 **대상 요소의 index** 를 준다
            # (`{"action_type":"click","index":"13"}`). 좌표 규약이 아예 없으므로
            # bounds 포함 판정 대신 index 동일성이 같은 역할을 한다.
            label = ACTION_TARGET
            rule = "action_index_matches_element"
            extra = {"action_index": act_index}
        elif coord and b is None:
            # bounds 가 없고 index 규약도 아니면 좌표 포함 판정을 **할 수 없다**.
            # 조용히 False 로 흘리면 그 요소가 NON_DERIVABLE 로 내려앉아 "유도 불가능"
            # 모집단이 부풀고, 그게 데이터셋마다 다르게 발동해 분포 비교를 망친다.
            # 판정 불가임을 근거에 남겨 뷰어에서 구분되게 한다.
            label = NON_DERIVABLE
            rule = "action_target_undecidable_no_bounds"
            extra = {"coordinate": list(coord)}
        else:
            # NON_DERIVABLE. 같은 자리에 다른 내용이 있었다면 그것을 근거로 남긴다 —
            # "같은 슬롯, 새 콘텐츠" 는 뷰어에서 서버 콘텐츠 여부를 눈으로 가르는
            # 가장 강한 단서다.
            if same_slot:
                matched = max(
                    same_slot, key=lambda c: _text_sim(c["own_text"], own)
                )
                rule = "same_slot_new_content"

        reason = {
            "rule": rule,
            "matched_current": _ref(matched),
            "text_sim": round(_text_sim(matched["own_text"], own), 4)
            if matched
            else 0.0,
        }
        reason.update(extra)

        result.append(
            {
                "element": el,
                "future_seq_idx": el["seq_idx"],
                "derivability": label,
                "derivable": is_derivable(label),
                "reason": reason,
            }
        )

    return result


def summarize_derivability(deriv_result: list[dict]) -> dict[str, int]:
    """라벨별 개수 (모든 라벨 키가 항상 존재한다 — 집계 코드가 키 유무를 안 따지게)."""
    counts = {lbl: 0 for lbl in DERIVABILITY_LABELS}
    for d in deriv_result:
        counts[d["derivability"]] += 1
    return counts


def derivability_lookup(deriv_result: list[dict]) -> dict[tuple[str, str], str]:
    """`(tag, slot) -> 라벨` 조회표. slot 은 `bounds` 우선, 없으면 `index` 폴백이다.

    채점기(`addmod_recall` 을 derivable/non_derivable 로 분리)와 뷰어는 seq_idx 가
    아니라 요소 자체로 라벨을 찾아야 한다 — 예측 트리는 GT 와 노드 수가 다르기 때문이다.
    같은 (tag, slot) 이 여러 번 나오면(동일 자리 중첩 컨테이너) **유도 불가능 쪽을
    남긴다**: 유도 가능으로 덮으면 모델의 실패가 지표에서 숨는다 (precision 우선).

    ⚠️ 키를 `bounds` 단독으로 두면 안 된다. EXP01~04 는 bounds 가 전부 빈 문자열이라
    키가 `(tag, "")` 로 붕괴하고, **같은 태그 요소 전부가 라벨 하나를 공유**한다 —
    거기에 위의 "유도 불가능 우선" 규칙이 겹치면 그 태그가 통째로 NON_DERIVABLE 로
    물든다. `slot_key` 의 index 폴백이 그 붕괴를 막는다.
    """
    out: dict[tuple[str, str], str] = {}
    for d in deriv_result:
        el = d["element"]
        key = (el["tag"], slot_key(el))
        cur = out.get(key)
        if cur is None or (is_derivable(cur) and not d["derivable"]):
            out[key] = d["derivability"]
    return out
