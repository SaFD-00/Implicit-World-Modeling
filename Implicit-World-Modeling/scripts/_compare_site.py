"""EXP 별 정성 비교 사이트(index.html + README.md) 빌더.

`scripts/eval_viewer.py --site` 가 호출한다. eval_viewer 가 레지스트리(EVAL_DATASETS)로
leaf 디렉토리·variant·aggregate metric 을 해결해서 넘겨주면, 이 모듈은
prediction/test jsonl 을 읽어 표본을 뽑고 **정본 채점기로 행 단위 점수를 매겨**
자체 완결형 HTML 한 장을 만든다.

산출 (outputs/_compare/):
    on_ac_exp02_stage1_state_compare/{index.html,README.md}
    on_ac_exp02_stage1_action_compare/{index.html,README.md}
    on_ac_exp02_stage2_action_compare/{index.html,README.md}

설계 원칙
---------
1. 카드에 찍히는 점수는 전부 정본 채점기(`_hungarian_eval.compute_hungarian_acc`,
   `_action_eval.evaluate_single{,_xy}`, `thought_eval.rouge_l_f1`) 산출값이다.
   사이트 전용 휴리스틱 점수를 새로 만들지 않는다 — 위쪽 aggregate 표와 카드 점수가
   서로 다른 정의였다면 읽는 사람이 카드 점수를 metric 으로 오해한다.
2. 프롬프트 파서는 실패를 세서 크게 터뜨린다. 두 계열(`## Current State` /
   `Current UI State:`) 중 어느 쪽도 못 읽으면 SystemExit — `woa` 필터가 두 계열을
   모두 놓쳐 전 실험에서 0행을 걸렀던 사고(2026-07-30)의 재발 방지.
3. 이미지는 싣지 않는다 (텍스트 XML/액션만).
"""

from __future__ import annotations

import json
import random
import re
import sys
import warnings
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _action_eval  # noqa: E402
import _hungarian_eval  # noqa: E402
import _state_diff_eval  # noqa: E402
import thought_eval  # noqa: E402
from _prompt_sections import (  # noqa: E402
    REQUIRED_SECTIONS,
    SECTION_MARKERS,
    parse_prompt,
)

REPO = _SCRIPTS.parent
KST = timezone(timedelta(hours=9))

# _action_eval._bbox_elements 가 XML 을 html.parser 로 읽어 행마다 경고를 낸다.
# 채점 결과와 무관한 노이즈라 진짜 경고를 덮지 않도록 여기서만 끈다.
try:
    from bs4 import XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:  # bs4 없는 환경 — 파서 단위 테스트만 돌 때
    pass

# state 예측이 vllm 기본값 1024 토큰에서 잘리던 버그의 수정 시각 (UTC). 이력 참조용
# 이고 **판정에는 쓰지 않는다** — 절단 여부는 `_state_diff_eval.truncated_reason` 이
# prediction 내용을 실측해 가른다. 정본은 `_state_diff_eval` — 절단 판정을 두 군데
# 두면 언젠가 조용히 갈린다 (실제로 갈려서 멀쩡한 EXP01 leaf 를 부당하게 막았다).
MAX_NEW_TOKENS_FIX_UTC = _state_diff_eval.MAX_NEW_TOKENS_FIX_UTC

# xy 통일 액션 스페이스 계열 — 채점 모드가 다르다 (scripts/stage1_eval.sh 와 정합).
XY_FAMILY = {"AC_EXP05", "AC_EXP06", "AC_EXP07_v1", "AC_EXP07_v2"}

TASK_TITLE = {
    "state": "next-state 예측",
    "action": "action 역추론",
    "stage2": "task 수행 (thought + action)",
}


# ── 프롬프트 파싱 ────────────────────────────────────────────────────────
# 파서 본체는 `_prompt_sections` 한 벌뿐이다 — 여기와 `_state_diff_eval` 이 각자
# 복제본을 들면 계열이 하나 더 생겼을 때 한쪽만 고쳐지고 다른 쪽이 조용히 틀린다.
# (그 실패 모드가 woa 필터 사고였다.) 아래 이름들은 기존 호출부 호환용 재수출이다.
_SECTION_MARKERS = SECTION_MARKERS


def detect_layout(prompt: str, kind: str) -> str:
    """프롬프트 실물에서 화면 레이아웃을 판정한다.

    stage 번호로 추정하면 안 된다 — EXP07 의 stage1 `-action` 은 `# Mode:
    NEXT_ACTION_PREDICTION` 이라 two-state 역추론이 아니라 stage2 와 같은
    (Task Instruction + Action History + Current State → thought + action) 이다.
    """
    if kind == "state":
        return "state"
    sections = parse_prompt(prompt)
    if sections.get("next_state"):
        return "action"
    if sections.get("instruction"):
        return "stage2"
    return "action"


def _parse_action_blob(raw: str):
    """'## Action' 본문 / '<action>{...}</action>' 어느 쪽이든 dict 로."""
    if not raw:
        return None
    m = _hungarian_eval.ACTION_TAG_RE.search(raw)
    if m:
        raw = m.group(1)
    return _action_eval.parse_action(raw)


class SiteBuildError(SystemExit):
    """빌드 중단 — 값이 아니라 배선이 깨졌을 때만 낸다."""


# ── 행 단위 채점 (정본 채점기 재사용) ────────────────────────────────────
# 채점기 배선 self-test 용 최소 XML. 두 match_mode 모두 element 가 잡혀야 한다.
_PROBE_XML = {
    "index": '<node index="0"><button index="1" aria-label="OK"/>'
    '<p index="2">hello</p></node>',
    "pos": '<node bounds="[0,0][10,10]" point="[5,5]">'
    '<button bounds="[1,1][5,5]" point="[3,3]">OK</button>'
    '<p bounds="[6,6][9,9]" point="[7,7]">hello</p></node>',
}


def assert_state_scorer_wired(match_mode: str) -> None:
    """Hungarian 채점기가 실제로 동작하는지 표본 채점 전에 한 번 확인한다.

    `_hungarian_eval` 의 bs4/솔버는 지연 로드다. 초기화 없이 부르면
    `extract_elements` 가 예외를 내고 `compute_hungarian_acc` 의 except 가 그걸
    삼켜 **전 행 0점**을 조용히 돌려준다 (2026-08-01 실측: 표본 f1 0.0 vs
    aggregate 0.71). 행 단위로는 구분할 수 없다 — 요소가 없는 GT 문서는 정상적으로
    0 이 나오기 때문이다. 그래서 데이터와 무관한 고정 XML 로 배선만 검사한다.
    """
    _hungarian_eval._lazy_deps()
    xml = _PROBE_XML[match_mode]
    elements = _hungarian_eval.extract_elements(xml, match_mode)
    same = _hungarian_eval.compute_hungarian_acc(xml, xml, match_mode)
    if not elements or same["hungarian_f1"] != 1.0:
        raise SiteBuildError(
            f"Hungarian 채점기 배선 실패 (match_mode={match_mode}): "
            f"elements={len(elements)}, self-F1={same['hungarian_f1']} "
            "— bs4/scipy 의존성을 확인하세요."
        )
    # 요소 단위 감사가 정본 숫자를 재현하는지 데이터 전에 한 번 확인한다.
    # `_state_diff_eval._PROBE` 를 쓰는 이유는 `cur`/`gt` 만으로는 change 축의
    # 지배항인 DELETED 경로를 못 밟기 때문이다 — `maxdel`(요소가 딱 하나이고
    # current 를 하나도 재현하지 않는 예측)이 그 경로를 덮는 유일한 fixture 고,
    # 동시에 "바닥이 0 이 아니다"를 배선 단계에서 증명한다.
    probe = _state_diff_eval._PROBE[match_mode]
    for label, pred in (
        ("perfect", probe["gt"]),
        ("copy", probe["cur"]),
        ("maxdel", probe["maxdel"]),
    ):
        diff = _state_diff_eval.compute_state_diff(
            pred, probe["gt"], probe["cur"], match_mode
        )
        _, derived = state_change_audit(pred, probe["gt"], probe["cur"], match_mode)
        err = _audit_consistency_error(derived, diff)
        if err:
            raise SiteBuildError(
                f"변화 감사 배선 실패 (match_mode={match_mode}, probe={label}): {err}"
            )


# ── 변화 감사 (요소 단위) ────────────────────────────────────────────────
# 숫자를 한 번 더 띄우는 것으로는 "base 와 epoch3 가 같은 화면을 냈는데 f1 이 왜
# 다르냐"에 답할 수 없다. 답하려면 **어떤 요소가 hit 이고 어떤 요소가 miss 인지**가
# 화면에 있어야 한다. 아래가 그 요소 단위 분류다.
#
# 점수 자체는 여전히 정본(`_state_diff_eval.compute_state_diff`) 산출값을 쓴다.
# 여기서 만든 분류는 **매 행 그 숫자를 재현하는지 대조**하고(`_audit_consistency_error`)
# 어긋나면 빌드를 세운다 — 색과 숫자가 조용히 갈리면 감사가 거짓이 되고, 그건 이
# 화면이 없는 것보다 나쁘다 (설계 원칙 1: 사이트 전용 정의를 만들지 않는다).

# 감사 리스트 상한. 퇴화 예측(current 를 하나도 재현하지 않는 예측)은 변화 주장이
# current 요소 수만큼 쏟아져 payload 가 터진다. **개수는 정본에서 읽고 리스트만
# 자른다** — 잘린 리스트를 세어 헤더를 만들면 정확히 감사하고 싶은 그 행에서 분모가
# 조용히 작아진다.
AUDIT_ITEM_CAP = 100
# 요소 텍스트 프리뷰 길이. `_text_sim` 이 토큰 집합 Jaccard 라 짧게 자르면 낮은 sim 의
# 근거가 화면에서 사라진다 — "같은 화면 같은데 왜 다르냐"가 원 질문이다.
AUDIT_TEXT_CHARS = 100


def _el_key(el: dict, match_mode: str) -> str:
    """채점 element ↔ 와이어프레임 노드를 잇는 키. 없으면 빈 문자열.

    pos 계열 XML 은 bounds, index 계열은 index 속성이 자연 키다. SITE_JS 의
    `nodeKey()` 와 **같은 규칙**이어야 색이 엉뚱한 노드에 붙지 않는다.
    같은 키를 가진 노드가 여럿이면 함께 칠해진다(과칠) — 색은 길잡이고 감사의
    정본은 Change 뷰의 요소 리스트다.

    ⚠️ `index=-1`(속성 없음)을 키로 쓰면 안 된다. EXP03/04 의 XML 은 bounds 만
    있고 index 속성이 없는데 `XY_FAMILY` 밖이라 index 모드로 채점된다 — 그러면 전
    요소의 키가 `-1` 로 같아져 **화면 전체가 칠해진다.** 그건 안 칠하는 것보다
    나쁘다 (그럴듯하게 틀린다). 여기서 ""를 내면 JS 의 `markCls` 가 그대로 건너뛴다.
    index 모드에서 bounds 로 대신 키를 잡을 수는 없다 — 정본 `extract_elements` 가
    index 모드 element dict 에 bounds 를 담지 않아서, 그러려면 XML 을 여기서 따로
    한 번 더 파싱해야 하고 그 순간 요소 집합이 정본과 갈릴 수 있다.
    """
    if match_mode == "pos":
        return el.get("bounds", "") or ""
    idx = el.get("index", -1)
    return "" if idx is None or int(idx) < 0 else str(idx)


def _el_brief(el: dict, match_mode: str) -> dict:
    """payload 에 실을 최소 필드. 키를 한 글자로 줄인 것은 크기 때문이다 —
    사이트 하나가 이미 8MB 대고 이 dict 는 (표본 × 세팅 × 요소) 만큼 실린다."""
    out = {"k": _el_key(el, match_mode), "g": el["tag"]}
    txt = (el.get("text") or "")[:AUDIT_TEXT_CHARS]
    if txt:
        out["x"] = txt
    return out


def state_change_audit(
    pred_text: str, gt_text: str, current_text: str, match_mode: str
) -> tuple[dict, dict]:
    """(요소 단위 감사 payload, 정본 대조용 파생값).

    `_state_diff_eval.compute_change_items` 의 hit 규칙을 요소 단위로 되짚는다.
    정본은 개수만 돌려주므로 "어느 요소였나"는 여기서 다시 만들 수밖에 없다 —
    그래서 반환하는 파생값으로 정본 숫자를 재현하는지 매 행 확인한다.

    분류
      GT 쪽  : GT 가 요구한 변화(ADDED/MODIFIED/DELETED) 각각이 strict hit /
               자리만 맞음(loose) / miss 중 무엇인가.
      pred 쪽: 예측이 주장한 변화가 hit 인가 지어낸 것(spurious)인가.

    `_classify_from_els` 인자는 정본과 **완전히 같게** 준다. 특히 pred 쪽의
    `empty_next_is_deletion=False` — 뒤집으면 2026-08-04 에 고친 "빈 예측이 바닥값을
    공짜로 받는" 버그가 감사 화면에서만 되살아난다.
    """
    _hungarian_eval._lazy_deps()
    try:
        pred_els = _hungarian_eval.extract_elements(pred_text, match_mode)
        gt_els = _hungarian_eval.extract_elements(gt_text, match_mode)
        cur_els = _hungarian_eval.extract_elements(current_text, match_mode)
    except Exception:
        # 정본도 이 행은 전 축 None + parse_fail=1 이다. 대조는 게이트가 막는다.
        return {"na": "extract"}, {}

    diff_gt = _state_diff_eval._classify_from_els(cur_els, gt_els, match_mode)
    diff_pred = _state_diff_eval._classify_from_els(
        cur_els, pred_els, match_mode, empty_next_is_deletion=False
    )
    pairs_pg, _ = _hungarian_eval._hungarian_match(
        pred_els, gt_els, match_mode, False
    )
    pred_to_gt = {i: j for i, j, _ in pairs_pg}
    gt_to_pred = {j: i for i, j, _ in pairs_pg}

    # `_classify_from_els(cur, X)` 의 `gt_idx` 는 두 번째 인자(X)의 인덱스다 —
    # pred 쪽 호출에서는 pred_els 의 인덱스이지 GT 인덱스가 아니다.
    gt_changed = {
        d["gt_idx"] for d in diff_gt if d["diff_type"] in ("ADDED", "MODIFIED")
    }
    gt_kind = {d["gt_idx"]: d["diff_type"] for d in diff_gt if "gt_idx" in d}
    gt_deleted = {d["cur_idx"] for d in diff_gt if d["diff_type"] == "DELETED"}
    pred_changed = {
        d["gt_idx"] for d in diff_pred if d["diff_type"] in ("ADDED", "MODIFIED")
    }
    pred_kind = {d["gt_idx"]: d["diff_type"] for d in diff_pred if "gt_idx" in d}
    pred_deleted = {d["cur_idx"] for d in diff_pred if d["diff_type"] == "DELETED"}

    tau = _state_diff_eval.CHANGE_TEXT_SIM_TAU

    def _sim(i: int, j: int) -> float:
        return _hungarian_eval._text_sim(pred_els[i]["text"], gt_els[j]["text"])

    # ── GT 쪽: "바뀌었어야 할 것"을 예측이 맞혔나
    gt_items: list[dict] = []
    hits_strict = hits_loose = 0
    for j in sorted(gt_changed):
        item = {**_el_brief(gt_els[j], match_mode), "t": gt_kind[j][0]}  # A / M
        i = gt_to_pred.get(j)
        if i is not None:
            item["s"] = round(_sim(i, j), 3)
            item["m"] = _el_brief(pred_els[i], match_mode)
            if i in pred_changed:
                hits_loose += 1
                strict = item["s"] >= tau
                hits_strict += strict
                item["h"] = "s" if strict else "l"
            else:
                # 짝은 붙었는데 예측 쪽에서는 그 요소가 current 와 같다고 분류됐다
                # = 바뀌어야 할 자리를 **그대로 베꼈다**. 복사 편향의 직접 증거다.
                item["w"] = "copy"
        gt_items.append(item)
    for c in sorted(gt_deleted):
        # DELETED 는 GT 요소가 아니라 **current 요소**를 가리킨다 (키 공간이 다르다).
        item = {**_el_brief(cur_els[c], match_mode), "t": "D"}
        if c in pred_deleted:
            # DELETED hit 에는 τ 가 걸리지 않는다 (지워진 요소엔 비교할 내용이 없다)
            # — strict/loose 양쪽에 똑같이 들어간다.
            item["h"] = "s"
            hits_strict += 1
            hits_loose += 1
        gt_items.append(item)

    # ── pred 쪽: 예측이 주장한 변화가 진짜였나
    pred_items: list[dict] = []
    for i in sorted(pred_changed):
        item = {**_el_brief(pred_els[i], match_mode), "t": pred_kind[i][0]}
        j = pred_to_gt.get(i)
        if j is not None:
            item["s"] = round(_sim(i, j), 3)
            item["m"] = _el_brief(gt_els[j], match_mode)
            if j in gt_changed:
                item["h"] = "s" if item["s"] >= tau else "l"
            else:
                item["w"] = "nochange"  # GT 는 그 자리를 안 바꿨다 = 지어낸 변화
        pred_items.append(item)
    for c in sorted(pred_deleted):
        item = {**_el_brief(cur_els[c], match_mode), "t": "D"}
        if c in gt_deleted:
            item["h"] = "s"
        pred_items.append(item)

    n_gt = len(gt_changed) + len(gt_deleted)
    n_pred = len(pred_changed) + len(pred_deleted)

    def _f1(hits: int) -> float:
        prec = hits / n_pred if n_pred else 0.0
        rec = hits / n_gt if n_gt else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    n_cur = len(cur_els)
    prec0 = len(gt_deleted) / n_cur if n_cur else 0.0
    rec0 = len(gt_deleted) / n_gt if n_gt else 0.0
    f1_floor = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0

    # addmod hit 은 change hit 과 **다른 판정**이다 — 정본 pred↔gt 매칭에 짝이 있으면
    # 그만이고 τ 도 "예측이 변화라고 주장했는가"도 묻지 않는다. 두 분자를 같은 화면에
    # 나란히 두는 것이 이 뷰의 요점이라 정의 차이를 여기서 뭉개면 안 된다.
    hit_gt = {j for _, j, _ in pairs_pg}
    n_addmod_hit = len(hit_gt & gt_changed)

    derived = {
        "n_change_gt": n_gt,
        "n_change_pred": n_pred,
        "n_change_hit_strict": hits_strict,
        "n_change_hit_loose": hits_loose,
        "change_f1_strict": _f1(hits_strict),
        "change_f1_loose": _f1(hits_loose),
        "change_f1_floor": f1_floor,
        "n_addmod_gt": len(gt_changed),
        "n_addmod_hit": n_addmod_hit,
        "addmod_recall": (
            n_addmod_hit / len(gt_changed) if gt_changed else None
        ),
    }
    audit = {"gt": gt_items[:AUDIT_ITEM_CAP], "pd": pred_items[:AUDIT_ITEM_CAP]}
    if len(gt_items) > AUDIT_ITEM_CAP:
        audit["gt_more"] = len(gt_items) - AUDIT_ITEM_CAP
    if len(pred_items) > AUDIT_ITEM_CAP:
        audit["pd_more"] = len(pred_items) - AUDIT_ITEM_CAP
    return audit, derived


def _audit_consistency_error(derived: dict, diff: dict) -> str | None:
    """감사(요소 단위)가 정본(행 단위)을 재현하지 못하면 설명, 같으면 None.

    **정의된 키에서만** 대조한다. 정본이 None 을 내는 경로가 넷이라 (추출 예외 /
    GT 요소 0개 early-return / 변화가 양쪽 다 0 / addmod 분모 공집합) 게이트 없이
    비교하면 정상 행에서 빌드가 죽는다. 비교는 tolerance 가 아니라 `round(_, 4)`
    동일성이다 — 정본이 round 4 로 내보내므로 이게 정확히 같은 연산이다.
    """
    if not derived:
        return None  # 추출 실패 행 — 정본도 전 축 None 이라 대조할 것이 없다
    if diff.get("change_f1_strict") is not None:
        # 이 값이 None 이 아니면 `compute_change_items` 가 실제로 돌았다는 뜻이라
        # n_change_* 도 정본 값이다 (early-return 경로에서는 zero_counts 가 남는다).
        for key in ("n_change_gt", "n_change_pred"):
            if derived[key] != diff[key]:
                return f"{key}: 감사 {derived[key]} vs 정본 {diff[key]}"
        for key in ("change_f1_strict", "change_f1_loose", "change_f1_floor"):
            if round(derived[key], 4) != diff[key]:
                return f"{key}: 감사 {round(derived[key], 4)} vs 정본 {diff[key]}"
    if diff.get("addmod_recall") is not None:
        want_n = diff["n_gt_added"] + diff["n_gt_modified"]
        if derived["n_addmod_gt"] != want_n:
            return f"n_addmod_gt: 감사 {derived['n_addmod_gt']} vs 정본 {want_n}"
        if round(derived["addmod_recall"], 4) != diff["addmod_recall"]:
            return (
                f"addmod_recall: 감사 {round(derived['addmod_recall'], 4)} "
                f"vs 정본 {diff['addmod_recall']}"
            )
    return None


def gt_change_marks(current_text: str, gt_text: str, match_mode: str) -> dict:
    """샘플 단위 "바뀌어야 할 자리". 세팅과 무관하므로 표본 행에 한 번만 싣는다.

    cur : current 에서 **사라져야** 할 요소 / gt : GT 에서 새로 생기거나 바뀐 요소.
    두 dict 의 키 공간이 다르다는 것이 요점이다 — current 키를 GT pane 에 칠하면
    화면은 멀쩡해 보이는데 엉뚱한 노드가 칠해진다.
    """
    _hungarian_eval._lazy_deps()
    try:
        cur_els = _hungarian_eval.extract_elements(current_text, match_mode)
        gt_els = _hungarian_eval.extract_elements(gt_text, match_mode)
    except Exception:
        return {"cur": {}, "gt": {}}
    cur_marks: dict[str, str] = {}
    gt_marks: dict[str, str] = {}
    for d in _state_diff_eval._classify_from_els(cur_els, gt_els, match_mode):
        # 빈 키(= 그 XML 계열에 식별 속성이 없다)는 싣지 않는다. `_el_key` 참고.
        if d["diff_type"] == "DELETED":
            key = _el_key(cur_els[d["cur_idx"]], match_mode)
            if key:
                cur_marks[key] = "mk-del"
        elif d["diff_type"] in ("ADDED", "MODIFIED"):
            key = _el_key(gt_els[d["gt_idx"]], match_mode)
            if key:
                gt_marks[key] = "mk-add" if d["diff_type"] == "ADDED" else "mk-mod"
    return {"cur": cur_marks, "gt": gt_marks}


def score_state_row(
    pred_text: str, gt_text: str, current_text: str, match_mode: str
) -> dict:
    """행 단위 점수 — hungarian 계열 + state-diff(addmod/change) 계열.

    `current_text` 는 선택 인자가 아니다. state-diff 축은 current 없이는 정의되지
    않는데, 기본값을 두면 호출부가 빠뜨렸을 때 **축 전체가 조용히 빈칸**이 된다 —
    이 코드베이스가 반복해 다친 실패 모드라 인자를 강제한다.
    """
    _hungarian_eval._lazy_deps()  # 미초기화면 전 행 0점 — 호출 순서에 기대지 않는다
    hung = _hungarian_eval.compute_hungarian_acc(pred_text, gt_text, match_mode)
    diff = _state_diff_eval.compute_state_diff(
        pred_text, gt_text, current_text, match_mode
    )
    audit, derived = state_change_audit(pred_text, gt_text, current_text, match_mode)
    err = _audit_consistency_error(derived, diff)
    if err:
        raise SiteBuildError(
            "요소 단위 감사가 정본 채점기 값을 재현하지 못했습니다 — "
            f"{err}. `_state_diff_eval` 의 hit 규칙이 바뀌었다면 "
            "`_compare_site.state_change_audit` 를 같이 고쳐야 합니다 "
            "(색과 숫자가 갈리면 이 화면은 거짓 감사가 된다)."
        )
    pos_key = "hungarian_pos" if match_mode == "pos" else "hungarian_idx"
    return {
        "exact": gt_text.strip() == pred_text.strip(),
        "f1": round(hung["hungarian_f1"] * 100, 1),
        "ea": round(hung["hungarian_ea"] * 100, 1),
        "prec": round(hung["hungarian_prec"] * 100, 1),
        "rec": round(hung["hungarian_rec"] * 100, 1),
        "text": round(hung["hungarian_text"] * 100, 1),
        "pos": round(hung[pos_key] * 100, 1),
        "pos_label": "위치(±50px)" if match_mode == "pos" else "index(±2)",
        "bleu4": round(_hungarian_eval.calc_bleu(gt_text, pred_text) * 100, 1),
        "rouge_l": round(_hungarian_eval.calc_rouge_l(gt_text, pred_text) * 100, 1),
        "pred_lines": pred_text.count("\n") + 1,
        "label_lines": gt_text.count("\n") + 1,
        # ── state-diff 축 (값은 전부 정본 산출값) ────────────────────────
        # 0~1 스케일 그대로 싣는다. 화면에서 %로 바꾸는 것은 표시 계층의 일이고,
        # 여기서 100 을 곱하면 위쪽 aggregate 표와 다른 수가 payload 에 남는다.
        "addmod_recall": diff["addmod_recall"],
        "change_f1_strict": diff["change_f1_strict"],
        "change_f1_loose": diff["change_f1_loose"],
        "change_f1_floor": diff["change_f1_floor"],
        "no_change_acc": diff["no_change_acc"],
        "copy_excess": diff["copy_excess"],
        "parse_fail": diff["parse_fail"],
        "n_change_gt": diff["n_change_gt"],
        "n_change_pred": diff["n_change_pred"],
        "n_cur_el": diff["n_cur"],
        "n_gt_el": diff["n_gt"],
        "n_pred_el": diff["n_pred"],
        # 분자는 정본이 내주지 않아 감사에서 가져온다. 위 `_audit_consistency_error`
        # 가 이 분자로 정본 비율을 재현하는지 매 행 확인한 뒤라 신뢰할 수 있다.
        "n_addmod_gt": derived.get("n_addmod_gt", 0),
        "n_addmod_hit": derived.get("n_addmod_hit", 0),
        "n_change_hit_strict": derived.get("n_change_hit_strict", 0),
        "n_change_hit_loose": derived.get("n_change_hit_loose", 0),
        "audit": audit,
    }


def score_action_row(gt_action, pred_action, ui_xml: str, coord_mode: str) -> dict:
    if coord_mode == "xy":
        r = _action_eval.evaluate_single_xy(gt_action, pred_action, ui_xml)
        field = (
            "bbox"
            if r["has_bbox_check"]
            else "dir"
            if r["has_dir_check"]
            else "app"
            if r["has_app_check"]
            else "text"
            if r["has_text_check"]
            else ""
        )
        no_bbox = r["no_bbox"]
    else:
        r = _action_eval.evaluate_single(gt_action, pred_action)
        field = (
            "index"
            if r["has_index_check"]
            else "dir"
            if r["has_dir_check"]
            else "app"
            if r["has_app_check"]
            else "text"
            if r["has_text_check"]
            else ""
        )
        no_bbox = False
    return {
        "parsed": bool(r["parsed"]),
        "type_correct": bool(r["type_correct"]),
        "step_correct": bool(r["step_correct"]),
        "field": field,
        "no_bbox": bool(no_bbox),
    }


# ── 표시용 헬퍼 ──────────────────────────────────────────────────────────
def exp_slug(exp: str) -> str:
    return "on_" + exp.lower()


def site_dirname(exp: str, stage: int, task: str) -> str:
    kind = "state" if task == "state" else "action"
    return f"{exp_slug(exp)}_stage{stage}_{kind}_compare"


def split_label(logical_key: str) -> str:
    key = logical_key
    woa = key.endswith("-without-open_app")
    if woa:
        key = key[: -len("-without-open_app")]
    if key.endswith("-id"):
        base = "ID"
    elif key.endswith("-ood"):
        base = "OOD"
    elif key.startswith("on-"):
        base = key[3:]
    else:
        base = key
    return f"{base} · woa" if woa else base


def setting_sort_key(model: str, vpath: str):
    """base → 학습 variant 순, 같은 variant 안에서는 epoch 오름차순."""
    head, _, tail = vpath.partition("/")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", tail or head)
    epoch = float(m.group(1)) if m else -1.0
    return (model, 0 if head == "base" else 1, head, epoch, vpath)


def humanize_setting(model: str, vpath: str, multi_model: bool) -> str:
    return f"{model} · {vpath}" if multi_model else vpath


def _fmt_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=UTC)
    return f"{dt.astimezone(KST):%Y-%m-%d %H:%M} KST"


def read_element_set(metric_path: Path) -> str | None:
    """aggregate metric 파일에 박힌 element 집합 스탬프. 없으면 None."""
    try:
        return json.loads(metric_path.read_text(encoding="utf-8")).get("element_set")
    except Exception:
        return None


def element_set_note(stamps: set[str | None], runtime: str | None) -> str:
    """표본 표와 aggregate 표의 element 집합이 갈렸으면 경고 문구, 아니면 "".

    `_hungarian_eval.ELEMENT_SET` 은 **전역**이라 인자로 흐르지 않는다. 사이트의
    표본 점수는 지금 이 프로세스의 전역으로 새로 채점되고, 위쪽 aggregate 표는
    디스크의 JSON 을 그대로 읽는다 — 그 JSON 이 다른 집합으로 채점됐다면 두 표는
    **정의가 다른 수**다. 실측(2026-08-21, 같은 50행): full↔legacy 격차가
    EXP05(pos) 최대 2.5p, EXP02(index) 최대 6.6p 였다. 표본 표에 addmod/change 축이
    생기면서 두 표에 **같은 이름의 열**이 나란히 놓였으므로 그냥 두면 오독한다.

    스탬프가 없는 파일은 스탬프 도입 이전 산출물이다 — "같다"고 단정할 근거가
    없으므로 조용히 통과시키지 않는다.
    """
    if runtime is None:
        return ""
    unknown = None in stamps
    other = {s for s in stamps if s is not None and s != runtime}
    if not unknown and not other:
        return ""
    parts = [
        f"⚠️ 표본 표는 지금 이 자리에서 `element_set={runtime}` 로 다시 채점한 값이다."
    ]
    if other:
        parts.append(
            "위쪽 aggregate 표의 일부 leaf 는 `element_set="
            + "`/`".join(sorted(other))
            + "` 로 채점돼 있어 **두 표의 정의가 다르다**."
        )
    if unknown:
        parts.append(
            "일부 leaf 의 metric 파일에는 element 집합 스탬프가 없다 "
            "(스탬프 도입 이전 산출물) — 같은 집합이라고 단정할 수 없다."
        )
    parts.append(
        "두 표를 나란히 읽으려면 `scripts/rebuild_eval_metrics.sh` 로 aggregate 를 "
        "다시 산출할 것. 카드 점수와 표본 표끼리는 정합하다."
    )
    return " ".join(parts)


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ── 사이트 조립 ──────────────────────────────────────────────────────────
def _probe_prompt(splits: list[dict], by_id: dict[str, dict]) -> str:
    """레이아웃 판정용으로 첫 split·첫 setting 의 첫 prediction prompt 한 줄만 읽는다."""
    sp = splits[0]
    leaf = sp["dirs"][sp["setting_ids"][0]]
    with (leaf / sp["pred_filename"]).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line).get("prompt", "")
    return ""


def build_site(
    *,
    stage: int,
    exp: str,
    kind: str,
    settings: list[dict],
    splits: list[dict],
    metric_keys: list[str],
    out_dir: Path,
    samples: int,
    seed: int,
) -> dict:
    """자체 완결형 비교 사이트를 out_dir 에 쓴다.

    settings : [{id, label, model, vpath}] — 전체 합집합 (표시 순서)
    splits   : [{key, pred_filename, test_path, setting_ids: [...],
                 dirs: {setting_id: Path}, metrics: {setting_id: dict}}]

    setting 은 분할마다 있을 수도 없을 수도 있다 (예: base 만 MB 에서 평가됨).
    합집합을 강제로 맞추면 없는 leaf 를 읽게 되므로 분할별 부분집합을 그대로 싣는다.
    """
    by_id = {st["id"]: st for st in settings}
    match_mode = "pos" if exp in XY_FAMILY else "index"
    coord_mode = "xy" if exp in XY_FAMILY else "index"
    task = detect_layout(_probe_prompt(splits, by_id), kind)
    metric_filename = (
        "hungarian_metrics.json" if kind == "state" else "action_metrics.json"
    )
    rng_base = f"{seed}|{exp}|stage{stage}|{kind}"
    if task == "state":
        assert_state_scorer_wired(match_mode)

    data_splits: dict[str, dict] = {}
    split_order: list[str] = []
    provenance: list[dict] = []
    element_sets: set[str | None] = set()
    parse_failures: list[str] = []
    truncated_leaves: list[str] = []

    for sp in splits:
        key = sp["key"]
        split_settings = [by_id[sid] for sid in sp["setting_ids"]]
        preds: dict[str, list[dict]] = {}
        for st in split_settings:
            leaf = sp["dirs"][st["id"]]
            pred_path = leaf / sp["pred_filename"]
            preds[st["id"]] = read_jsonl(pred_path)
            mtime = pred_path.stat().st_mtime
            provenance.append(
                {
                    "split": key,
                    "setting": st["label"],
                    "path": str(leaf.relative_to(REPO)),
                    "mtime": _fmt_ts(mtime),
                    # 채점 파일 존재 여부. merged dict 로 판단하면 predict_results 만
                    # 있어도 True 가 되어 "hungarian 미채점"을 놓친다.
                    "has_metrics": (leaf / metric_filename).is_file(),
                }
            )
            if kind == "state" and (leaf / metric_filename).is_file():
                element_sets.add(read_element_set(leaf / metric_filename))
            if kind == "state" and _state_diff_eval.truncated_reason(str(pred_path)):
                truncated_leaves.append(str(leaf.relative_to(REPO)))

        lengths = {st["id"]: len(preds[st["id"]]) for st in split_settings}
        if len(set(lengths.values())) > 1:
            raise SiteBuildError(
                f"{exp}/stage{stage}/{key}: prediction 행 수 불일치 — {lengths}"
            )
        n = next(iter(lengths.values()))

        gt_entries: list[dict] | None = None
        test_path = sp.get("test_path")
        if test_path is not None and Path(test_path).is_file():
            gt_entries = read_jsonl(Path(test_path))
            if len(gt_entries) != n:
                raise SiteBuildError(
                    f"{exp}/stage{stage}/{key}: test({Path(test_path).name}) "
                    f"{len(gt_entries)} != predictions {n}"
                )
        elif task in ("action", "stage2"):
            raise SiteBuildError(
                f"{exp}/stage{stage}/{key}: action 채점에 필요한 test jsonl 이 없음 "
                f"({test_path})"
            )

        rng = random.Random(f"{rng_base}|{key}")
        k = min(samples, n)
        indices = sorted(rng.sample(range(n), k))

        anchor = split_settings[0]["id"]
        sample_recs = []
        for i in indices:
            base_rec = preds[anchor][i]
            # 세팅 간 prompt/label 동일성 — README 가 주장하는 정렬 검증의 실체.
            for st in split_settings[1:]:
                other = preds[st["id"]][i]
                if other.get("prompt") != base_rec.get("prompt") or other.get(
                    "label"
                ) != base_rec.get("label"):
                    raise SiteBuildError(
                        f"{exp}/stage{stage}/{key} row {i}: 세팅 간 prompt/label 불일치 "
                        f"({split_settings[0]['label']} vs {st['label']}) — 같은 test 가 아님"
                    )

            sections = parse_prompt(base_rec.get("prompt", ""))
            missing = [
                s for s in REQUIRED_SECTIONS[task] if not sections.get(s, "").strip()
            ]
            if missing:
                parse_failures.append(f"{key} row {i}: 누락 섹션 {missing}")
                continue

            gt_label = base_rec.get("label", "")
            rec: dict = {"index": i}

            if task == "state":
                rec["current_state"] = sections["current_state"]
                rec["label"] = gt_label
                rec["action"] = _parse_action_blob(sections.get("action", ""))
                rec["action_raw"] = sections.get("action", "")
                # "바뀌어야 할 자리"는 (current, GT) 만의 함수라 세팅과 무관하다 —
                # 세팅마다 다시 계산하면 같은 값을 N 번 싣게 된다.
                rec["change_marks"] = gt_change_marks(
                    sections["current_state"], gt_label, match_mode
                )
                rec["predictions"] = {
                    st["id"]: {
                        "text": preds[st["id"]][i].get("predict", ""),
                        "stats": score_state_row(
                            preds[st["id"]][i].get("predict", ""),
                            gt_label,
                            sections["current_state"],
                            match_mode,
                        ),
                    }
                    for st in split_settings
                }
            else:
                gt_entry = gt_entries[i]
                gt_action = _action_eval.parse_action(gt_entry["messages"][-1]["value"])
                ui_xml = (
                    _action_eval._extract_ui_xml(gt_entry) if coord_mode == "xy" else ""
                )
                rec["current_state"] = sections["current_state"]
                rec["gt_action"] = gt_action
                rec["gt_action_raw"] = gt_label.strip()
                if task == "action":
                    rec["next_state"] = sections.get("next_state", "")
                else:
                    rec["instruction"] = sections.get("instruction", "")
                    rec["history"] = sections.get("history", "")
                    rec["gt_thought"] = thought_eval._extract_thought(gt_label) or ""
                rec["predictions"] = {}
                for st in split_settings:
                    pred_text = preds[st["id"]][i].get("predict", "")
                    pred_action = _action_eval.parse_action(pred_text)
                    stats = score_action_row(gt_action, pred_action, ui_xml, coord_mode)
                    entry = {
                        "text": pred_text,
                        "action": pred_action,
                        "stats": stats,
                    }
                    if task == "stage2":
                        pred_thought = thought_eval._extract_thought(pred_text) or ""
                        entry["thought"] = pred_thought
                        stats["thought_rouge_l"] = round(
                            thought_eval.rouge_l_f1(pred_thought, rec["gt_thought"])
                            * 100,
                            1,
                        )
                    rec["predictions"][st["id"]] = entry

            sample_recs.append(rec)

        data_splits[key] = {
            "label": split_label(key),
            "population": n,
            "settings": [st["id"] for st in split_settings],
            "indices": [r["index"] for r in sample_recs],
            "metrics": {
                st["id"]: sp["metrics"].get(st["id"], {}) for st in split_settings
            },
            "sample_metrics": _sample_metrics(task, split_settings, sample_recs),
            "samples": sample_recs,
        }
        split_order.append(key)

    if parse_failures:
        head = "\n  ".join(parse_failures[:10])
        raise SiteBuildError(
            f"{exp}/stage{stage}/{task}: 프롬프트 파싱 실패 {len(parse_failures)}건 — "
            f"파서가 프롬프트 계열을 못 읽고 있습니다.\n  {head}"
        )

    slug = site_dirname(exp, stage, kind)
    data = {
        "title": f"{exp} · stage{stage} {TASK_TITLE[task]}",
        "slug": slug,
        "exp": exp,
        "stage": stage,
        "task": task,
        "seed": seed,
        "sample_size": samples,
        "match_mode": match_mode,
        "coord_mode": coord_mode,
        # change 축 strict 의 내용 임계. 화면이 "sim 0.44 < τ 0.9 라서 탈락"까지
        # 보여야 감사가 되므로 상수를 payload 에 싣는다 (JS 에 하드코딩하면 정본과
        # 조용히 갈린다).
        "change_tau": _state_diff_eval.CHANGE_TEXT_SIM_TAU,
        # 요소 색칠이 이 EXP 에서 쓸 수 있는가. EXP03/04 처럼 XML 에 index 속성이
        # 없으면 (index 모드인데 bounds 만 있다) 키를 만들 수 없어 아무것도 안
        # 칠해진다 — 그걸 "변화가 없다"로 오독하지 않게 화면에 명시한다.
        # 위 aggregate 표(디스크 JSON)와 아래 표본 표(지금 채점)의 element 집합이
        # 갈렸는지. 갈렸으면 두 표는 같은 이름의 다른 정의다 — `element_set_note` 참고.
        "element_set": getattr(_hungarian_eval, "ELEMENT_SET", None),
        "element_set_note": (
            element_set_note(
                element_sets, getattr(_hungarian_eval, "ELEMENT_SET", None)
            )
            if task == "state"
            else ""
        ),
        "marks_usable": any(
            r.get("change_marks", {}).get("gt") or r.get("change_marks", {}).get("cur")
            for sp in data_splits.values()
            for r in sp["samples"]
        )
        if task == "state"
        else False,
        "settings": [st["id"] for st in settings],
        "setting_labels": {st["id"]: st["label"] for st in settings},
        "metric_keys": metric_keys,
        "split_order": split_order,
        "splits": data_splits,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    index_path.write_text(render_html(data), encoding="utf-8")
    (out_dir / "README.md").write_text(
        render_readme(data, provenance, sorted(set(truncated_leaves))),
        encoding="utf-8",
    )
    return {
        "path": index_path,
        "slug": slug,
        "title": data["title"],
        "n_settings": len(settings),
        "n_splits": len(split_order),
        "n_samples": sum(len(v["samples"]) for v in data_splits.values()),
        "size": index_path.stat().st_size,
    }


def _sample_metrics(task: str, settings: list[dict], recs: list[dict]) -> dict:
    """표본(N=samples) 안에서의 집계 — aggregate 표와 같은 채점 정의로 계산된다."""
    out: dict[str, dict] = {}
    for st in settings:
        rows = [r["predictions"][st["id"]]["stats"] for r in recs]
        if not rows:
            out[st["id"]] = {}
            continue
        n = len(rows)

        def avg(key, rows=rows, n=n):
            return round(sum(float(r.get(key) or 0) for r in rows) / n, 1)

        if task == "state":
            # state-diff 축은 **행마다 정의 여부가 다르다** (변화 없는 행, GT 요소 0개,
            # 파싱 실패 행에서 None). `avg()` 처럼 None 을 0 으로 세면 정의불능이
            # 실패로 둔갑해 평균이 아래로 끌린다. 정본이 `n_*` 를 쌍으로 내는 것과
            # 같은 이유로 **분모를 따로 싣는다** — 분모가 세팅마다 다르면 두 평균은
            # 서로 다른 population 위의 수라 나란히 못 읽는다.
            def avg_defined(key, rows=rows):
                vals = [r[key] for r in rows if r.get(key) is not None]
                if not vals:
                    return None, 0
                return round(100 * sum(vals) / len(vals), 1), len(vals)

            addmod, n_addmod = avg_defined("addmod_recall")
            strict, n_strict = avg_defined("change_f1_strict")
            loose, n_loose = avg_defined("change_f1_loose")
            floor, n_floor = avg_defined("change_f1_floor")
            # 세 축은 정본이 **정확히 같은 행에서만** 정의한다고 약속한다. 어긋나면
            # 세 평균의 분모가 갈려 "strict < floor" 같은 판단이 무의미해지므로
            # 조용히 넘기지 않는다.
            if not (n_strict == n_loose == n_floor):
                raise SiteBuildError(
                    "change 축 정의 구간 불일치 — strict/loose/floor 의 분모가 "
                    f"{n_strict}/{n_loose}/{n_floor} 로 갈렸습니다. "
                    "`_state_diff_eval.compute_change_items` 의 None 규칙을 확인하세요."
                )
            out[st["id"]] = {
                "hung_f1": avg("f1"),
                "hung_ea": avg("ea"),
                "addmod_recall": addmod,
                "n_addmod_recall": n_addmod,
                "change_f1_strict": strict,
                "change_f1_loose": loose,
                "change_f1_floor": floor,
                "n_change_f1": n_strict,
                "bleu4": avg("bleu4"),
                "rouge_l": avg("rouge_l"),
                "exact": round(100 * sum(bool(r["exact"]) for r in rows) / n, 1),
            }
        else:
            m = {
                "parse_rate": round(100 * sum(r["parsed"] for r in rows) / n, 1),
                "type_acc": round(100 * sum(r["type_correct"] for r in rows) / n, 1),
                "step_acc": round(100 * sum(r["step_correct"] for r in rows) / n, 1),
            }
            if task == "stage2":
                m["thought_rouge_l"] = avg("thought_rouge_l")
            out[st["id"]] = m
    return out


# ── README ──────────────────────────────────────────────────────────────
def render_readme(data: dict, provenance: list[dict], truncated: list[str]) -> str:
    task = data["task"]
    lines = [
        f"# {data['title']} 정성 비교",
        "",
        "브라우저에서 `index.html`을 직접 열면 됩니다. 별도 서버나 설치가 필요하지 않습니다.",
        "",
        f"- 생성 시각: {datetime.now(tz=KST):%Y-%m-%d %H:%M} KST "
        "(평가가 진행 중이면 그 시점까지 완료된 세팅만 담긴다 — 끝난 뒤 다시 만들 것)",
        f"- 랜덤 시드: `{data['seed']}` (분할별 독립 추출 — 시드가 같으면 같은 표본)",
        f"- 표본: 분할별 `{data['sample_size']}`개",
        f"- 세팅 {len(data['settings'])}개: "
        + ", ".join(data["setting_labels"][s] for s in data["settings"]),
    ]
    for key in data["split_order"]:
        sp = data["splits"][key]
        lines.append(
            f"- 분할 `{key}` ({sp['label']}): 모집단 {sp['population']:,}개 · "
            f"표본 {len(sp['samples'])}개"
        )
    lines += [
        "- 모든 세팅에서 prompt/label 의 전체 행 수 일치 + 표본 행의 prompt/label "
        "byte-identity 검증 완료 (불일치 시 빌드가 실패한다)",
        "- 이미지는 싣지 않는다 — XML/액션 텍스트만 비교한다",
        "",
        "## 점수의 출처",
        "",
        "카드와 표본 표의 점수는 전부 정본 채점기를 **이 자리에서 다시 돌린** 값이다 "
        "(사이트 전용 정의를 만들지 않는다).",
    ]
    if data.get("element_set_note"):
        lines += ["", f"> {data['element_set_note']}"]
    elif task == "state":
        lines += [
            "",
            f"- element 집합 `{data.get('element_set')}` — 위쪽 aggregate 표의 metric "
            "파일과 같은 값이라 두 표를 나란히 읽어도 된다.",
        ]
    if task == "state":
        lines += [
            f"- `scripts/_hungarian_eval.py::compute_hungarian_acc` "
            f"(`--match-mode {data['match_mode']}`) → F1 / EA / prec / rec / text / 위치",
            "- 같은 모듈의 `calc_bleu` · `calc_rouge_l` → BLEU-4 / ROUGE-L",
            "- `scripts/_state_diff_eval.py::compute_state_diff` → addmod recall / "
            "change F1 (strict · loose · 바닥)",
            "",
            "### 카드의 `Change` 탭 — 점수 감사",
            "",
            "카드마다 `Change` 뷰가 있고, 거기서 **분자에 들어간 요소와 빠진 요소**를 "
            "요소 단위로 볼 수 있다. GT 가 요구한 변화(ADDED/MODIFIED/DELETED) 각각이 "
            "strict hit / 자리만 맞음 / miss 중 무엇인지, 어떤 예측 요소와 짝이 붙었고 "
            "`text_sim` 이 얼마였는지가 한 줄씩 나온다. 요소 분류는 "
            "`_state_diff_eval._classify_from_els` 산출이고, 그 분류로 다시 센 값이 "
            "정본 `compute_state_diff` 의 수치를 재현하는지 **행마다 대조**한다 "
            "(어긋나면 빌드가 실패한다 — 색과 숫자가 갈리면 감사가 거짓이 된다).",
            "",
            f"> `change_f1_strict` 를 `change_f1_floor` 없이 읽지 말 것. 이 축의 바닥은 "
            "0 이 아니라 0.2~0.4 이고, 바닥을 빼면 퇴화 모델이 학습 모델보다 좋아 보인다. "
            "카드 헤드라인의 `Δ` 가 그 차이(strict − 바닥)이고 **색은 Δ 의 부호**로 정해진다.",
            "",
            f"> 상단 두 화면의 색은 GT 기준 '바뀌어야 할 자리'다 — 왼쪽(current)의 빨강은 "
            "사라져야 할 요소, 오른쪽(GT)의 초록/주황은 새로 생기거나 바뀐 요소. "
            f"노드 매칭 키는 {'bounds' if data['match_mode'] == 'pos' else 'index'} 속성이라 "
            "키가 겹치는 노드는 함께 칠해진다 — 감사의 정본은 `Change` 뷰의 리스트다.",
        ]
    else:
        fn = "evaluate_single_xy" if data["coord_mode"] == "xy" else "evaluate_single"
        lines.append(
            f"- `scripts/_action_eval.py::{fn}` "
            f"(`--coord-mode {data['coord_mode']}`) → parse / type / step 정답 판정"
        )
        if task == "stage2":
            lines.append(
                "- `scripts/thought_eval.py::rouge_l_f1` → thought ROUGE-L (F1)"
            )
    lines += [
        "",
        "## 표본 출처 (leaf · 예측 파일 생성 시각)",
        "",
        "| 분할 | 세팅 | leaf | predictions mtime |",
        "|---|---|---|---|",
    ]
    for p in provenance:
        note = "" if p["has_metrics"] else " ⚠️ metric 파일 없음"
        lines.append(
            f"| {p['split']} | {p['setting']} | `{p['path']}` | {p['mtime']}{note} |"
        )
    if any(not p["has_metrics"] for p in provenance):
        lines += [
            "",
            "> ⚠️ 위 표의 일부 leaf 에 aggregate metric 파일(`hungarian_metrics.json` /"
            " `action_metrics.json`)이 없다. 그 세팅의 '전체 지표' 행은 값이 0 이 아니라"
            " **미채점**이며, 사이트의 '표본 지표'와 카드 점수는 그와 무관하게 이 자리에서"
            " 다시 채점한 값이라 정상이다. `scripts/rebuild_eval_metrics.sh` 로 채운다.",
        ]
    if truncated:
        lines += [
            "",
            "> ⚠️ **절단 경고** — 아래 leaf 의 state prediction 은 `max_new_tokens` 기본값"
            " 1024 토큰에서 하드 컷됐다 (`6a4b59e` 이전 실행). 예측의 상당수가 **정확히"
            " 1024 토큰**인 것을 실측해 판정했다 — 화면의 예측과 hungarian 계열 점수는"
            " 모델 성능이 아니라 절단의 결과다.",
            "",
        ]
        lines += [f"> - `{p}`" for p in truncated]
    lines += [
        "",
        "## 재생성",
        "",
        "```bash",
        f"python scripts/eval_viewer.py --site --stages {data['stage']} \\",
        "    --include <EXP>:<MODEL> [...] \\",
        f"    --samples {data['sample_size']} --seed {data['seed']}",
        "```",
        "",
    ]
    return "\n".join(lines)


# ── HTML ────────────────────────────────────────────────────────────────
def _pane_config(task: str) -> tuple[dict, dict]:
    """(paneA, paneB) — 각 pane 의 제목과 wireframe 지원 여부."""
    if task == "state":
        return (
            {"title": "Current UI State", "wire": True},
            {"title": "Ground Truth · Next State", "wire": True},
        )
    if task == "action":
        return (
            {"title": "Current UI State", "wire": True},
            {"title": "Next UI State", "wire": True},
        )
    return (
        {"title": "Task Instruction · Action History", "wire": False},
        {"title": "Current UI State", "wire": True},
    )


def _pane_html(pane_id: str, cfg: dict) -> str:
    tools = []
    if cfg["wire"]:
        tools.append(
            f'<button class="mini active" data-context-view="wire" '
            f'data-target="{pane_id}">와이어프레임</button>'
        )
        tools.append(
            f'<button class="mini" data-context-view="raw" '
            f'data-target="{pane_id}">원문</button>'
        )
    tools.append(f'<button class="mini" data-copy="{pane_id}">복사</button>')
    return (
        '<div class="context-cell">'
        f'<div class="section-title"><span>{cfg["title"]}</span></div>'
        f'<div class="context-tools">{"".join(tools)}</div>'
        f'<div class="context-view" id="pane-{pane_id}" '
        f'data-wire="{1 if cfg["wire"] else 0}"></div>'
        "</div>"
    )


def render_html(data: dict) -> str:
    pane_a, pane_b = _pane_config(data["task"])
    split_buttons = "".join(
        '<button data-split="{k}"{cls}>{label}</button>'.format(
            k=k,
            cls=' class="active"' if i == 0 else "",
            label=data["splits"][k]["label"],
        )
        for i, k in enumerate(data["split_order"])
    )
    subtitle = {
        "state": "동일 사례에 대한 세팅별 next-state 예측 비교",
        "action": "두 UI 상태 사이에서 역추론한 action 비교",
        "stage2": "task 수행 스텝의 thought + action 비교",
    }[data["task"]]
    gt_block = (
        '<div class="gt-thought"><span>GT thought</span><p id="gtThought"></p></div>'
        if data["task"] == "stage2"
        else ""
    )
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{data["title"]} 정성 비교</title>
<style>{SITE_CSS}</style>
</head>
<body>
<header class="topbar"><div class="topbar-inner">
  <div class="brand"><h1>{data["title"]} · Qualitative Comparator</h1><p>{subtitle}</p></div>
  <div class="toolbar">
    <div class="seg" id="splitSeg">{split_buttons}</div>
    <button class="btn" id="prevBtn" title="이전 샘플 (←)">←</button>
    <select id="sampleSelect" aria-label="샘플 선택"></select>
    <button class="btn" id="nextBtn" title="다음 샘플 (→)">→</button>
    <button class="btn" id="wrapBtn">줄바꿈</button>
    <button class="btn" id="exportBtn">평가 내보내기</button>
  </div>
</div></header>
<main>
  <section class="hero">
    <div class="panel intro"><h2>동일한 표본을 세팅별로 나란히 비교</h2>
      <p id="introText"></p>
      <div class="seed">
        <span class="pill">랜덤 시드 <strong id="seedText"></strong></span>
        <span class="pill">표본 <strong id="sampleSizeText"></strong></span>
        <span class="pill">채점 <strong id="modeText"></strong></span>
        <span class="pill">행 정렬 검증 <strong>통과</strong></span>
        <span class="pill">키보드 <span class="kbd">←</span> <span class="kbd">→</span></span>
      </div>
    </div>
    <div class="panel metric-overview">
      <h3 id="metricTitle">전체 지표</h3><div id="metricTable"></div>
      <h3 id="sampleMetricTitle">표본 지표</h3><div id="sampleMetricTable"></div>
      <div id="elementSetNote"></div>
    </div>
  </section>
  <div class="navrow"><div class="sample-id" id="sampleId"></div>
    <div class="progress"><div id="progressBar"></div></div></div>
  <section class="panel context">
    <div class="context-head"><strong id="actionChipLabel">액션</strong>
      <span class="action-chip" id="actionChip"></span>
      <span class="mark-legend" id="markLegend"></span></div>
    {gt_block}
    <div class="context-grid">
      {_pane_html("A", pane_a)}
      {_pane_html("B", pane_b)}
    </div>
  </section>
  <section class="compare-grid" id="compareGrid"></section>
  <section class="panel notes"><h3>샘플 메모</h3>
    <textarea id="sampleNotes" placeholder="공통 오류 패턴, 특정 세팅의 장단점, 확인할 포인트 등을 기록하세요."></textarea>
    <div class="notes-foot"><span>메모와 등급은 브라우저에 자동 저장됩니다.</span><span id="saveState">저장됨</span></div>
  </section>
</main>
<div class="toast" id="toast"></div>
<script>
const DATA = {payload};
{SITE_JS}
</script>
</body>
</html>
"""


SITE_CSS = """
:root{
  --bg:#f4f6f9;--surface:#fff;--surface-2:#f8fafc;--ink:#172033;--muted:#667085;
  --line:#d9e0ea;--accent:#315efb;--accent-soft:#e9efff;--good:#11875d;--warn:#a15c00;--bad:#c13c3c;
  --code:#111827;--code-ink:#e5e7eb;--add:#123e2c;--del:#4b2227;
  --shadow:0 8px 28px rgba(15,23,42,.07);--radius:14px;
}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Apple SD Gothic Neo","Noto Sans KR",Segoe UI,sans-serif}
button,select,textarea,input{font:inherit}.topbar{position:sticky;top:0;z-index:50;background:rgba(244,246,249,.93);backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
.topbar-inner{width:min(2360px,calc(100vw - 24px));margin:auto;padding:14px 20px;display:flex;gap:16px;align-items:center;justify-content:space-between}.brand h1{font-size:18px;margin:0}.brand p{font-size:12px;color:var(--muted);margin:3px 0 0}.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.btn,.seg button,select{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:9px;padding:8px 11px;cursor:pointer}.btn:hover,.seg button:hover{border-color:#aeb9c8}.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}.btn:disabled{opacity:.45;cursor:not-allowed}.seg{display:flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--surface)}.seg button{border:0;border-radius:0;border-right:1px solid var(--line)}.seg button:last-child{border-right:0}.seg button.active{background:var(--accent);color:#fff}.kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:1px 5px;background:white;color:var(--muted)}
main{width:min(2360px,calc(100vw - 24px));margin:auto;padding:20px 0}.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:16px;margin-bottom:16px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}.intro{padding:20px}.intro h2{margin:0 0 7px;font-size:24px}.intro p{margin:0;color:var(--muted);line-height:1.6}.seed{display:flex;gap:14px;align-items:center;margin-top:14px;flex-wrap:wrap}.pill{display:inline-flex;gap:6px;align-items:center;padding:5px 9px;border-radius:999px;background:var(--surface-2);border:1px solid var(--line);font-size:12px;color:var(--muted)}.pill strong{color:var(--ink)}
.metric-overview{padding:14px;overflow:auto}.metric-overview h3{font-size:13px;margin:0 0 10px;color:var(--muted)}.metric-overview h3+div+h3{margin-top:14px}.metrics-table{width:100%;border-collapse:collapse;font-size:12px}.metrics-table th,.metrics-table td{text-align:right;padding:7px;border-bottom:1px solid var(--line);white-space:nowrap}.metrics-table th:first-child,.metrics-table td:first-child{text-align:left}.metrics-table tr:last-child td{border-bottom:0}.best{font-weight:800;color:var(--good)}
.navrow{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:14px 0}.sample-id{font-weight:750}.sample-id small{font-weight:400;color:var(--muted);margin-left:7px}.progress{height:5px;background:#e6eaf0;border-radius:99px;overflow:hidden;flex:1;max-width:500px}.progress>div{height:100%;background:var(--accent);transition:width .2s}
.context{margin-bottom:16px;overflow:visible;background:transparent;border:0;box-shadow:none}.context-head{display:flex;align-items:center;gap:10px;padding:14px 16px;background:var(--surface-2);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}.action-chip{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;background:#1f2937;color:white;border-radius:8px;padding:7px 10px;overflow-wrap:anywhere}
.gt-thought{margin-top:10px;padding:12px 16px;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}.gt-thought span{font-size:11px;color:var(--muted);font-weight:750}.gt-thought p{margin:5px 0 0;font-size:13px;line-height:1.6}
.context-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:14px}.context-cell{min-width:0;overflow:hidden;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}.section-title{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid var(--line);font-size:13px;font-weight:750}.context-tools{display:flex;gap:5px;padding:8px;border-bottom:1px solid var(--line);background:var(--surface-2);flex-wrap:wrap}.context-view{height:380px;overflow:auto;background:var(--code);color:var(--code-ink);position:relative}.wrap pre{white-space:pre-wrap;overflow-wrap:anywhere}
.compare-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start}.prediction-card{min-width:0;overflow:hidden;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}.card-head{padding:12px 13px;border-bottom:1px solid var(--line);display:flex;align-items:flex-start;justify-content:space-between;gap:8px}.setting-name{font-weight:800;overflow-wrap:anywhere}.score{font-size:20px;font-weight:850;letter-spacing:-.5px}.score.good{color:var(--good)}.score.mid{color:var(--warn)}.score.bad{color:var(--bad)}.subscore{font-size:11px;color:var(--muted);margin-top:2px}.badge{font-size:10px;border-radius:999px;padding:4px 7px;background:var(--surface-2);border:1px solid var(--line);white-space:nowrap}.badge.exact{background:#e9f8f1;color:var(--good);border-color:#bce8d5}.badge.bad{background:#fdecec;color:var(--bad);border-color:#f2c2c2}.badge.warn{background:#fdf4e3;color:var(--warn);border-color:#efd9ab}
.card-tools{display:flex;gap:5px;padding:8px;border-bottom:1px solid var(--line);background:var(--surface-2);flex-wrap:wrap}.mini{font-size:11px;padding:5px 7px;border:1px solid var(--line);border-radius:7px;background:white;cursor:pointer}.mini.active{background:#1f2937;color:#fff;border-color:#1f2937}.view{height:465px;overflow:auto;background:var(--code);color:var(--code-ink);position:relative}
.wireframe{height:100%;background:#202938;padding:10px;display:flex;align-items:flex-start;justify-content:center}.phone{position:relative;height:100%;width:auto;max-width:100%;background:#f8fafc;border:4px solid #0b1220;border-radius:18px;overflow:hidden;box-shadow:0 5px 16px rgba(0,0,0,.3)}.node{position:absolute;border:1px solid rgba(49,94,251,.42);background:rgba(49,94,251,.035);overflow:hidden;color:#172033;font:7px/1.1 ui-monospace,monospace;padding:1px}.node.button,.node.input{border-color:rgba(193,60,60,.65);background:rgba(193,60,60,.08)}.node.p{border-color:rgba(17,135,93,.5);background:rgba(17,135,93,.06)}.node-label{pointer-events:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.node.hit{border:2px solid #f59e0b;background:rgba(245,158,11,.25);z-index:5}.node.gt-hit{border:2px solid #10b981;background:rgba(16,185,129,.25);z-index:5}
.tree-wireframe{height:100%;overflow:auto;background:#202938;padding:10px}.tree-canvas{min-height:100%;padding:8px;background:#f8fafc;border:3px solid #0b1220;border-radius:14px}.tree-note{position:sticky;top:0;z-index:2;margin:-8px -8px 8px;padding:7px 9px;background:#e9efff;color:#315efb;border-bottom:1px solid #9bb0ff;font-size:10px;font-weight:750}.tree-node{min-height:24px;margin-top:3px;padding:4px 7px;border:1px solid rgba(49,94,251,.42);border-radius:5px;background:rgba(49,94,251,.06);color:#172033;font:10px/1.35 ui-monospace,monospace;display:flex;gap:8px;align-items:center;max-width:calc(100% - 8px);overflow:hidden}.tree-node strong{flex:none;color:#315efb}.tree-node span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tree-node.button,.tree-node.input{border-color:rgba(193,60,60,.65);background:rgba(193,60,60,.08)}.tree-node.p{border-color:rgba(17,135,93,.5);background:rgba(17,135,93,.06)}.tree-node.hit{border-color:#f59e0b;background:rgba(245,158,11,.22)}
.codeview{margin:0;padding:12px;font:10.5px/1.48 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre;min-height:100%}.diff-line{display:block;min-height:1.48em;padding:0 8px 0 4px}.diff-line.add{background:var(--add);color:#d1fae5}.diff-line.del{background:var(--del);color:#fee2e2}.diff-line.equal{color:#cbd5e1}.sign{display:inline-block;width:15px;user-select:none;color:#94a3b8}.collapse-line{display:block;background:#293446;color:#94a3b8;text-align:center;padding:3px;cursor:pointer}
.actionview{padding:12px;display:flex;flex-direction:column;gap:10px;min-height:100%}.act-block{border:1px solid #334155;border-radius:9px;overflow:hidden}.act-block h4{margin:0;padding:6px 9px;font-size:11px;background:#1f2937;color:#cbd5e1;font-weight:700}.act-block pre{margin:0;padding:9px;font:11px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap;overflow-wrap:anywhere;color:#e5e7eb}.act-block.gt{border-color:#11875d}.act-block.pred.ok{border-color:#11875d}.act-block.pred.ng{border-color:#c13c3c}
.fieldtable{width:100%;border-collapse:collapse;font:11px/1.5 ui-monospace,Menlo,monospace;color:#e5e7eb}.fieldtable td{padding:4px 8px;border-top:1px solid #334155;vertical-align:top;overflow-wrap:anywhere}.fieldtable td:first-child{color:#94a3b8;width:33%}.fieldtable tr.ng td{background:rgba(193,60,60,.18)}.fieldtable tr.ok td{background:rgba(17,135,93,.16)}
.thoughtview{padding:12px;font:12px/1.6 ui-sans-serif,system-ui;color:#e5e7eb;white-space:pre-wrap;overflow-wrap:anywhere}
.stats-row{padding:8px 11px;display:grid;grid-template-columns:repeat(3,1fr);gap:5px;border-top:1px solid var(--line);font-size:10px;color:var(--muted)}.stats-row strong{display:block;color:var(--ink);font-size:12px}
/* 변화 감사 — 요소 단위 색칠. mk-add/mod/del 은 "GT 가 요구한 변화", mk-hit/part/miss 는
   "예측이 맞혔나". 색은 셋(초록/주황/빨강)을 공유하지만 **키 공간이 달라** 이름을 나눈다. */
.node.mk-add,.node.mk-hit{border:2px solid #10b981;background:rgba(16,185,129,.28);z-index:6}
.node.mk-mod,.node.mk-part{border:2px solid #f59e0b;background:rgba(245,158,11,.28);z-index:6}
.node.mk-del,.node.mk-miss{border:2px solid #ef4444;background:rgba(239,68,68,.26);z-index:6}
.tree-node.mk-add,.tree-node.mk-hit{border-color:#10b981;background:rgba(16,185,129,.20)}
.tree-node.mk-mod,.tree-node.mk-part{border-color:#f59e0b;background:rgba(245,158,11,.20)}
.tree-node.mk-del,.tree-node.mk-miss{border-color:#ef4444;background:rgba(239,68,68,.18)}
.mark-legend{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:11px;color:var(--muted)}.mark-legend em{font-style:normal}.mark-legend b{font-weight:700;font-size:10px;border-radius:999px;padding:3px 8px;border:1px solid var(--line)}.mark-legend b.mk-add{background:rgba(16,185,129,.16);border-color:#10b981;color:#0b6b4a}.mark-legend b.mk-mod{background:rgba(245,158,11,.16);border-color:#f59e0b;color:#8a5300}.mark-legend b.mk-del{background:rgba(239,68,68,.14);border-color:#ef4444;color:#a12a2a}
.delta{margin-top:4px;font-size:11px;font-weight:750;border-radius:7px;padding:3px 7px;border:1px solid var(--line);white-space:nowrap}.delta.up{background:#e9f8f1;color:var(--good);border-color:#bce8d5}.delta.down{background:#fdecec;color:var(--bad);border-color:#f2c2c2}.delta.flat,.delta.na{background:var(--surface-2);color:var(--muted)}
.audit-row{padding:8px 11px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;border-top:1px solid var(--line)}.audit-row .ax{border:1px solid var(--line);border-radius:8px;padding:6px 8px;background:var(--surface-2);min-width:0}.audit-row .ax span{display:block;font-size:10px;color:var(--muted)}.audit-row .ax b{display:block;font:10.5px/1.4 ui-monospace,Menlo,monospace;color:var(--ink);overflow-wrap:anywhere}.audit-row .ax i{font-style:normal;font-weight:800;font-size:13px}.audit-row .ax.floor{background:#f3f4f6;border-style:dashed}.axis-note{padding:0 11px 9px;font-size:11px;line-height:1.5;color:var(--warn)}
.warnbox{margin-top:10px;padding:8px 10px;border:1px solid #efd9ab;background:#fdf4e3;color:var(--warn);border-radius:8px;font-size:11px;line-height:1.6}
.chg-ctx{font-size:11px;color:var(--muted);padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--surface-2);line-height:1.6}.chg-ctx small{margin-left:6px;opacity:.75}
.changeview{padding:11px;display:flex;flex-direction:column;gap:11px;min-height:100%;background:var(--surface)}.chg-sec h4{margin:0 0 6px;font-size:12px;color:var(--ink)}.chg-sec h4 small{font-weight:400;color:var(--muted)}.chg-item{display:flex;gap:7px;align-items:baseline;flex-wrap:wrap;padding:5px 7px;border-radius:7px;border:1px solid var(--line);margin-bottom:4px;font-size:11px;background:var(--surface-2)}.chg-item.ok{border-color:#10b981;background:rgba(16,185,129,.10)}.chg-item.part{border-color:#f59e0b;background:rgba(245,158,11,.10)}.chg-item.miss{border-color:#ef4444;background:rgba(239,68,68,.08)}.chg-item code{font:10.5px/1.45 ui-monospace,Menlo,monospace;color:var(--ink);overflow-wrap:anywhere;max-width:100%}.chg-k{font-size:9px;font-weight:800;border-radius:999px;padding:2px 6px;background:#1f2937;color:#fff;white-space:nowrap}.chg-k.k-A{background:#0b6b4a}.chg-k.k-M{background:#8a5300}.chg-k.k-D{background:#a12a2a}.chg-arrow{color:var(--muted)}.chg-sim{font:10px ui-monospace,Menlo,monospace;color:var(--muted);white-space:nowrap}.chg-v{margin-left:auto;color:var(--muted);white-space:nowrap}.chg-empty{padding:10px;border:1px dashed var(--line);border-radius:8px;color:var(--muted);font-size:11px;line-height:1.6}
.rating{padding:9px 11px;border-top:1px solid var(--line);display:flex;align-items:center;gap:6px}.rating span{font-size:11px;color:var(--muted);margin-right:auto}.rate-btn{width:27px;height:27px;border-radius:7px;border:1px solid var(--line);background:white;cursor:pointer;filter:grayscale(1)}.rate-btn.selected{filter:none;background:var(--accent-soft);border-color:#9bb0ff}
.notes{margin-top:16px;padding:14px}.notes h3{font-size:14px;margin:0 0 8px}.notes textarea{width:100%;min-height:92px;border:1px solid var(--line);border-radius:9px;padding:10px;resize:vertical}.notes-foot{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:7px}.empty{padding:20px;color:var(--muted)}
.toast{position:fixed;right:20px;bottom:20px;background:#111827;color:white;padding:10px 14px;border-radius:9px;box-shadow:var(--shadow);opacity:0;transform:translateY(8px);transition:.2s;pointer-events:none;z-index:100}.toast.show{opacity:1;transform:none}
@media(max-width:1100px){.compare-grid,.context-grid{grid-template-columns:1fr}.hero{grid-template-columns:1fr}}
@media(max-width:650px){.topbar-inner{width:100%;padding-left:12px;padding-right:12px;align-items:flex-start;flex-direction:column}.toolbar{justify-content:flex-start}main{width:100%;padding:12px}.intro h2{font-size:20px}.navrow{align-items:flex-start;flex-direction:column}.progress{width:100%;max-width:none}.view{height:520px}}
"""


SITE_JS = r"""
const MODE = DATA.task;
const IS_ACTION = MODE !== 'state';
const STORE_KEY = DATA.slug + '_annotations';
const state = {split: DATA.split_order[0], pos: 0, wrap: false};
const annotations = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
const views = {};
const contextViews = {A: 'wire', B: 'wire'};
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];

function esc(s){return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function toast(msg){const el=$('#toast');el.textContent=msg;el.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>el.classList.remove('show'),1400)}
function splitData(){return DATA.splits[state.split]}
function splitSettings(){return splitData().settings || DATA.settings}
function current(){return splitData().samples[state.pos]}
function annotationKey(){return `${state.split}:${current().index}`}
function saveAnnotations(){localStorage.setItem(STORE_KEY, JSON.stringify(annotations));$('#saveState').textContent='저장됨'}
function scoreClass(v){return v>=75?'good':v>=45?'mid':'bad'}
function num(v){return (v===null||v===undefined||Number.isNaN(v))?'—':(typeof v==='number'?(Number.isInteger(v)?v:v.toFixed(4)):v)}

/* ── 액션 표현 ────────────────────────────────────────────────── */
function actionType(a){return a ? (a.action || a.action_type || a.type || 'unknown') : 'unparsed'}
function actionText(a){
  if(!a) return '(파싱 실패)';
  const type = actionType(a);
  const rest = Object.entries(a).filter(([k])=>!['action','action_type','type'].includes(k))
    .map(([k,v])=>`${k}=${JSON.stringify(v)}`).join(' · ');
  return `${type}${rest?' · '+rest:''}`;
}
function actionFields(a){
  const out = {};
  if(!a) return out;
  for(const [k,v] of Object.entries(a)){
    if(k === 'params' && v && typeof v === 'object'){for(const [k2,v2] of Object.entries(v)) out[k2]=v2; continue}
    if(['action','action_type','type'].includes(k)) continue;
    out[k] = v;
  }
  return out;
}
function pretty(a, raw){
  if(!a) return raw || '(파싱 실패)';
  return JSON.stringify(a, null, 2);
}

/* ── 지표 표 ──────────────────────────────────────────────────── */
const LOWER_IS_BETTER = new Set(['no_bbox_n']);
/* 랭킹(초록 굵게)에서 뺄 열. floor 는 **지표가 아니라 눈금**이라 최대값을 "최고"로
   칠하면 정확히 반대로 읽힌다 (퇴화가 심할수록 바닥이 높다). n_* 는 분모라 성능이
   아니다. */
const NOT_RANKED = new Set(['total', 'no_bbox_n',
  'avg_change_f1_floor', 'avg_n_change_gt', 'avg_n_change_pred', 'change_f1_floor']);
function isRanked(k){return !NOT_RANKED.has(k) && !k.startsWith('n_')}
function metricsTable(rows, keys, fmt){
  const best = {};
  for(const k of keys){
    const vals = splitSettings().map(s => rows[s] ? rows[s][k] : undefined).filter(v => typeof v === 'number');
    if(!vals.length || !isRanked(k)) continue;
    best[k] = LOWER_IS_BETTER.has(k) ? Math.min(...vals) : Math.max(...vals);
  }
  let h = '<table class="metrics-table"><thead><tr><th>Setting</th>' +
    keys.map(k=>`<th>${esc(k)}</th>`).join('') + '</tr></thead><tbody>';
  for(const s of splitSettings()){
    const r = rows[s] || {};
    const missing = Object.keys(r).length === 0
      ? ' <small style="color:var(--muted)">(metric 파일 없음)</small>' : '';
    h += `<tr><td>${esc(DATA.setting_labels[s])}${missing}</td>` + keys.map(k=>{
      const v = r[k];
      const cls = (typeof v === 'number' && best[k] !== undefined && v === best[k]) ? 'best' : '';
      return `<td class="${cls}">${fmt(v)}</td>`;
    }).join('') + '</tr>';
  }
  return h + '</tbody></table>';
}
function renderMetrics(){
  const sp = splitData();
  $('#metricTable').innerHTML = metricsTable(sp.metrics, DATA.metric_keys, num);
  $('#metricTitle').textContent = `전체 ${sp.label} 지표 (모집단 ${sp.population.toLocaleString()}개 · 정본 metric 파일)`;
  const sKeys = Object.keys(sp.sample_metrics[splitSettings()[0]] || {});
  $('#sampleMetricTable').innerHTML = sKeys.length
    /* n_* 는 개수라 소수점을 붙이면 %로 오독된다 — 정수는 정수로 찍는다. */
    ? metricsTable(sp.sample_metrics, sKeys,
        v => typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(1)) : '—')
    : '<p class="empty">표본 없음</p>';
  $('#sampleMetricTitle').textContent = `표본 ${sp.samples.length}개 기준 (%, 같은 채점기)`;
  /* 두 표에 같은 이름의 열(addmod/change)이 나란히 놓이므로, 정의가 갈렸을 때
     말하지 않으면 그대로 비교당한다. */
  $('#elementSetNote').innerHTML = DATA.element_set_note
    ? `<div class="warnbox">${esc(DATA.element_set_note)}</div>` : '';
}

/* ── 와이어프레임 ─────────────────────────────────────────────── */
/* `_compare_site._el_key` 와 **같은 규칙**이어야 한다. pos 계열은 bounds, index
   계열은 index 가 자연 키다. 규칙이 갈리면 아무것도 안 칠해지거나(무해) 엉뚱한
   노드가 칠해진다(유해). 키가 겹치는 노드는 함께 칠해진다 — 색은 길잡이고 감사의
   정본은 Change 뷰의 요소 리스트다. */
function nodeKey(attrs){
  const m = DATA.match_mode==='pos'
    ? attrs.match(/bounds="([^"]*)"/) : attrs.match(/\bindex="([^"]*)"/);
  return m ? m[1] : '';
}
function markCls(marks, key){return (marks && marks.cls && key && marks.cls[key]) || ''}
function parseNodes(xml){
  const out=[];const re=/<([A-Za-z][\w:.-]*)\b([^>]*)>([^<]{0,80})?/g;let m,maxX=0,maxY=0;
  while((m=re.exec(xml))&&out.length<420){
    const attrs=m[2]||'';
    const bm=attrs.match(/bounds="\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"/);
    if(!bm)continue;
    let [x1,y1,x2,y2]=bm.slice(1).map(Number);
    if(x2<=x1||y2<=y1||x2<0||y2<0)continue;
    x1=Math.max(0,x1);y1=Math.max(0,y1);maxX=Math.max(maxX,x2);maxY=Math.max(maxY,y2);
    const tag=m[1].toLowerCase();
    const label=(attrs.match(/aria-label="([^"]*)"/)||attrs.match(/placeholder="([^"]*)"/)||attrs.match(/value="([^"]*)"/)||[])[1]||(m[3]||'').trim();
    out.push({tag,x1,y1,x2,y2,label,key:nodeKey(attrs)});
  }
  return {nodes:out,width:Math.max(maxX,1),height:Math.max(maxY,1)};
}
function structureWireframe(xml, marks){
  const shell=document.createElement('div');shell.className='tree-wireframe';
  const canvas=document.createElement('div');canvas.className='tree-canvas';
  const note=document.createElement('div');note.className='tree-note';
  note.textContent='bounds 정보 없음 · XML 계층 구조 기반 와이어프레임';
  canvas.appendChild(note);
  const hits = new Set((marks&&marks.indices)||[]);
  let shown=0;
  for(const line of xml.split('\n')){
    if(shown>=520)break;
    const m=line.match(/^(\s*)<\s*(?!\/)([A-Za-z][\w:.-]*)\b([^>]*)>([^<]{0,100})?/);
    if(!m)continue;
    const depth=Math.min(16,Math.floor(m[1].replace(/\t/g,'  ').length/2));
    const attrs=m[3]||'';const tag=m[2].toLowerCase();
    const idx=(attrs.match(/\bindex="([^"]+)"/)||[])[1];
    const label=(attrs.match(/aria-label="([^"]*)"/)||attrs.match(/placeholder="([^"]*)"/)||attrs.match(/value="([^"]*)"/)||[])[1]||(m[4]||'').trim();
    const node=document.createElement('div');
    const mc=markCls(marks, nodeKey(attrs));
    node.className=`tree-node ${tag}`
      + (idx!==undefined && hits.has(String(idx)) ? ' hit' : '') + (mc?' '+mc:'');
    node.style.marginLeft=`${depth*18}px`;
    const name=document.createElement('strong');
    name.textContent=`<${tag}${idx!==undefined?' #'+idx:''}>`;
    node.appendChild(name);
    if(label){const text=document.createElement('span');text.textContent=label;node.appendChild(text)}
    canvas.appendChild(node);shown++;
  }
  if(!shown){const empty=document.createElement('div');empty.className='empty';empty.textContent='표시할 XML 노드를 찾지 못했습니다.';canvas.appendChild(empty)}
  shell.appendChild(canvas);return shell;
}
function wireframe(xml, marks){
  const parsed=parseNodes(xml);
  if(!parsed.nodes.length)return structureWireframe(xml, marks);
  const {nodes,width,height}=parsed;
  const phone=document.createElement('div');phone.className='phone';
  phone.style.aspectRatio=`${width} / ${height}`;
  nodes.sort((a,b)=>((b.x2-b.x1)*(b.y2-b.y1))-((a.x2-a.x1)*(a.y2-a.y1)));
  for(const n of nodes){
    const d=document.createElement('div');d.className=`node ${n.tag}`;
    const mc=markCls(marks, n.key); if(mc) d.className+=' '+mc;
    d.style.left=`${n.x1/width*100}%`;d.style.top=`${n.y1/height*100}%`;
    d.style.width=`${(n.x2-n.x1)/width*100}%`;d.style.height=`${(n.y2-n.y1)/height*100}%`;
    d.title=`<${n.tag}> ${n.label||''} [${n.x1},${n.y1}][${n.x2},${n.y2}]`;
    if(n.label&&(['button','input','p','textview'].includes(n.tag)||((n.x2-n.x1)>width*.2&&(n.y2-n.y1)>height*.025))){
      const l=document.createElement('div');l.className='node-label';l.textContent=n.label;d.appendChild(l);
    }
    phone.appendChild(d);
  }
  for(const pt of (marks&&marks.points)||[]){
    const d=document.createElement('div');d.className='node '+(pt.kind==='gt'?'gt-hit':'hit');
    const r=Math.max(width,height)*0.012;
    d.style.left=`${(pt.x-r)/width*100}%`;d.style.top=`${(pt.y-r)/height*100}%`;
    d.style.width=`${2*r/width*100}%`;d.style.height=`${2*r/height*100}%`;
    d.style.borderRadius='50%';d.title=`${pt.kind==='gt'?'GT':'예측'} [${pt.x},${pt.y}]`;
    phone.appendChild(d);
  }
  const shell=document.createElement('div');shell.className='wireframe';shell.appendChild(phone);return shell;
}

/* ── diff / raw ───────────────────────────────────────────────── */
function lcsDiff(a,b){
  const A=a.split('\n'),B=b.split('\n'),n=A.length,m=B.length;
  const dp=Array.from({length:n+1},()=>new Uint16Array(m+1));
  for(let i=n-1;i>=0;i--)for(let j=m-1;j>=0;j--)dp[i][j]=A[i]===B[j]?dp[i+1][j+1]+1:Math.max(dp[i+1][j],dp[i][j+1]);
  const out=[];let i=0,j=0;
  while(i<n&&j<m){
    if(A[i]===B[j]){out.push({t:'equal',s:A[i++]});j++}
    else if(dp[i+1][j]>=dp[i][j+1])out.push({t:'del',s:A[i++]});
    else out.push({t:'add',s:B[j++]});
  }
  while(i<n)out.push({t:'del',s:A[i++]});
  while(j<m)out.push({t:'add',s:B[j++]});
  return out;
}
function diffView(label,pred){
  const pre=document.createElement('pre');pre.className='codeview';
  const rows=lcsDiff(label,pred);
  function makeLine(r){
    const line=document.createElement('span');line.className=`diff-line ${r.t}`;
    const sign=document.createElement('span');sign.className='sign';
    sign.textContent=r.t==='add'?'+':r.t==='del'?'-':' ';
    line.appendChild(sign);line.appendChild(document.createTextNode(r.s));return line;
  }
  const append=r=>pre.appendChild(makeLine(r));
  let i=0;
  while(i<rows.length){
    if(rows[i].t==='equal'){
      let j=i;while(j<rows.length&&rows[j].t==='equal')j++;
      const len=j-i;
      if(len>14){
        for(let k=i;k<i+5;k++)append(rows[k]);
        const fold=document.createElement('span');fold.className='collapse-line';
        fold.textContent=`… 동일한 ${len-10}줄 접힘 (클릭하여 펼치기) …`;
        const from=i+5, to=j-5;
        fold.onclick=()=>{const frag=document.createDocumentFragment();for(let k=from;k<to;k++)frag.appendChild(makeLine(rows[k]));fold.replaceWith(frag)};
        pre.appendChild(fold);
        for(let k=j-5;k<j;k++)append(rows[k]);
      } else for(let k=i;k<j;k++)append(rows[k]);
      i=j;
    } else {append(rows[i]);i++}
  }
  return pre;
}
function rawView(text){const pre=document.createElement('pre');pre.className='codeview';pre.textContent=text;return pre}
function textView(text){const d=document.createElement('div');d.className='thoughtview';d.textContent=text||'(없음)';return d}

/* ── context pane ─────────────────────────────────────────────── */
function paneSource(pane){
  const s=current();
  if(MODE==='stage2') return pane==='A'
    ? `Task Instruction:\n${s.instruction}\n\nAction History:\n${s.history||'None'}`
    : s.current_state;
  if(MODE==='action') return pane==='A' ? s.current_state : s.next_state;
  return pane==='A' ? s.current_state : s.label;
}
function paneMarks(pane){
  if(MODE==='state'){
    /* pane A = current("무엇이 사라져야 하나") / pane B = GT next state("무엇이
       생기고 바뀌나"). 두 dict 의 키 공간이 다르므로 빌드 시점에 따로 싣고 여기서
       고른다 — 섞어도 화면은 칠해져서 정상으로 보이기 때문에 조용히 틀린다. */
    const m = current().change_marks;
    return m ? {cls: pane==='A' ? m.cur : m.gt} : null;
  }
  const isCurrent = (MODE==='stage2' && pane==='B') || (MODE==='action' && pane==='A');
  if(!isCurrent) return null;
  const s=current(), f=actionFields(s.gt_action), pts=[], idxs=[];
  for(const key of ['coordinate','coordinate1','coordinate2']){
    const v=f[key];
    if(Array.isArray(v)&&v.length>=2) pts.push({x:+v[0],y:+v[1],kind:'gt'});
  }
  if(f.index!==undefined) idxs.push(String(f.index));
  return {points:pts, indices:idxs};
}
function renderContextView(pane){
  const root=$(`#pane-${pane}`);
  const canWire=root.dataset.wire==='1';
  const mode=canWire?(contextViews[pane]||'wire'):'raw';
  const text=paneSource(pane);
  root.replaceChildren(mode==='wire'?wireframe(text,paneMarks(pane)):rawView(text));
  $$(`[data-context-view][data-target="${pane}"]`).forEach(b=>b.classList.toggle('active',b.dataset.contextView===mode));
}

/* ── 변화 감사 (Change 뷰) ────────────────────────────────────────
   원 요구는 "점수를 눈으로 확인하고 싶다"이지 "숫자를 한 번 더 보고 싶다"가 아니다
   (예전에 base 와 epoch3 가 같은 화면을 냈는데 f1 이 갈린 이유를 손으로 확인해야
   했다). 그래서 이 뷰는 값이 아니라 **분자에 들어간 요소와 빠진 요소**를 나열한다. */
const CHG_KIND={A:'ADDED',M:'MODIFIED',D:'DELETED'};
function fmtPct(v){return (v===null||v===undefined)?'—':(v*100).toFixed(1)+'%'}
function elLabel(e){return e?`<${e.g}${e.k?' '+e.k:''}>${e.x?' '+e.x:''}`:'—'}

/* `—` 하나로 뭉개면 안 되는 세 가지를 가른다: 파싱 실패 / 변화 없는 행(복사가 정답) /
   진짜 0.0. 셋을 같은 칸으로 렌더하면 이 화면의 목적이 그 자리에서 깨진다. */
function axisNote(st){
  if(st.parse_fail) return '⚠️ 예측에서 요소를 하나도 추출하지 못했습니다 (parse_fail) '
    + '— 정본은 이 행의 change 축을 0.0(실패)으로, addmod 축을 미정의로 셉니다.';
  if(st.n_change_gt===0) return 'ℹ️ 이 행은 GT 가 current 와 같습니다 (변화 0개). '
    + '복사가 정답이라 change/addmod 축이 정의되지 않고, 대신 no_change_acc = '
    + (st.no_change_acc===null||st.no_change_acc===undefined?'—':st.no_change_acc)
    + ' 가 이 구간을 잽니다.';
  if(st.addmod_recall===null && st.change_f1_strict!==null)
    return 'ℹ️ GT 의 변화가 DELETED 뿐이라 addmod 축(분모 = ADDED+MODIFIED)이 정의되지 않습니다.';
  return '';
}
function axisCell(label, frac, val, cls){
  return `<div class="ax ${cls||''}"><span>${label}</span><b>${frac}</b><i>${val}</i></div>`;
}
/* 카드 대표 점수를 Hungarian F1 하나로 두지 않기 위한 블록 (AGENTS.md 13b).
   색은 **값이 아니라 Δ 의 부호**로 정한다 — strict 값으로 색을 매기면 0.114(학습)와
   0.258(바닥)이 똑같이 "낮음"으로 보여 "바닥에 졌다"는 유일한 판별 정보가 사라진다. */
function changeHeadline(st){
  if(st.change_f1_strict===null||st.change_f1_strict===undefined)
    return '<div class="delta na">change 축 미정의</div>';
  const d=st.change_f1_strict-st.change_f1_floor;
  const cls=d>0?'up':d<0?'down':'flat';
  return `<div class="delta ${cls}" title="change_f1_strict − change_f1_floor. 이 축의 바닥은 0 이 아니라 floor 다.">`
    +`change ${fmtPct(st.change_f1_strict)} · 바닥 ${fmtPct(st.change_f1_floor)}`
    +` · Δ ${d>0?'+':''}${(d*100).toFixed(1)}p</div>`;
}
function auditRow(st){
  const note=axisNote(st);
  const gtN=`GT ${st.n_change_gt} · 예측 ${st.n_change_pred}`;
  return `<div class="audit-row">
    ${axisCell('addmod recall', `${st.n_addmod_hit} / ${st.n_addmod_gt}`, fmtPct(st.addmod_recall))}
    ${axisCell('change F1 (strict)', `hit ${st.n_change_hit_strict} · ${gtN}`, fmtPct(st.change_f1_strict))}
    ${axisCell('change F1 (loose)', `hit ${st.n_change_hit_loose} · ${gtN}`, fmtPct(st.change_f1_loose))}
    ${axisCell('change F1 바닥', '퇴화 예측의 상한', fmtPct(st.change_f1_floor), 'floor')}
  </div>` + (note?`<div class="axis-note">${esc(note)}</div>`:'');
}
function chgVerdict(it, side){
  if(it.h==='s') return 'hit · 내용까지 일치';
  if(it.h==='l') return '자리만 맞음 · 내용 불일치로 strict 탈락';
  if(it.w==='copy') return 'miss · 예측이 이 자리를 current 그대로 베꼈다';
  if(it.w==='nochange') return 'spurious · GT 는 이 자리를 바꾸지 않았다';
  return side==='gt' ? 'miss · 예측에 대응 요소가 없다' : 'spurious · GT 에 대응 요소가 없다';
}
function chgItem(it, side){
  const cls=it.h==='s'?'ok':it.h==='l'?'part':'miss';
  let h=`<div class="chg-item ${cls}"><span class="chg-k k-${it.t}">${CHG_KIND[it.t]}</span>`
    + `<code>${esc(elLabel(it))}</code>`;
  if(it.m) h+=`<span class="chg-arrow">↔</span><code>${esc(elLabel(it.m))}</code>`;
  if(it.s!==undefined) h+=`<span class="chg-sim">text_sim ${it.s.toFixed(3)} `
    + `${it.s>=DATA.change_tau?'≥':'&lt;'} τ ${DATA.change_tau}</span>`;
  return h+`<span class="chg-v">${chgVerdict(it, side)}</span></div>`;
}
function chgSection(title, items, more, side, empty){
  const cut = more?` <small>(표시 ${items.length}개 · ${more}개 생략 — 개수는 정본 값)</small>`:'';
  return `<div class="chg-sec"><h4>${title}${cut}</h4>`
    + (items.length ? items.map(it=>chgItem(it,side)).join('') : `<div class="chg-empty">${empty}</div>`)
    + '</div>';
}
function changeView(st){
  const a=st.audit||{}, d=document.createElement('div');
  d.className='changeview';
  if(a.na){
    d.innerHTML='<div class="chg-empty">예측·GT·current 중 하나에서 요소 추출이 예외로 실패했습니다 '
      + '— 정본도 이 행의 state-diff 축을 전부 미정의로 냅니다.</div>';
    return d;
  }
  const ce = st.copy_excess;
  d.innerHTML = `<div class="chg-ctx">요소 수 — current ${st.n_cur_el} · GT ${st.n_gt_el} · 예측 ${st.n_pred_el}`
      + ` <span title="예측이 current 와 겹친 비율 − GT 가 current 와 겹친 비율. 큰 양수 = 바뀌었어야 할 자리까지 베꼈다.">`
      + `· copy_excess ${ce===null||ce===undefined?'—':(ce>0?'+':'')+ce.toFixed(3)}</span>`
      + (st.no_change_acc===null||st.no_change_acc===undefined?'':` · no_change_acc ${st.no_change_acc}`)
      + ` <small>점수 요약은 카드 하단에 항상 보입니다</small></div>`
    + chgSection(`GT 가 요구한 변화 ${st.n_change_gt}개 — 예측이 맞혔나`,
        a.gt||[], a.gt_more, 'gt', 'GT 가 current 와 같다 (요구된 변화 없음).')
    + chgSection(`예측이 주장한 변화 ${st.n_change_pred}개 — 진짜였나`,
        a.pd||[], a.pd_more, 'pd', '예측이 아무 변화도 주장하지 않았다 (= 복사 또는 생성 실패).');
  return d;
}
/* 예측 카드 와이어프레임용 마크. DELETED 주장은 **current 요소 키**라 예측 XML 에
   존재하지 않는다 — 키 공간이 달라 여기 섞으면 엉뚱한 노드가 칠해진다. */
function predMarks(st){
  const a=st.audit;
  if(!a||!a.pd||!a.pd.length) return null;
  const cls={};
  for(const it of a.pd){
    if(it.t==='D'||!it.k) continue;
    cls[it.k]= it.h==='s'?'mk-hit':it.h==='l'?'mk-part':'mk-miss';
  }
  return {cls};
}

/* ── 카드 ─────────────────────────────────────────────────────── */
function stateCard(setting, pred, s){
  const st=pred.stats;
  const el=document.createElement('article');
  el.className='prediction-card';el.dataset.setting=setting;
  el.innerHTML=`<div class="card-head">
      <div><div class="setting-name">${esc(DATA.setting_labels[setting])}</div>
        <div class="subscore">Hungarian F1 + change 축 · 정본 채점기</div></div>
      <div style="text-align:right"><div class="score ${scoreClass(st.f1)}">${st.f1.toFixed(1)}%</div>
        ${st.exact?'<span class="badge exact">EXACT</span>':'<span class="badge">비정확 일치</span>'}
        ${changeHeadline(st)}</div>
    </div>
    <div class="card-tools">
      <button class="mini" data-view="wire">와이어프레임</button>
      <button class="mini active" data-view="diff">Diff</button>
      <button class="mini" data-view="change">Change</button>
      <button class="mini" data-view="raw">원문</button>
      <button class="mini" data-copy-pred>복사</button>
    </div>
    <div class="view"></div>
    <div class="stats-row">
      <div><span>EA</span><strong>${st.ea.toFixed(1)}%</strong></div>
      <div><span>text</span><strong>${st.text.toFixed(1)}%</strong></div>
      <div><span>${esc(st.pos_label)}</span><strong>${st.pos.toFixed(1)}%</strong></div>
      <div><span>BLEU-4</span><strong>${st.bleu4.toFixed(1)}</strong></div>
      <div><span>ROUGE-L</span><strong>${st.rouge_l.toFixed(1)}</strong></div>
      <div><span>줄 수</span><strong>${st.pred_lines} / ${st.label_lines}</strong></div>
    </div>
    ${auditRow(st)}`;
  const v=el.querySelector('.view');
  const show=mode=>{
    views[setting]=mode;
    v.replaceChildren(
      mode==='wire'?wireframe(pred.text,predMarks(st)):
      mode==='change'?changeView(st):
      mode==='raw'?rawView(pred.text):diffView(s.label,pred.text));
    el.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===mode));
  };
  el.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>show(b.dataset.view));
  el.querySelector('[data-copy-pred]').onclick=()=>navigator.clipboard.writeText(pred.text).then(()=>toast('예측을 복사했습니다'));
  show(views[setting]||'diff');
  return el;
}

function verdict(st){
  if(!st.parsed) return {text:'파싱 실패', cls:'bad', score:'✗', badge:'badge bad'};
  if(st.step_correct) return {text:'STEP 정답', cls:'good', score:'✓', badge:'badge exact'};
  if(st.type_correct) return {text:'타입만 일치', cls:'mid', score:'△', badge:'badge warn'};
  return {text:'오답', cls:'bad', score:'✗', badge:'badge bad'};
}
function fieldTable(gt, pred){
  const gf=actionFields(gt), pf=actionFields(pred);
  const keys=[...new Set([...Object.keys(gf), ...Object.keys(pf)])];
  const rows=[['action', actionType(gt), actionType(pred)],
              ...keys.map(k=>[k, gf[k], pf[k]])];
  let h='<table class="fieldtable"><tbody>';
  for(const [k,a,b] of rows){
    const same=JSON.stringify(a)===JSON.stringify(b);
    h+=`<tr class="${same?'ok':'ng'}"><td>${esc(k)}</td><td>${esc(JSON.stringify(a)??'—')}</td><td>${esc(JSON.stringify(b)??'—')}</td></tr>`;
  }
  return h+'</tbody></table>';
}
function actionView(s, pred){
  const wrap=document.createElement('div');wrap.className='actionview';
  const st=pred.stats;
  wrap.innerHTML=`
    <div class="act-block gt"><h4>Ground Truth</h4><pre>${esc(pretty(s.gt_action, s.gt_action_raw))}</pre></div>
    <div class="act-block pred ${st.step_correct?'ok':'ng'}"><h4>예측</h4><pre>${esc(pretty(pred.action, pred.text))}</pre></div>
    <div class="act-block"><h4>필드 대조 (GT / 예측)</h4>${fieldTable(s.gt_action, pred.action)}</div>`;
  return wrap;
}
function actionCard(setting, pred, s){
  const st=pred.stats, vd=verdict(st);
  const el=document.createElement('article');
  el.className='prediction-card';el.dataset.setting=setting;
  const thoughtScore = MODE==='stage2'
    ? `<div><span>thought ROUGE-L</span><strong>${(st.thought_rouge_l??0).toFixed(1)}%</strong></div>` : '';
  const tools = MODE==='stage2'
    ? `<button class="mini active" data-view="action">액션</button>
       <button class="mini" data-view="thought">Thought</button>
       <button class="mini" data-view="tdiff">Thought diff</button>
       <button class="mini" data-view="raw">원문</button>`
    : `<button class="mini active" data-view="action">액션</button>
       <button class="mini" data-view="raw">원문</button>`;
  el.innerHTML=`<div class="card-head">
      <div><div class="setting-name">${esc(DATA.setting_labels[setting])}</div>
        <div class="subscore">${esc(DATA.coord_mode==='xy'?'xy 채점 (bbox 포함)':'index 채점')} · 정본 채점기</div></div>
      <div style="text-align:right"><div class="score ${vd.cls}">${vd.score}</div>
        <span class="${vd.badge}">${vd.text}</span></div>
    </div>
    <div class="card-tools">${tools}<button class="mini" data-copy-pred>복사</button></div>
    <div class="view"></div>
    <div class="stats-row">
      <div><span>parse</span><strong>${st.parsed?'O':'X'}</strong></div>
      <div><span>type</span><strong>${st.type_correct?'O':'X'}</strong></div>
      <div><span>step</span><strong>${st.step_correct?'O':'X'}</strong></div>
      <div><span>대조 필드</span><strong>${esc(st.field||'—')}</strong></div>
      <div><span>no_bbox</span><strong>${st.no_bbox?'O':'—'}</strong></div>
      ${thoughtScore}
    </div>`;
  const v=el.querySelector('.view');
  const show=mode=>{
    views[setting]=mode;
    let node;
    if(mode==='raw') node=rawView(pred.text);
    else if(mode==='thought') node=textView(pred.thought);
    else if(mode==='tdiff') node=diffView(s.gt_thought||'', pred.thought||'');
    else node=actionView(s, pred);
    v.replaceChildren(node);
    el.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===mode));
  };
  el.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>show(b.dataset.view));
  el.querySelector('[data-copy-pred]').onclick=()=>navigator.clipboard.writeText(pred.text).then(()=>toast('예측을 복사했습니다'));
  show(views[setting]||'action');
  return el;
}
function card(setting, s){
  const pred=s.predictions[setting];
  const el=IS_ACTION?actionCard(setting,pred,s):stateCard(setting,pred,s);
  const rating=document.createElement('div');rating.className='rating';
  rating.innerHTML=`<span>정성 등급</span>
    <button class="rate-btn" data-rate="good" title="좋음">👍</button>
    <button class="rate-btn" data-rate="mid" title="보통">😐</button>
    <button class="rate-btn" data-rate="bad" title="나쁨">👎</button>`;
  el.appendChild(rating);
  const key=annotationKey();
  annotations[key] ??= {notes:'', ratings:{}};
  rating.querySelectorAll('[data-rate]').forEach(b=>{
    b.classList.toggle('selected', annotations[key].ratings[setting]===b.dataset.rate);
    b.onclick=()=>{
      annotations[key].ratings[setting]=b.dataset.rate;
      rating.querySelectorAll('[data-rate]').forEach(x=>x.classList.toggle('selected',x===b));
      saveAnnotations();
    };
  });
  return el;
}

/* ── 렌더 ─────────────────────────────────────────────────────── */
function render(){
  const sp=splitData(), s=current();
  $('#seedText').textContent=DATA.seed;
  $('#sampleSizeText').textContent=`분할별 ${DATA.sample_size}개`;
  $('#modeText').textContent=IS_ACTION?`coord-mode ${DATA.coord_mode}`:`match-mode ${DATA.match_mode}`;
  $('#introText').textContent=IS_ACTION
    ? '동일한 입력에 대해 각 세팅이 내놓은 액션을 정본 채점기 판정과 함께 비교합니다. GT 액션은 현재 상태 와이어프레임 위에 표시됩니다.'
    : '현재 UI 상태와 액션을 기준으로 정답 next state 와 각 체크포인트의 예측을 비교합니다. 와이어프레임, 정답 대비 line diff, Hungarian 계열 점수에 더해 카드의 Change 탭에서 addmod/change 축의 분자·분모를 요소 단위로 감사할 수 있습니다.';
  $('#sampleId').innerHTML=`${sp.label} 샘플 ${state.pos+1} / ${sp.samples.length}` +
    `<small>원본 행 #${s.index} · 모집단 ${sp.population.toLocaleString()}개</small>`;
  $('#progressBar').style.width=`${(state.pos+1)/sp.samples.length*100}%`;
  $('#actionChipLabel').textContent=MODE==='state'?'실행 액션 (입력)':'정답 액션 (GT)';
  $('#actionChip').textContent=actionText(MODE==='state'?s.action:s.gt_action);
  if(MODE==='state') $('#markLegend').innerHTML = DATA.marks_usable
    ? '<em>아래 두 화면의 색 = GT 기준 "바뀌어야 할 자리"</em>'
      + '<b class="mk-add">ADDED (오른쪽)</b><b class="mk-mod">MODIFIED (오른쪽)</b>'
      + '<b class="mk-del">DELETED (왼쪽)</b>'
    /* 색이 없는 것을 "변화가 없다"로 읽으면 안 된다 — 키가 없어 못 칠하는 것이다. */
    : '<em>⚠️ 이 EXP 의 XML 에는 요소 식별 속성('
      + (DATA.match_mode==='pos'?'bounds':'index')
      + ')이 없어 와이어프레임 색칠을 쓸 수 없습니다. 카드의 <b>Change</b> 탭에서 '
      + '요소 단위 감사를 보세요.</em>';
  if(MODE==='stage2') $('#gtThought').textContent=s.gt_thought||'(없음)';
  renderContextView('A');renderContextView('B');
  $('#sampleSelect').value=String(state.pos);
  $('#prevBtn').disabled=state.pos===0;
  $('#nextBtn').disabled=state.pos===sp.samples.length-1;
  $('#compareGrid').replaceChildren(...splitSettings().map(setting=>card(setting,s)));
  const key=annotationKey();
  annotations[key] ??= {notes:'', ratings:{}};
  $('#sampleNotes').value=annotations[key].notes||'';
  renderMetrics();
  window.scrollTo({top:0,behavior:'instant'});
}
function setupSelect(){
  const sel=$('#sampleSelect');
  sel.replaceChildren(...splitData().samples.map((s,i)=>{
    const o=document.createElement('option');o.value=i;
    o.textContent=`${i+1} / ${splitData().samples.length} · #${s.index}`;return o;
  }));
  sel.value=state.pos;
}
$$('#splitSeg button').forEach(b=>b.onclick=()=>{
  state.split=b.dataset.split;state.pos=0;
  $$('#splitSeg button').forEach(x=>x.classList.toggle('active',x===b));
  setupSelect();render();
});
$('#prevBtn').onclick=()=>{if(state.pos>0){state.pos--;render()}};
$('#nextBtn').onclick=()=>{if(state.pos<splitData().samples.length-1){state.pos++;render()}};
$('#sampleSelect').onchange=e=>{state.pos=+e.target.value;render()};
$('#wrapBtn').onclick=()=>{state.wrap=!state.wrap;document.body.classList.toggle('wrap',state.wrap);$('#wrapBtn').classList.toggle('primary',state.wrap)};
$$('[data-context-view]').forEach(b=>b.onclick=()=>{contextViews[b.dataset.target]=b.dataset.contextView;renderContextView(b.dataset.target)});
$$('[data-copy]').forEach(b=>b.onclick=()=>navigator.clipboard.writeText(paneSource(b.dataset.copy)).then(()=>toast('복사했습니다')));
$('#sampleNotes').addEventListener('input',e=>{
  const key=annotationKey();
  annotations[key] ??= {notes:'', ratings:{}};
  annotations[key].notes=e.target.value;
  $('#saveState').textContent='저장 중…';
  clearTimeout(window.saveT);window.saveT=setTimeout(saveAnnotations,350);
});
$('#exportBtn').onclick=()=>{
  const payload={site:DATA.slug,exp:DATA.exp,stage:DATA.stage,task:DATA.task,seed:DATA.seed,
                 exported_at:new Date().toISOString(),annotations};
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download=`${DATA.slug}_${DATA.seed}.json`;a.click();URL.revokeObjectURL(a.href);
  toast('평가 JSON을 내보냈습니다');
};
document.addEventListener('keydown',e=>{
  if(['TEXTAREA','INPUT','SELECT'].includes(document.activeElement.tagName))return;
  if(e.key==='ArrowLeft')$('#prevBtn').click();
  if(e.key==='ArrowRight')$('#nextBtn').click();
});
setupSelect();render();
"""


# ── _compare/ 루트 인덱스 ────────────────────────────────────────────────
def write_root_index(compare_root: Path) -> Path | None:
    """outputs/_compare/ 아래 생성된 사이트 목록 페이지. 없으면 만들지 않는다."""
    sites = sorted(
        p for p in compare_root.glob("*_compare") if (p / "index.html").is_file()
    )
    if not sites:
        return None
    rows = "".join(
        f'<li><a href="{p.name}/index.html">{p.name}</a>'
        f" <small>({(p / 'index.html').stat().st_size / 1024 / 1024:.1f} MB)</small>"
        f' · <a href="{p.name}/README.md">README</a></li>'
        for p in sites
    )
    doc = (
        "<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>"
        "<title>EXP 정성 비교 사이트</title>"
        "<style>body{font-family:ui-sans-serif,system-ui,'Noto Sans KR',sans-serif;"
        "margin:40px auto;max-width:820px;color:#172033}h1{font-size:20px}"
        "li{margin:7px 0;line-height:1.6}small{color:#667085}"
        "code{background:#f4f6f9;padding:2px 5px;border-radius:5px}</style></head><body>"
        "<h1>EXP 정성 비교 사이트</h1>"
        "<p><code>python scripts/eval_viewer.py --site --include EXP:MODEL ...</code> 산출물입니다.</p>"
        f"<ul>{rows}</ul></body></html>"
    )
    out = compare_root / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out
