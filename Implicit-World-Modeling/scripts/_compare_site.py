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


def score_state_row(pred_text: str, gt_text: str, match_mode: str) -> dict:
    _hungarian_eval._lazy_deps()  # 미초기화면 전 행 0점 — 호출 순서에 기대지 않는다
    hung = _hungarian_eval.compute_hungarian_acc(pred_text, gt_text, match_mode)
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
                rec["predictions"] = {
                    st["id"]: {
                        "text": preds[st["id"]][i].get("predict", ""),
                        "stats": score_state_row(
                            preds[st["id"]][i].get("predict", ""),
                            gt_label,
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
            out[st["id"]] = {
                "hung_f1": avg("f1"),
                "hung_ea": avg("ea"),
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
        "카드에 표시되는 점수는 전부 정본 채점기 산출값이라 위쪽 aggregate 표와 정의가 같다.",
    ]
    if task == "state":
        lines += [
            f"- `scripts/_hungarian_eval.py::compute_hungarian_acc` "
            f"(`--match-mode {data['match_mode']}`) → F1 / EA / prec / rec / text / 위치",
            "- 같은 모듈의 `calc_bleu` · `calc_rouge_l` → BLEU-4 / ROUGE-L",
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
    </div>
  </section>
  <div class="navrow"><div class="sample-id" id="sampleId"></div>
    <div class="progress"><div id="progressBar"></div></div></div>
  <section class="panel context">
    <div class="context-head"><strong id="actionChipLabel">액션</strong>
      <span class="action-chip" id="actionChip"></span></div>
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
const NOT_RANKED = new Set(['total', 'no_bbox_n']);
function metricsTable(rows, keys, fmt){
  const best = {};
  for(const k of keys){
    const vals = splitSettings().map(s => rows[s] ? rows[s][k] : undefined).filter(v => typeof v === 'number');
    if(!vals.length || NOT_RANKED.has(k)) continue;
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
    ? metricsTable(sp.sample_metrics, sKeys, v => typeof v === 'number' ? v.toFixed(1) : '—')
    : '<p class="empty">표본 없음</p>';
  $('#sampleMetricTitle').textContent = `표본 ${sp.samples.length}개 기준 (%, 같은 채점기)`;
}

/* ── 와이어프레임 ─────────────────────────────────────────────── */
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
    out.push({tag,x1,y1,x2,y2,label});
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
    node.className=`tree-node ${tag}` + (idx!==undefined && hits.has(String(idx)) ? ' hit' : '');
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
  if(MODE==='state') return null;
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

/* ── 카드 ─────────────────────────────────────────────────────── */
function stateCard(setting, pred, s){
  const st=pred.stats;
  const el=document.createElement('article');
  el.className='prediction-card';el.dataset.setting=setting;
  el.innerHTML=`<div class="card-head">
      <div><div class="setting-name">${esc(DATA.setting_labels[setting])}</div>
        <div class="subscore">Hungarian F1 · 정본 채점기</div></div>
      <div style="text-align:right"><div class="score ${scoreClass(st.f1)}">${st.f1.toFixed(1)}%</div>
        ${st.exact?'<span class="badge exact">EXACT</span>':'<span class="badge">비정확 일치</span>'}</div>
    </div>
    <div class="card-tools">
      <button class="mini" data-view="wire">와이어프레임</button>
      <button class="mini active" data-view="diff">Diff</button>
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
    </div>`;
  const v=el.querySelector('.view');
  const show=mode=>{
    views[setting]=mode;
    v.replaceChildren(mode==='wire'?wireframe(pred.text,null):mode==='raw'?rawView(pred.text):diffView(s.label,pred.text));
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
    : '현재 UI 상태와 액션을 기준으로 정답 next state 와 각 체크포인트의 예측을 비교합니다. 와이어프레임, 정답 대비 line diff, Hungarian 계열 점수를 함께 제공합니다.';
  $('#sampleId').innerHTML=`${sp.label} 샘플 ${state.pos+1} / ${sp.samples.length}` +
    `<small>원본 행 #${s.index} · 모집단 ${sp.population.toLocaleString()}개</small>`;
  $('#progressBar').style.width=`${(state.pos+1)/sp.samples.length*100}%`;
  $('#actionChipLabel').textContent=MODE==='state'?'실행 액션 (입력)':'정답 액션 (GT)';
  $('#actionChip').textContent=actionText(MODE==='state'?s.action:s.gt_action);
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
