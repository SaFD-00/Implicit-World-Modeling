#!/usr/bin/env python3
"""EXP09 (Time-Mgmt LoRA, Qwen2.5-VL-3B) stage1 eval 결과 집계 + 플롯.

읽는 트리 (모두 `--eval-root` 아래):
    base/{id-seen,id-unseen,ood-chegal,ood-digibites}/
    general-full/{...}/
    time-mgmt-lora/epoch-{0.25,0.5,0.75,1,2,3}/{...}/

각 leaf 는 `hungarian_metrics.json` + `state_diff_metrics.json` 두 파일을 갖는다 (둘 다
scripts/_state_diff_eval.py / _hungarian_eval.py 의 **single-pair 모드** 출력 — flat
dict, `overall`/`in_domain`/`out_of_domain` 섹션 없음). 키는 `avg_` 접두사가 붙어
디스크에 있다 (`avg_addmod_recall` 등); 이 스크립트가 소비자에게 내보내는 bare 이름
(`addmod_recall` 등)과의 매핑은 STATE_DIFF_KEYS / HUNGARIAN_KEY 로 한 곳에 모아 둔다.
**절대 legacy alias(`avg_diff_recall`, `avg_change_f1`, ...) 를 읽지 않는다** —
`_state_diff_eval.py` 의 `_LEGACY_KEY_ALIAS` 가 옛 정의를 새 키 이름 옆에 얹어 내보내는
하위호환용이라, 같은 값을 두 번 세거나 옛 스키마 파일을 새 스키마로 오독하는 함정이다.

pooled-OOD (ood-chegal ∪ ood-digibites) 는 **근사가 아니라 정확한 값**이다:
`_state_diff_eval.aggregate()` 는 macro 평균(`avg_X = sum(rows)/n_X`)을 내고 그 분모
`n_X` 를 나란히 적어 두므로, 두 split 을 가중합하면 원래 380행을 한 번에 채점했을 때와
`round()` 오차 내로 동일하다 (`(avg_A*n_A + avg_B*n_B)/(n_A+n_B)`). `hungarian_f1` 은
`_hungarian_eval.evaluate_pairs.hung_avg` 가 항상 전체 `total` 을 분모로 쓰므로(실패 시
0.0 폴백, None-스킵 없음) `total` 가중이 정확하다. 그래서 state_diff 6개 지표는
`n_<metric>` 으로, hungarian_f1 은 `total` 로 가중한다 — 두 분모가 다를 수 있는 이유는
metric 마다 정의되는 행 수가 달라서다 (예: added_recall 은 GT 에 ADDED 요소가 있는
행에서만 정의된다).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent

# ── 트리 정의 (glob 아님 — enumerate 해야 오타 디렉터리가 조용히 사라지지 않는다) ──
SPLITS = ["id-seen", "id-unseen", "ood-chegal", "ood-digibites"]
EXPECTED_N = {"id-seen": 100, "id-unseen": 100, "ood-chegal": 88, "ood-digibites": 92}
LORA_EPOCHS = [0.25, 0.5, 0.75, 1, 2, 3]
OOD_APPS = {"ood-chegal": "com.chegal.alarm", "ood-digibites": "com.digibites.calendar"}

# (model_dir, epoch, display_label) — 곡선 순서 그대로. base 가 곡선의 epoch=0 점이다
# (general-full 은 곡선이 아니라 수평 기준선이라 따로 뺀다). epoch 은 LORA_EPOCHS 의
# 원래 타입(0.25 는 float, 1/2/3 은 int)을 그대로 쓴다 — float() 로 강제하면 "1" 이
# "1.0" 으로 출력돼 CSV/플롯 라벨이 "0, 0.25, ..., 1, 2, 3" 이라는 스펙 표기와 어긋난다.
CURVE_SPECS = [("base", 0, "Base(0)")] + [
    (f"time-mgmt-lora/epoch-{e}", e, f"LoRA({e})") for e in LORA_EPOCHS
]
STATIC_SPEC = ("general-full", "static", "General Full(static)")
ALL_MODEL_SPECS = CURVE_SPECS + [STATIC_SPEC]

# bare 이름 → 소스 JSON 의 avg_ 키. state_diff 쪽은 pooling 가중치로 쓸 n_ 키도 같이 문다.
STATE_DIFF_KEYS = {
    "addmod_recall": "avg_addmod_recall",
    "change_f1_strict": "avg_change_f1_strict",
    "copy_excess": "avg_copy_excess",
    "unchanged_recall": "avg_unchanged_recall",
    "added_recall": "avg_added_recall",
    "modified_recall": "avg_modified_recall",
}
HUNGARIAN_KEY = ("hungarian_f1", "avg_hungarian_f1")
ALL_METRIC_KEYS = list(STATE_DIFF_KEYS) + [HUNGARIAN_KEY[0]]
STATE_DIFF_SCHEMA = "2026-08-04"


class CorruptLeaf(RuntimeError):
    pass


@dataclass
class Leaf:
    model: str
    epoch: object  # float | "static"
    split: str
    n: int
    metrics: dict = field(default_factory=dict)  # bare key -> value
    weights: dict = field(default_factory=dict)  # bare key -> denominator row count


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise CorruptLeaf(f"{path}: JSON 파싱 실패 ({e})") from e


def _read_state_diff(path: Path) -> tuple[dict, dict, int, dict, float]:
    data = _load_json(path)
    if any(k in data for k in ("overall", "in_domain", "out_of_domain")):
        raise CorruptLeaf(
            f"{path}: split-mode(3-섹션) 산출물로 보인다 — EXP09 leaf 는 "
            "single-pair(flat) 모드여야 한다."
        )
    schema = data.get("metrics_schema")
    if schema != STATE_DIFF_SCHEMA:
        raise CorruptLeaf(
            f"{path}: metrics_schema={schema!r} (기대 {STATE_DIFF_SCHEMA!r}) — "
            "옛 스키마 파일은 같은 키 이름 아래 다른 정의를 담을 수 있다 "
            "(_state_diff_eval.py 의 _LEGACY_KEY_ALIAS 주석 참고)."
        )
    if "total" not in data:
        raise CorruptLeaf(f"{path}: 'total' 키가 없다.")
    metrics, weights = {}, {}
    for bare, avg_key in STATE_DIFF_KEYS.items():
        if avg_key not in data:
            raise CorruptLeaf(f"{path}: 필수 키 {avg_key!r} 가 없다.")
        metrics[bare] = data[avg_key]
        n_key = f"n_{bare}"
        if n_key not in data:
            raise CorruptLeaf(f"{path}: 필수 키 {n_key!r} 가 없다.")
        weights[bare] = data[n_key]
    for k in ("element_set", "xml_schema", "current_state_source"):
        if k not in data:
            raise CorruptLeaf(f"{path}: 필수 키 {k!r} 가 없다.")
    stamp = {k: data[k] for k in ("element_set", "xml_schema", "current_state_source")}
    return metrics, weights, data["total"], stamp, data.get("parse_fail_rate", 0.0)


def _read_hungarian(path: Path) -> tuple[dict, int, dict]:
    data = _load_json(path)
    if any(k in data for k in ("overall", "in_domain", "out_of_domain")):
        raise CorruptLeaf(
            f"{path}: split-mode(3-섹션) 산출물로 보인다 — EXP09 leaf 는 "
            "single-pair(flat) 모드여야 한다."
        )
    bare, avg_key = HUNGARIAN_KEY
    if avg_key not in data:
        raise CorruptLeaf(f"{path}: 필수 키 {avg_key!r} 가 없다.")
    if "total" not in data:
        raise CorruptLeaf(f"{path}: 'total' 키가 없다.")
    for k in ("element_set", "xml_schema"):
        if k not in data:
            raise CorruptLeaf(f"{path}: 필수 키 {k!r} 가 없다.")
    stamp = {k: data[k] for k in ("element_set", "xml_schema")}
    return {bare: data[avg_key]}, data["total"], stamp


# copy_excess 를 parse_fail_rate 없이 모델 간에 비교하지 말라는 경고가 소스에 명시돼
# 있다 (_state_diff_eval.py:763). 값 자체를 고치진 않고, 넘는 leaf 를 anomaly 로 올려
# Plot D(copy_excess 비교) 를 읽을 때 눈에 띄게 한다.
PARSE_FAIL_THRESHOLD = 0.02


def collect(eval_root: Path) -> tuple[list[Leaf], list[str], list[str], dict | None]:
    """(leaves, missing 설명, anomaly 설명, 관측된 채점 스탬프). corrupt 입력은 그대로 raise 돼 전파된다.

    `element_set`/`xml_schema`/`current_state_source` 는 `stamp_schema()` 가 "채점기가
    실제로 본 값"으로 찍어 두는 것이다 (_state_diff_eval.py:1204-1217 docstring). 8
    모델×4 split 을 한 표/곡선에 나란히 놓으므로, sibling 두 파일끼리도, leaf 전체에
    걸쳐서도 이 스탬프가 어긋나면 hard fail 한다 — 그렇지 않으면 서로 다른 채점 체제의
    숫자를 이어 그린 곡선이 조용히 나온다 (LoRA leaf 는 나중에 증분으로 채워지므로
    특히 중요).
    """
    leaves: list[Leaf] = []
    missing: list[str] = []
    anomalies: list[str] = []
    global_stamp: dict | None = None
    global_stamp_leaf: str | None = None
    for model_dir, epoch, _label in ALL_MODEL_SPECS:
        for split in SPLITS:
            leaf_dir = eval_root / model_dir / split
            tag = f"{model_dir}/{split}"
            if not leaf_dir.is_dir():
                missing.append(f"{tag}: 디렉터리 없음")
                continue
            hg_path = leaf_dir / "hungarian_metrics.json"
            sd_path = leaf_dir / "state_diff_metrics.json"
            if not hg_path.exists():
                missing.append(f"{tag}: hungarian_metrics.json 없음")
                continue
            if not sd_path.exists():
                missing.append(f"{tag}: state_diff_metrics.json 없음")
                continue
            hg_metrics, hg_total, hg_stamp = _read_hungarian(hg_path)
            sd_metrics, sd_weights, sd_total, sd_stamp, parse_fail_rate = _read_state_diff(sd_path)
            sd_pair = {k: sd_stamp[k] for k in ("element_set", "xml_schema")}
            if hg_stamp != sd_pair:
                raise CorruptLeaf(
                    f"{tag}: hungarian_metrics.json 과 state_diff_metrics.json 의 "
                    f"element_set/xml_schema 스탬프가 어긋난다 (hungarian={hg_stamp}, "
                    f"state_diff={sd_pair}) — 채점 설정 전파 실패다 (stamp_schema 가 "
                    "이걸 잡으라고 존재한다)."
                )
            if global_stamp is None:
                global_stamp, global_stamp_leaf = sd_stamp, tag
            elif sd_stamp != global_stamp:
                raise CorruptLeaf(
                    f"{tag}: 채점 스탬프가 {global_stamp_leaf} 와 다르다 "
                    f"(거기서 {global_stamp}, 여기서 {sd_stamp}) — 서로 다른 채점 "
                    "체제의 값을 같은 표/곡선에 섞을 수 없다."
                )
            if hg_total != sd_total:
                anomalies.append(
                    f"{tag}: hungarian total={hg_total} != state_diff total={sd_total}"
                )
            if sd_total != EXPECTED_N[split]:
                anomalies.append(
                    f"{tag}: total={sd_total} != 기대 split 크기 {EXPECTED_N[split]}"
                )
            if parse_fail_rate > PARSE_FAIL_THRESHOLD:
                anomalies.append(
                    f"{tag}: parse_fail_rate={parse_fail_rate} > {PARSE_FAIL_THRESHOLD} "
                    "— copy_excess 모델 간 비교 주의 (_state_diff_eval.py:763)"
                )
            metrics = {**sd_metrics, **hg_metrics}
            weights = {**sd_weights, HUNGARIAN_KEY[0]: hg_total}
            leaves.append(Leaf(model_dir, epoch, split, sd_total, metrics, weights))
    return leaves, missing, anomalies, global_stamp


def _leaf_lookup(leaves: list[Leaf]) -> dict[tuple[str, str], Leaf]:
    return {(l.model, l.split): l for l in leaves}


def pool_ood(model_dir: str, epoch: object, lut: dict[tuple[str, str], Leaf]) -> tuple[Leaf | None, Leaf | None]:
    """(ood-pooled, ood-macro) — 둘 다 chegal+digibites 가 모두 있어야 계산된다."""
    a, b = lut.get((model_dir, "ood-chegal")), lut.get((model_dir, "ood-digibites"))
    if a is None or b is None:
        return None, None
    n = a.n + b.n
    pooled_m, macro_m, pooled_w = {}, {}, {}
    for k in ALL_METRIC_KEYS:
        wa, wb = a.weights[k], b.weights[k]
        pooled_m[k] = round((a.metrics[k] * wa + b.metrics[k] * wb) / (wa + wb), 4) if (wa + wb) else None
        pooled_w[k] = wa + wb
        macro_m[k] = round((a.metrics[k] + b.metrics[k]) / 2, 4)
    pooled = Leaf(model_dir, epoch, "ood-pooled", n, pooled_m, pooled_w)
    macro = Leaf(model_dir, epoch, "ood-macro", n, macro_m, dict(pooled_w))
    return pooled, macro


def build_rows(leaves: list[Leaf]) -> list[Leaf]:
    lut = _leaf_lookup(leaves)
    rows = list(leaves)
    for model_dir, epoch, _label in ALL_MODEL_SPECS:
        pooled, macro = pool_ood(model_dir, epoch, lut)
        if pooled is not None:
            rows.extend([pooled, macro])
    return rows


# ── 출력: CSV / JSON ────────────────────────────────────────────────────
# 스펙이 못박은 컬럼 순서 그대로 (hungarian_f1 이 3번째) — ALL_METRIC_KEYS 정의 순서로
# 그냥 풀면 hungarian_f1 이 끝으로 밀린다.
CSV_COLUMNS = ["model", "epoch", "split", "n", "addmod_recall", "change_f1_strict",
               "hungarian_f1", "copy_excess", "unchanged_recall", "added_recall", "modified_recall"]
METRIC_COLUMNS = CSV_COLUMNS[4:]


def write_csv_json(rows: list[Leaf], out_dir: Path) -> None:
    order = {model_dir: i for i, (model_dir, _e, _l) in enumerate(ALL_MODEL_SPECS)}
    split_order = {s: i for i, s in enumerate([*SPLITS, "ood-pooled", "ood-macro"])}
    rows_sorted = sorted(rows, key=lambda r: (order[r.model], split_order[r.split]))

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows_sorted:
            w.writerow([r.model, r.epoch, r.split, r.n, *(r.metrics[k] for k in METRIC_COLUMNS)])

    payload = [
        {"model": r.model, "epoch": r.epoch, "split": r.split, "n": r.n,
         **{k: r.metrics[k] for k in METRIC_COLUMNS}}
        for r in rows_sorted
    ]
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ── 출력: markdown 표 ────────────────────────────────────────────────────
def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _md_table(headers: list[str], body: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def write_tables(rows: list[Leaf], out_dir: Path, missing: list[str], anomalies: list[str],
                  global_stamp: dict | None) -> None:
    lut = {(r.model, r.split): r for r in rows}
    sections = []

    sections.append("# EXP09 stage1 eval 결과 요약\n")
    if global_stamp is not None:
        sections.append(
            "_채점 스탬프(전 leaf 공통, collect() 가 sibling·전체 일치를 강제): "
            f"element_set=`{global_stamp['element_set']}`, "
            f"xml_schema=`{global_stamp['xml_schema']}`, "
            f"current_state_source=`{global_stamp['current_state_source']}`_"
        )

    # 1) primary: addmod_recall, 열 = ID-Seen/ID-Unseen/Chegal/DigiBites/OOD-Pooled
    headers = ["Model", "ID-Seen", "ID-Unseen", "Chegal", "DigiBites", "OOD-Pooled"]
    body = []
    for model_dir, _epoch, label in ALL_MODEL_SPECS:
        vals = [lut.get((model_dir, s)) for s in ("id-seen", "id-unseen", "ood-chegal", "ood-digibites", "ood-pooled")]
        body.append([label, *(v.metrics["addmod_recall"] if v else None for v in vals)])
    sections.append(
        "## Primary — addmod_recall\n\n" + _md_table(headers, body)
        + "\n\n(OOD-Pooled = chegal(n=88)/digibites(n=92) 를 metric 별 `n_<metric>` 으로 "
        "가중합한 값. `aggregate()` 가 macro 평균을 내므로 이는 근사가 아니라 두 split 을 "
        "합쳐 한 번에 채점했을 때와 동일한 값이다 — 스크립트 상단 docstring 참고.)"
    )

    # 2) detailed
    headers2 = ["Model", "Split", "n", "addmod_recall", "change_f1_strict", "hungarian_f1", "copy_excess", "unchanged_recall"]
    body2 = []
    for model_dir, _epoch, label in ALL_MODEL_SPECS:
        for split in [*SPLITS, "ood-pooled", "ood-macro"]:
            r = lut.get((model_dir, split))
            if r is None:
                continue
            body2.append([label, split, r.n, r.metrics["addmod_recall"], r.metrics["change_f1_strict"],
                          r.metrics["hungarian_f1"], r.metrics["copy_excess"], r.metrics["unchanged_recall"]])
    sections.append("## Detailed (per model x split)\n\n" + _md_table(headers2, body2))

    # 3) seen - unseen gap
    headers3 = ["Model", "ID-Seen addmod_recall", "ID-Unseen addmod_recall", "Gap (Seen-Unseen)",
                "ID-Seen hungarian_f1", "ID-Unseen hungarian_f1", "Gap (Seen-Unseen)"]
    body3 = []
    for model_dir, _epoch, label in ALL_MODEL_SPECS:
        seen, unseen = lut.get((model_dir, "id-seen")), lut.get((model_dir, "id-unseen"))
        if seen is None or unseen is None:
            continue
        gap_ar = round(seen.metrics["addmod_recall"] - unseen.metrics["addmod_recall"], 4)
        gap_hf = round(seen.metrics["hungarian_f1"] - unseen.metrics["hungarian_f1"], 4)
        body3.append([label, seen.metrics["addmod_recall"], unseen.metrics["addmod_recall"], gap_ar,
                      seen.metrics["hungarian_f1"], unseen.metrics["hungarian_f1"], gap_hf])
    sections.append(
        "## Seen - Unseen gap (fit vs. generalization, descriptive)\n\n" + _md_table(headers3, body3)
    )

    if missing:
        sections.append("## Missing leaves (" + str(len(missing)) + ")\n\n" + "\n".join(f"- {m}" for m in missing))
    if anomalies:
        sections.append("## Anomalies (" + str(len(anomalies)) + ")\n\n" + "\n".join(f"- {a}" for a in anomalies))

    (out_dir / "tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


# ── 플롯 ─────────────────────────────────────────────────────────────────
# x 축: 실측 epoch 값(0, 0.25, ..., 3) 을 그대로 쓰면 0→1 구간(변화가 실제로 일어나는
# 구간)이 전체 폭의 1/3 로 짓눌린다. 균등 간격 categorical 틱을 쓰고 라벨만 실값으로
# 붙인다 — 7 점을 모두 고르게 비교 가능하게 읽히도록 하는 선택이고, 물리적 epoch
# 간격의 정직성은 포기한다 (한 줄로 명시).
X_POS = list(range(len(CURVE_SPECS)))  # [0,1,2,3,4,5,6] <-> epoch [0,.25,.5,.75,1,2,3]
X_LABELS = [str(e) for _m, e, _l in CURVE_SPECS]


def _curve(rows_by_model: dict, split: str, key: str) -> list[float]:
    out = []
    for model_dir, _epoch, _label in CURVE_SPECS:
        r = rows_by_model.get((model_dir, split))
        v = r.metrics[key] if r is not None else None
        out.append(v if v is not None else math.nan)  # matplotlib breaks the line at NaN
    return out


def _static_value(rows_by_model: dict, split: str, key: str):
    r = rows_by_model.get((STATIC_SPEC[0], split))
    return r.metrics[key] if r is not None else None


def make_plots(rows: list[Leaf], out_dir: Path, dpi: int) -> list[str]:
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    lut = {(r.model, r.split): r for r in rows}
    produced = []

    def setup_ax(ax, title, ylabel):
        ax.set_xticks(X_POS)
        ax.set_xticklabels(X_LABELS)
        ax.set_xlabel("epoch (categorical, non-uniform spacing collapsed — see script docstring)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    # A: headline — addmod_recall on id-unseen, LoRA curve (base=epoch0 included) + general-full ref line
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(X_POS, _curve(lut, "id-unseen", "addmod_recall"), marker="o", linestyle="-",
            color="tab:blue", label="time-mgmt-lora (incl. base@0)")
    ref = _static_value(lut, "id-unseen", "addmod_recall")
    if ref is not None:
        ax.axhline(ref, color="tab:gray", linestyle="--", marker="x", label=f"general-full (static, {ref:.3f})")
    setup_ax(ax, "Plot A — addmod_recall on ID-Unseen", "addmod_recall")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "plot_a_headline_id_unseen.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    produced.append(p.name)

    # B: ID-Seen vs ID-Unseen (fit / generalization gap)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(X_POS, _curve(lut, "id-seen", "addmod_recall"), marker="o", linestyle="-",
            color="tab:blue", label="ID-Seen")
    ax.plot(X_POS, _curve(lut, "id-unseen", "addmod_recall"), marker="s", linestyle="--",
            color="tab:orange", label="ID-Unseen")
    setup_ax(ax, "Plot B — ID-Seen vs ID-Unseen (addmod_recall)", "addmod_recall")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "plot_b_seen_vs_unseen.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    produced.append(p.name)

    # C: chegal vs digibites (+ ood-pooled, subordinate)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(X_POS, _curve(lut, "ood-chegal", "addmod_recall"), marker="^", linestyle="-",
            color="tab:blue", label=OOD_APPS["ood-chegal"])
    ax.plot(X_POS, _curve(lut, "ood-digibites", "addmod_recall"), marker="v", linestyle="-",
            color="tab:orange", label=OOD_APPS["ood-digibites"])
    ax.plot(X_POS, _curve(lut, "ood-pooled", "addmod_recall"), marker=None, linestyle=":",
            color="tab:gray", linewidth=1.2, label="ood-pooled (n-weighted, subordinate)")
    setup_ax(ax, "Plot C — OOD apps (addmod_recall)", "addmod_recall")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "plot_c_ood_apps.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    produced.append(p.name)

    # D: diagnostic — is the model just copying? addmod_recall vs hungarian_f1 vs copy_excess on id-unseen
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(X_POS, _curve(lut, "id-unseen", "addmod_recall"), marker="o", linestyle="-",
            color="tab:blue", label="addmod_recall")
    ax.plot(X_POS, _curve(lut, "id-unseen", "hungarian_f1"), marker="s", linestyle="--",
            color="tab:orange", label="hungarian_f1")
    ax.plot(X_POS, _curve(lut, "id-unseen", "copy_excess"), marker="^", linestyle=":",
            color="tab:green", label="copy_excess")
    ax.axhline(0.0, color="black", linewidth=0.6)
    setup_ax(ax, "Plot D — copy-check diagnostic (ID-Unseen)", "value")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "plot_d_diagnostic_copy_check.png"
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    produced.append(p.name)

    return produced


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root", type=Path,
        default=PROJ / "outputs/AndroidControl_EXP09/eval/qwen2.5-vl-3b/stage1_eval",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="기본값: <eval-root>/_summary")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.eval_root / "_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    leaves, missing, anomalies, global_stamp = collect(args.eval_root)
    if not leaves:
        print(f"[summarize_exp09] ERROR: {args.eval_root} 아래에서 유효한 leaf 를 하나도 못 찾음", file=sys.stderr)
        return 2

    rows = build_rows(leaves)
    write_csv_json(rows, out_dir)
    write_tables(rows, out_dir, missing, anomalies, global_stamp)
    plots = make_plots(rows, out_dir, args.dpi)

    print(f"[summarize_exp09] leaves found: {len(leaves)}/32, rows (incl. pooled/macro): {len(rows)}")
    if missing:
        print(f"[summarize_exp09] missing leaves: {len(missing)}", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
    if anomalies:
        print(f"[summarize_exp09] anomalies: {len(anomalies)}", file=sys.stderr)
        for a in anomalies:
            print(f"  - {a}", file=sys.stderr)
    print(f"[summarize_exp09] wrote: results.csv, results.json, tables.md, {', '.join(plots)}")
    print(f"[summarize_exp09] out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
