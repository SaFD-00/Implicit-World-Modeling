"""Build pair-aligned HTML comparison of Stage 1/2 eval outputs.

각 (stage × logical-dataset) 별로 (EXP × MODEL × variant) prediction 을 하나의
HTML 로 묶고, in-page checkbox 로 column/row 를 토글한다. 같은 EXP 안의 비교
와 EXP 간 동일 stage 비교를 단일 CLI (`--include EXP:MODEL ...`) 로 처리한다.

Output
------
Single spec  : outputs/{DS_DATADIR(exp)}/eval/{model}/stage{N}_eval/pairs_*.html
Multi spec   : outputs/_compare/stage{N}_eval/pairs_*.html
(같은 위치에 `pairs_summary.md` 도 생성.)

Examples
--------
    # 단일 EXP — 그 EXP 의 eval/ 디렉토리에 산출
    python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b
    python scripts/eval_viewer.py --include AC_EXP01:qwen3-vl-8b_ratio73 --stages 2

    # 다중 EXP cross-compare — outputs/_compare/ 에 산출
    python scripts/eval_viewer.py --include AC_EXP01:qwen3-vl-8b_ratio73 AC_EXP02:qwen3-vl-8b

    # 데이터셋/variant 필터
    python scripts/eval_viewer.py --include AC_EXP02:qwen3-vl-8b \\
        --datasets on-AC-state-id on-AC-action-id \\
        --variants "lora_world-model/epoch-1"

EVAL_DATASETS 는 (stage, EXP, logical_key) → on-disk dir / predictions jsonl /
test jsonl / metric files 를 단일 매핑으로 갖는다. 디렉토리 명명은
`scripts/stage{1,2}_eval.sh` (`on-{EVAL_DS}[-state|-action][-without-open_app]`)
와 `scripts/_common.sh::DS_DATADIR` 에 정합.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# DS key → outputs/ 직속 디렉토리 (scripts/_common.sh::DS_DATADIR 와 정합).
DS_DATADIR: dict[str, str] = {
    "AC_EXP01": "AndroidControl_EXP01",
    "AC_EXP02": "AndroidControl_EXP02",
    "MC":       "MonkeyCollection",
}

STATE_METRIC_KEYS = [
    "total", "exact_match_rate",
    "avg_bleu", "avg_rouge_l",
    "avg_hungarian_ea", "avg_hungarian_f1",
    "avg_hungarian_prec", "avg_hungarian_rec",
    "avg_hungarian_text", "avg_hungarian_idx",
    "predict_bleu-4", "predict_rouge-l",
]
ACTION_METRIC_KEYS = [
    "total", "parse_rate",
    "type_accuracy", "step_accuracy", "macro_step_accuracy",
    "cond_index_acc", "cond_dir_acc", "cond_app_acc", "cond_text_acc",
    "predict_bleu-4", "predict_rouge-l",
]


def _ac_stage1_entries(exp: str) -> dict:
    """AC_EXP01/AC_EXP02 stage1 dual-task entries (ID/OOD × state/action × ±without-open_app)."""
    ds = DS_DATADIR[exp]
    data = REPO / "data" / ds
    actual = f"on-{exp}"  # on-AC_EXP01 / on-AC_EXP02
    return {
        "on-AC-state-id": {
            "dir":  f"{actual}-state",
            "pred": "generated_predictions_id.jsonl",
            "test": data / "implicit-world-modeling_stage1_test_id_state_pred.jsonl",
            "metric_files": [
                ("predict_results_id.json", None),
                ("hungarian_metrics.json", "in_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-state-ood": {
            "dir":  f"{actual}-state",
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "implicit-world-modeling_stage1_test_ood_state_pred.jsonl",
            "metric_files": [
                ("predict_results_ood.json", None),
                ("hungarian_metrics.json", "out_of_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-state-id-without-open_app": {
            "dir":  f"{actual}-state-without-open_app",
            "pred": "generated_predictions_id.jsonl",
            "test": data / "implicit-world-modeling_stage1_test_id_state_pred_without_open_app.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", "in_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-state-ood-without-open_app": {
            "dir":  f"{actual}-state-without-open_app",
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "implicit-world-modeling_stage1_test_ood_state_pred_without_open_app.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", "out_of_domain"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-AC-action-id": {
            "dir":  f"{actual}-action",
            "pred": "generated_predictions_id.jsonl",
            "test": data / "implicit-world-modeling_stage1_test_id_action_pred.jsonl",
            "metric_files": [
                ("predict_results_id.json", None),
                ("action_metrics.json", "in_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
        "on-AC-action-ood": {
            "dir":  f"{actual}-action",
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "implicit-world-modeling_stage1_test_ood_action_pred.jsonl",
            "metric_files": [
                ("predict_results_ood.json", None),
                ("action_metrics.json", "out_of_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
    }


def _mb_stage1_entries() -> dict:
    data = REPO / "data" / "MobiBench"
    return {
        "on-MB": {
            "dir":  "on-MB",
            "pred": "generated_predictions.jsonl",
            "test": data / "implicit-world-modeling_stage1.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", None),     # single-pair: top-level flat
                ("hungarian_metrics.json", "overall"),  # 호환: 혹시 nested 면 overall
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        "on-MB-without-open_app": {
            "dir":  "on-MB-without-open_app",
            "pred": "generated_predictions.jsonl",
            "test": data / "implicit-world-modeling_stage1_without_open_app.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", None),
                ("hungarian_metrics.json", "overall"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
    }


def _mc_stage1_entries() -> dict:
    data = REPO / "data" / "MonkeyCollection"
    return {
        "on-MC": {
            "dir":  "on-MC",
            "pred": "generated_predictions.jsonl",
            "test": data / "implicit-world-modeling_stage1_test.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("hungarian_metrics.json", None),
                ("hungarian_metrics.json", "overall"),
            ],
            "metric_keys": STATE_METRIC_KEYS,
        },
        # MC 의 without-open_app GT 는 data/MonkeyCollection/ 에 없으므로 등록하지 않는다.
    }


def _ac_stage2_entries(exp: str) -> dict:
    ds = DS_DATADIR[exp]
    data = REPO / "data" / ds
    actual = f"on-{exp}"
    return {
        "on-AC-id": {
            "dir":  actual,
            "pred": "generated_predictions_id.jsonl",
            "test": data / "implicit-world-modeling_stage2_test_id.jsonl",
            "metric_files": [
                ("predict_results_id.json", None),
                ("action_metrics.json", "in_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
        "on-AC-ood": {
            "dir":  actual,
            "pred": "generated_predictions_ood.jsonl",
            "test": data / "implicit-world-modeling_stage2_test_ood.jsonl",
            "metric_files": [
                ("predict_results_ood.json", None),
                ("action_metrics.json", "out_of_domain"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
    }


def _mb_stage2_entries() -> dict:
    data = REPO / "data" / "MobiBench"
    return {
        "on-MB": {
            "dir":  "on-MB",
            "pred": "generated_predictions.jsonl",
            "test": data / "implicit-world-modeling_stage2.jsonl",
            "metric_files": [
                ("predict_results.json", None),
                ("action_metrics.json", None),
                ("action_metrics.json", "overall"),
            ],
            "metric_keys": ACTION_METRIC_KEYS,
        },
    }


STAGE_CONFIG: dict[int, dict] = {
    1: {"eval_subdir": "stage1_eval"},
    2: {"eval_subdir": "stage2_eval"},
}

# (stage, EXP) → {logical_key: entry}
EVAL_DATASETS: dict[int, dict[str, dict[str, dict]]] = {
    1: {
        "AC_EXP01": {**_ac_stage1_entries("AC_EXP01"), **_mb_stage1_entries(), **_mc_stage1_entries()},
        "AC_EXP02": {**_ac_stage1_entries("AC_EXP02"), **_mb_stage1_entries(), **_mc_stage1_entries()},
        "MC":       {**_mc_stage1_entries(), **_mb_stage1_entries()},
    },
    2: {
        "AC_EXP01": {**_ac_stage2_entries("AC_EXP01"), **_mb_stage2_entries()},
        "AC_EXP02": {**_ac_stage2_entries("AC_EXP02"), **_mb_stage2_entries()},
    },
}


PROMPT_RE = re.compile(
    r"^system\n(?P<sys>.*?)\nuser\n\n## Current State\n(?P<xml>.*?)\n\n## Action\n(?P<act>.*?)\nassistant\n?$",
    re.DOTALL,
)


def split_prompt(prompt: str) -> tuple[str, str, str]:
    m = PROMPT_RE.match(prompt)
    if not m:
        return prompt, "", ""
    return m.group("sys"), m.group("xml"), m.group("act")


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def action_oneliner(act_json: str) -> str:
    try:
        a = json.loads(act_json)
    except Exception:
        return act_json[:80].replace("\n", " ")
    t = a.get("type", "?")
    params = a.get("params", {})
    extras = []
    if "index" in a:
        extras.append(f"index={a['index']}")
    if "default" in a:
        extras.append(f"default={a['default']}")
    p = json.dumps(params, ensure_ascii=False) if params else ""
    return f"{t}  {p}  {' '.join(extras)}".strip()


def fmt_num(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return str(v) if v is not None else ""


def load_metrics(target_dir: Path, metric_files: list[tuple[str, str | None]]) -> dict:
    """metric_files = [(filename, section_or_None)] 을 차례로 읽어 flat dict 로 합친다.

    section 이 None 이면 JSON top-level 의 numeric scalar 만 merge.
    section 이 str 이면 JSON[section] 이 dict 일 때만 그 안의 numeric scalar 만 merge.
    파일/섹션 부재는 silent skip.
    """
    merged: dict = {}
    for fn, section in metric_files:
        p = target_dir / fn
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if section is not None:
            data = data.get(section)
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if not isinstance(v, (dict, list)):
                merged.setdefault(k, v)
    return merged


def discover_variants(eval_root: Path, actual_dir: str, pred_filename: str) -> list[str]:
    """eval_root 아래에서 `{variant_path}/{actual_dir}/{pred_filename}` 가 존재하는 variant_path 들을 찾는다.

    variant_path 는 1-level (예: `base`) 또는 2-level (예: `lora_world-model/epoch-3`).
    """
    found: list[str] = []
    if not eval_root.is_dir():
        return found

    def has_target(v: Path) -> bool:
        return (v / actual_dir / pred_filename).is_file()

    for child in sorted(eval_root.iterdir()):
        if not child.is_dir():
            continue
        if has_target(child):
            found.append(child.name)
            continue
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and has_target(sub):
                found.append(f"{child.name}/{sub.name}")
    return found


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; margin: 1rem; color: #222; }
h1 { font-size: 18px; margin: 0 0 8px; }
.meta { color: #666; font-size: 12px; margin-bottom: 12px; }
#variant-controls { padding: 8px 0 12px; font-size: 12px; border-bottom: 1px solid #eee; margin-bottom: 12px; }
#variant-controls strong { margin-right: 8px; }
#variant-controls label { margin-right: 12px; cursor: pointer; white-space: nowrap; display: inline-block; }
#variant-controls .actions { margin-left: 8px; }
#variant-controls button { font-size: 11px; padding: 2px 6px; cursor: pointer; }
table.metric { border-collapse: collapse; margin-bottom: 16px; font-size: 12px; }
table.metric th, table.metric td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
table.metric th:first-child, table.metric td:first-child { text-align: left; font-weight: 600; }
details.row { border: 1px solid #e0e0e0; border-radius: 6px; margin: 4px 0; padding: 4px 10px; }
details.row[open] { background: #fafbfc; }
details.row > summary { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; cursor: pointer; line-height: 1.5; list-style: none; }
details.row > summary::-webkit-details-marker { display: none; }
details.row > summary::before { content: "\\25B8 "; color: #888; }
details.row[open] > summary::before { content: "\\25BE "; }
.idx { color: #888; }
.img { color: #0a64ad; }
.act { color: #444; }
.grid { display: grid; grid-template-columns: 1.3fr 1fr repeat(var(--n-active, 1), minmax(0, 1fr)); gap: 8px; margin: 8px 0 4px; }
.grid section { border: 1px solid #eee; border-radius: 4px; padding: 6px 8px; min-width: 0; background: #fff; }
.grid h3 { margin: 0 0 4px; font-size: 12px; color: #555; font-weight: 600; }
.grid pre { white-space: pre; overflow-x: auto; font-size: 11px; line-height: 1.4; margin: 4px 0; background: #f7f7f9; padding: 6px; border-radius: 3px; font-family: ui-monospace, Menlo, Consolas, monospace; }
pre.sys { background: #fff7e6; }
pre.action { background: #e6f4ff; }
pre.imginfo { background: #f0f0f0; color: #555; }
.col-input pre.xml { max-height: 480px; overflow: auto; }
.col-label pre.xml, .col-pred pre.xml { max-height: 480px; overflow: auto; }
"""

JS = """
(function () {
  const controls = document.querySelectorAll('#variant-controls input[type=checkbox]');
  function refresh() {
    const active = new Set();
    controls.forEach(cb => { if (cb.checked) active.add(cb.dataset.variant); });
    document.querySelectorAll('[data-variant]').forEach(el => {
      el.style.display = active.has(el.dataset.variant) ? '' : 'none';
    });
    document.documentElement.style.setProperty('--n-active', Math.max(1, active.size));
  }
  controls.forEach(cb => cb.addEventListener('change', refresh));
  document.getElementById('btn-all').addEventListener('click', () => {
    controls.forEach(cb => { cb.checked = true; });
    refresh();
  });
  document.getElementById('btn-none').addEventListener('click', () => {
    controls.forEach(cb => { cb.checked = false; });
    refresh();
  });
  refresh();
})();
"""


def variant_label(exp: str, model: str, variant_path: str, multi: bool) -> str:
    """multi-spec 일 때는 `[EXP] model/variant_path`, 단일이면 `variant_path` 그대로."""
    if multi:
        return f"[{exp}] {model}/{variant_path}"
    return variant_path


def build_dataset(
    stage: int,
    logical_key: str,
    spec_variants: list[tuple[str, str, str, dict, Path]],
    multi: bool,
) -> tuple[str, dict, int]:
    """spec_variants = [(exp, model, variant_path, entry, eval_root)]"""
    metric_keys = spec_variants[0][3]["metric_keys"]

    # predictions 적재. 첫 spec 의 entry 의 test 파일이 있으면 anchor 로 사용.
    pred_lists: dict[str, list[dict]] = {}
    metrics_by_label: dict[str, dict] = {}
    anchor_label: str | None = None
    anchor_test: Path | None = None

    for exp, model, vpath, entry, eval_root in spec_variants:
        label = variant_label(exp, model, vpath, multi)
        target_dir = eval_root / vpath / entry["dir"]
        recs = read_jsonl(target_dir / entry["pred"])
        pred_lists[label] = recs
        metrics_by_label[label] = load_metrics(target_dir, entry["metric_files"])
        if anchor_test is None:
            tp = entry.get("test")
            if tp is not None and Path(tp).is_file():
                anchor_test = Path(tp)
                anchor_label = label

    # 행 수 일관성 검증 — 모든 prediction 의 row 수가 같아야 같은 인덱스로 정렬됨.
    lengths = {label: len(recs) for label, recs in pred_lists.items()}
    n_set = set(lengths.values())
    if len(n_set) > 1:
        raise SystemExit(
            f"stage{stage}/{logical_key}: prediction row count mismatch — {lengths}. "
            "EXP01/EXP02 stage 데이터는 byte-identical copy 여야 cross-compare 가 가능합니다."
        )
    n = n_set.pop()

    # images[]: anchor_test 가 있으면 거기서, 없으면 빈 문자열.
    if anchor_test is not None:
        test_recs = read_jsonl(anchor_test)
        if len(test_recs) != n:
            raise SystemExit(
                f"stage{stage}/{logical_key}: anchor test ({anchor_test.relative_to(REPO)}) "
                f"len {len(test_recs)} != predictions len {n}"
            )
        images = [r.get("images", [""])[0] if r.get("images") else "" for r in test_recs]
    else:
        images = ["" for _ in range(n)]

    labels = list(pred_lists.keys())

    cb_html = "".join(
        f'<label><input type="checkbox" data-variant="{esc(lab)}"'
        f'{" checked" if i < 4 else ""}> {esc(lab)}</label>'
        for i, lab in enumerate(labels)
    )
    controls = (
        '<div id="variant-controls"><strong>모델 선택:</strong>'
        f"{cb_html}"
        '<span class="actions">'
        '<button id="btn-all" type="button">all</button> '
        '<button id="btn-none" type="button">none</button>'
        "</span></div>"
    )

    metric_header = "".join(f"<th>{esc(k)}</th>" for k in metric_keys)
    metric_body = ""
    for lab in labels:
        d = metrics_by_label[lab]
        cells = "".join(f"<td>{fmt_num(d.get(k))}</td>" for k in metric_keys)
        metric_body += f'<tr data-variant="{esc(lab)}"><th>{esc(lab)}</th>{cells}</tr>'
    metric_table = (
        f'<table class="metric"><thead><tr><th>variant</th>{metric_header}</tr></thead>'
        f"<tbody>{metric_body}</tbody></table>"
    )

    anchor_rel = (
        str(anchor_test.relative_to(REPO))
        if anchor_test is not None
        else "(no GT jsonl — prediction file 의 prompt/label 만 사용)"
    )
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Eval pairs · stage{stage} · {logical_key}</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>Eval pairs · stage{stage} · {logical_key} · n={n}</h1>",
        f'<div class="meta">test: {esc(anchor_rel)}<br>',
        f"variants: {len(labels)} (multi-spec)" if multi else f"variants: {len(labels)} (single-spec)",
        "</div>",
        controls,
        metric_table,
    ]

    base_label = labels[0]
    for i in range(n):
        anchor = pred_lists[base_label][i]
        sys_msg, cur_xml, act_json = split_prompt(anchor.get("prompt", ""))
        label_xml = anchor.get("label", "")
        summary_line = (
            f'<span class="idx">#{i:04d}</span> · '
            f'<span class="img">{esc(images[i])}</span> · '
            f'<span class="act">{esc(action_oneliner(act_json))}</span>'
        )
        parts.append(f'<details class="row"><summary>{summary_line}</summary>')
        parts.append('<div class="grid">')
        parts.append(
            '<section class="col-input"><h3>Input</h3>'
            f'<pre class="imginfo">image: {esc(images[i])}</pre>'
            f'<pre class="sys">{esc(sys_msg)}</pre>'
            f'<pre class="xml">{esc(cur_xml)}</pre>'
            f'<pre class="action">{esc(act_json)}</pre>'
            "</section>"
        )
        parts.append(
            f'<section class="col-label"><h3>Label</h3><pre class="xml">{esc(label_xml)}</pre></section>'
        )
        for lab in labels:
            pred = pred_lists[lab][i].get("predict", "")
            parts.append(
                f'<section class="col-pred" data-variant="{esc(lab)}">'
                f"<h3>{esc(lab)}</h3>"
                f'<pre class="xml">{esc(pred)}</pre>'
                "</section>"
            )
        parts.append("</div></details>")

    parts.append(f"<script>{JS}</script>")
    parts.append("</body></html>")
    return "".join(parts), metrics_by_label, n


def build_summary_md(
    stage: int,
    out_root: Path,
    per_ds: dict[str, tuple[list[str], dict[str, dict], int]],
    spec_specs: list[tuple[str, str]],
) -> str:
    out = [f"# Eval pairs summary · stage{stage}", ""]
    out.append(f"- out_root: `{out_root.relative_to(REPO)}`")
    out.append("- script: `scripts/eval_viewer.py`")
    out.append(f"- include: {', '.join(f'`{e}:{m}`' for e, m in spec_specs)}")
    out.append("")
    for ds, (metric_keys, metrics_by_label, n) in per_ds.items():
        out.append(f"## {ds} (n={n})")
        out.append("")
        out.append("| variant | " + " | ".join(metric_keys) + " |")
        out.append("|" + "---|" * (len(metric_keys) + 1))
        for lab, d in metrics_by_label.items():
            row = "| " + lab + " | " + " | ".join(fmt_num(d.get(k)) for k in metric_keys) + " |"
            out.append(row)
        out.append("")
    return "\n".join(out)


def parse_spec(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise SystemExit(f"--include 항목 '{s}' 은 EXP:MODEL 형식이어야 함 (예: AC_EXP02:qwen3-vl-8b).")
    exp, model = s.split(":", 1)
    exp = exp.strip()
    model = model.strip()
    if exp not in DS_DATADIR:
        raise SystemExit(
            f"--include 항목 '{s}' 의 EXP '{exp}' 미등록 — 허용: {sorted(DS_DATADIR)}"
        )
    if not model:
        raise SystemExit(f"--include 항목 '{s}' 의 MODEL 이 비어있음.")
    return exp, model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--include", nargs="+", required=True, metavar="EXP:MODEL",
        help="비교할 (EXP, MODEL) 쌍. 1개면 단일-EXP 모드, 2개 이상이면 cross-EXP 모드. "
             "EXP ∈ {AC_EXP01, AC_EXP02, MC}, MODEL = outputs/<DS_DATADIR(EXP)>/eval/ 아래 디렉토리 명. "
             "예: --include AC_EXP01:qwen3-vl-8b_ratio73 AC_EXP02:qwen3-vl-8b",
    )
    p.add_argument(
        "--stages", type=int, nargs="+", choices=[1, 2], default=[1, 2],
        help="처리할 stage (기본 1 2 모두).",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None, metavar="LOGICAL_KEY",
        help="처리할 logical key (예: on-AC-state-id, on-MB). 기본 = 각 EXP 가 가진 logical key 합집합.",
    )
    p.add_argument(
        "--variants", nargs="+", default=None,
        help="처리할 variant_path 화이트리스트 (예: base 'lora_world-model/epoch-3'). 기본 = auto-discover.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs: list[tuple[str, str]] = [parse_spec(s) for s in args.include]
    multi = len(specs) > 1

    for stage in args.stages:
        eval_subdir = STAGE_CONFIG[stage]["eval_subdir"]

        # 처리할 logical key 결정
        if args.datasets is not None:
            logical_keys = list(args.datasets)
        else:
            seen: set[str] = set()
            logical_keys = []
            for exp, _ in specs:
                for k in EVAL_DATASETS[stage].get(exp, {}):
                    if k not in seen:
                        seen.add(k)
                        logical_keys.append(k)

        if not logical_keys:
            print(f"skip stage{stage}: spec 들이 가진 logical key 없음")
            continue

        # 출력 경로 분기
        if multi:
            out_root = REPO / "outputs" / "_compare" / eval_subdir
        else:
            exp, model = specs[0]
            out_root = REPO / "outputs" / DS_DATADIR[exp] / "eval" / model / eval_subdir
        out_root.mkdir(parents=True, exist_ok=True)

        per_ds: dict[str, tuple[list[str], dict[str, dict], int]] = {}
        for logical_key in logical_keys:
            spec_variants: list[tuple[str, str, str, dict, Path]] = []
            for exp, model in specs:
                entry = EVAL_DATASETS[stage].get(exp, {}).get(logical_key)
                if entry is None:
                    continue
                eval_root = REPO / "outputs" / DS_DATADIR[exp] / "eval" / model / eval_subdir
                discovered = discover_variants(eval_root, entry["dir"], entry["pred"])
                if args.variants is not None:
                    discovered = [v for v in discovered if v in args.variants]
                for v in discovered:
                    spec_variants.append((exp, model, v, entry, eval_root))

            if not spec_variants:
                print(f"skip stage{stage}/{logical_key}: no variants found across specs")
                continue

            try:
                doc, metrics_by_label, n = build_dataset(stage, logical_key, spec_variants, multi)
            except SystemExit:
                raise
            except Exception as e:
                print(f"error stage{stage}/{logical_key}: {e}")
                continue

            target = out_root / f"pairs_{logical_key}.html"
            target.write_text(doc)
            size_mb = target.stat().st_size / 1024 / 1024
            print(
                f"wrote {target.relative_to(REPO)}  rows={n}  variants={len(spec_variants)}  size={size_mb:.1f}MB"
            )
            per_ds[logical_key] = (spec_variants[0][3]["metric_keys"], metrics_by_label, n)

        if per_ds:
            summary_path = out_root / "pairs_summary.md"
            summary_path.write_text(build_summary_md(stage, out_root, per_ds, specs))
            print(f"wrote {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
