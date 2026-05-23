"""Build pair-aligned HTML comparison of Stage 1/2 eval outputs.

For each (stage, eval-DS) combination, bakes all model-variant predictions into
one HTML and adds in-page checkboxes to toggle prediction columns / metric rows.

Output (per stage, written next to the existing variant dirs):
  outputs/{data_dir}/eval/{model}/stage{N}_eval/
    pairs_on-AC.html
    pairs_on-MB.html
    pairs_summary.md
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STAGE_CONFIG: dict[int, dict] = {
    1: {
        "data_dir": "AC",
        "eval_subdir": "stage1_eval",
        "datasets": {
            # AC 는 Stage 1 도 ID/OOD split 이라 GT 가 두 파일로 분리됨.
            # 현 viewer 는 단일 GT 파일만 다루므로 ID 파일을 anchor 로 사용.
            # ID/OOD 양쪽을 한 화면에 보려면 viewer 자체 확장이 필요 (TODO).
            "on-AC":                  REPO / "data/AndroidControl/implicit-world-modeling_stage1_test_id.jsonl",
            "on-AC-without-open_app": REPO / "data/AndroidControl/implicit-world-modeling_stage1_test_id_without_open_app.jsonl",
            "on-MB":                  REPO / "data/MobiBench/implicit-world-modeling_stage1.jsonl",
            "on-MB-without-open_app": REPO / "data/MobiBench/implicit-world-modeling_stage1_without_open_app.jsonl",
        },
        "metric_files": ["predict_results.json", "hungarian_metrics.json"],
        "metric_keys": [
            "predict_bleu-4",
            "predict_rouge-l",
            "exact_match_rate",
            "avg_bleu",
            "avg_rouge_l",
            "avg_hungarian_ea",
            "avg_hungarian_f1",
            "avg_hungarian_prec",
            "avg_hungarian_rec",
            "avg_hungarian_text",
            "avg_hungarian_idx",
        ],
    },
    2: {
        "data_dir": "AC",
        "eval_subdir": "stage2_eval",
        "datasets": {
            "on-AC": REPO / "data/AndroidControl/implicit-world-modeling_stage2_test.jsonl",
            "on-MB": REPO / "data/MobiBench/implicit-world-modeling_stage2.jsonl",
        },
        "metric_files": ["predict_results.json", "action_metrics.json"],
        "metric_keys": [
            "total",
            "parse_rate",
            "type_accuracy",
            "step_accuracy",
            "macro_step_accuracy",
            "cond_index_acc",
            "cond_dir_acc",
            "cond_app_acc",
            "cond_text_acc",
            "predict_bleu-4",
            "predict_rouge-l",
        ],
    },
}

# `--data-dir` 별 STAGE_CONFIG override. AC_2 는 split 없는 단일 test 파일을 사용하므로
# `data_dir` 와 `datasets` 만 덮어쓰면 됨 (metric_files / metric_keys / eval_subdir 은 동일).
DATA_DIR_OVERRIDES: dict[str, dict[int, dict]] = {
    "AC": {},
    "AC_2": {
        1: {
            "data_dir": "AC_2",
            "datasets": {
                "on-AC":                  REPO / "data/AndroidControl_2/implicit-world-modeling_stage1_test.jsonl",
                "on-AC-without-open_app": REPO / "data/AndroidControl_2/implicit-world-modeling_stage1_test_without_open_app.jsonl",
                "on-MB":                  REPO / "data/MobiBench/implicit-world-modeling_stage1.jsonl",
                "on-MB-without-open_app": REPO / "data/MobiBench/implicit-world-modeling_stage1_without_open_app.jsonl",
            },
        },
        2: {
            "data_dir": "AC_2",
            "datasets": {
                "on-AC": REPO / "data/AndroidControl_2/implicit-world-modeling_stage2_test.jsonl",
                "on-MB": REPO / "data/MobiBench/implicit-world-modeling_stage2.jsonl",
            },
        },
    },
}


def resolve_cfg(stage: int, data_dir: str) -> dict:
    base = STAGE_CONFIG[stage]
    override = DATA_DIR_OVERRIDES.get(data_dir, {}).get(stage, {})
    return {**base, **override}

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
    return html.escape(s, quote=False)


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
    if isinstance(v, (int, float)):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return str(v) if v is not None else ""


def load_metrics(model_dir: Path, metric_files: list[str]) -> dict:
    merged: dict = {}
    for fn in metric_files:
        p = model_dir / fn
        if p.exists():
            data = json.loads(p.read_text())
            for k, v in data.items():
                if not isinstance(v, (dict, list)):
                    merged[k] = v
    return merged


def discover_variants(eval_root: Path, ds_marker: str) -> list[str]:
    found: list[str] = []
    if not eval_root.is_dir():
        return found
    for child in sorted(eval_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / ds_marker).is_dir():
            found.append(child.name)
        else:
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and (sub / ds_marker).is_dir():
                    found.append(f"{child.name}/{sub.name}")
    return found


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


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def build_dataset(
    stage: int,
    ds_name: str,
    test_path: Path,
    eval_root: Path,
    variants: list[str],
    metric_files: list[str],
    metric_keys: list[str],
) -> tuple[str, dict, int]:
    test_recs = read_jsonl(test_path)
    n = len(test_recs)
    images = [r.get("images", [""])[0] for r in test_recs]

    pred_recs: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}
    for v in variants:
        ds_dir = eval_root / v / ds_name
        recs = read_jsonl(ds_dir / "generated_predictions.jsonl")
        assert len(recs) == n, f"{v}/{ds_name}: {len(recs)} != {n}"
        pred_recs[v] = recs
        metrics[v] = load_metrics(ds_dir, metric_files)

    # control panel
    cb_html = "".join(
        f'<label><input type="checkbox" data-variant="{esc(v)}"'
        f'{" checked" if v == "base" else ""}> {esc(v)}</label>'
        for v in variants
    )
    controls = (
        '<div id="variant-controls"><strong>모델 선택:</strong>'
        f"{cb_html}"
        '<span class="actions">'
        '<button id="btn-all" type="button">all</button> '
        '<button id="btn-none" type="button">none</button>'
        "</span></div>"
    )

    # metric table
    metric_header = "".join(f"<th>{esc(k)}</th>" for k in metric_keys)
    metric_body = ""
    for v in variants:
        d = metrics[v]
        cells = "".join(f"<td>{fmt_num(d.get(k))}</td>" for k in metric_keys)
        metric_body += f'<tr data-variant="{esc(v)}"><th>{esc(v)}</th>{cells}</tr>'
    metric_table = (
        f'<table class="metric"><thead><tr><th>variant</th>{metric_header}</tr></thead>'
        f"<tbody>{metric_body}</tbody></table>"
    )

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Eval pairs · stage{stage} · {ds_name}</title>",
        f"<style>{CSS}</style></head><body>",
        f"<h1>Eval pairs · stage{stage} · {ds_name} · n={n}</h1>",
        f'<div class="meta">test: {esc(str(test_path.relative_to(REPO)))}<br>',
        f"eval_root: {esc(str(eval_root.relative_to(REPO)))}",
        "</div>",
        controls,
        metric_table,
    ]

    base_variant = variants[0]
    for i in range(n):
        anchor = pred_recs[base_variant][i]
        sys_msg, cur_xml, act_json = split_prompt(anchor["prompt"])
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
        for v in variants:
            pred = pred_recs[v][i].get("predict", "")
            parts.append(
                f'<section class="col-pred" data-variant="{esc(v)}">'
                f"<h3>{esc(v)}</h3>"
                f'<pre class="xml">{esc(pred)}</pre>'
                "</section>"
            )
        parts.append("</div></details>")

    parts.append(f"<script>{JS}</script>")
    parts.append("</body></html>")
    return "".join(parts), metrics, n


def build_summary_md(
    stage: int,
    eval_root: Path,
    metric_keys: list[str],
    per_ds: dict[str, tuple[dict, int]],
) -> str:
    out = [f"# Eval pairs summary · stage{stage}", ""]
    out.append(f"- eval_root: `{eval_root.relative_to(REPO)}`")
    out.append(f"- script: `scripts/eval_viewer.py`")
    out.append("")
    for ds, (metrics_by_variant, n) in per_ds.items():
        out.append(f"## {ds} (n={n})")
        out.append("")
        out.append("| variant | " + " | ".join(metric_keys) + " |")
        out.append("|" + "---|" * (len(metric_keys) + 1))
        for v, d in metrics_by_variant.items():
            row = "| " + v + " | " + " | ".join(fmt_num(d.get(k)) for k in metric_keys) + " |"
            out.append(row)
        out.append("")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        choices=list(DATA_DIR_OVERRIDES.keys()),
        default="AC",
        help="Output/data directory. AC=AndroidControl (default), AC_2=AndroidControl_2.",
    )
    p.add_argument(
        "--model",
        nargs="+",
        default=["qwen2.5-vl-7b"],
        help="Model dir(s) under outputs/{data_dir}/eval/. Multiple allowed.",
    )
    p.add_argument("--stages", type=int, nargs="+", choices=[1, 2], default=[1, 2])
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["on-AC", "on-AC-without-open_app", "on-MB", "on-MB-without-open_app"],
        help="Stage 1 은 정규/필터 4개가 기본. Stage 2 는 -without-open_app 을 산출하지 않으므로 "
             "해당 항목은 자동 skip.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant dirs (e.g. base full_world-model/epoch-3). Default: auto-discover.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for model in args.model:
        for stage in args.stages:
            cfg = resolve_cfg(stage, args.data_dir)
            eval_root = REPO / "outputs" / cfg["data_dir"] / "eval" / model / cfg["eval_subdir"]
            if not eval_root.is_dir():
                print(f"skip {model}/stage{stage}: {eval_root.relative_to(REPO)} not found")
                continue

            # discover_variants 는 단일 ds_marker 가 모든 variant 에 존재한다고 가정.
            # Stage 1 이면 정규 on-AC 가 anchor, Stage 2 면 첫 stage2 dataset (현재는 on-AC).
            ds_marker = next((d for d in args.datasets if d in cfg["datasets"]), None)
            if ds_marker is None:
                print(f"skip {model}/stage{stage}: no datasets matched stage{stage} config")
                continue
            discovered = discover_variants(eval_root, ds_marker)
            if args.variants is not None:
                unknown = [v for v in args.variants if v not in discovered]
                if unknown:
                    raise SystemExit(
                        f"{model}/stage{stage}: unknown variant(s) {unknown}. discovered: {discovered}"
                    )
                variants = list(args.variants)
            else:
                variants = discovered
            if not variants:
                print(f"skip {model}/stage{stage}: no variants found under {eval_root}")
                continue

            per_ds: dict[str, tuple[dict, int]] = {}
            for ds_name in args.datasets:
                if ds_name not in cfg["datasets"]:
                    # Stage 2 에는 -without-open_app 항목이 없음. 자동 skip.
                    continue
                test_path = cfg["datasets"][ds_name]
                ds_variants = [v for v in variants if (eval_root / v / ds_name).is_dir()]
                if not ds_variants:
                    print(f"skip {model}/stage{stage}/{ds_name}: no variants have this DS")
                    continue
                doc, metrics, n = build_dataset(
                    stage,
                    ds_name,
                    test_path,
                    eval_root,
                    ds_variants,
                    cfg["metric_files"],
                    cfg["metric_keys"],
                )
                target = eval_root / f"pairs_{ds_name}.html"
                target.write_text(doc)
                size_mb = target.stat().st_size / 1024 / 1024
                print(
                    f"wrote {target.relative_to(REPO)}  rows={n}  variants={len(ds_variants)}  size={size_mb:.1f}MB"
                )
                per_ds[ds_name] = (metrics, n)
            if per_ds:
                summary_path = eval_root / "pairs_summary.md"
                summary_path.write_text(build_summary_md(stage, eval_root, cfg["metric_keys"], per_ds))
                print(f"wrote {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
