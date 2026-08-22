#!/usr/bin/env python3
"""token_weights 시각 감사 사이트 — 학습 전에 "diff 가 제대로 잡혔나"를 눈으로 본다.

집계 숫자만으로는 이 파이프라인의 조용한 실패(위치 신호 사망, 컨테이너가 문서 전체를
덮음, 구조 div 미커버)가 **정상과 구분되지 않는다**. 그래서 assistant 토큰 하나하나에
어떤 가중치가 붙었는지, 그 근거가 어느 element 의 어떤 diff 판정인지 요소 단위로 편다.

  python scripts/diff_loss/weight_site.py \\
      --weighted data/AndroidControl_EXP08/_build/state_train_split_weighted.jsonl \\
      --out outputs/AndroidControl_EXP08/_weight_site

포맷별로 N 건씩 뽑아 하나의 자족 HTML 로 낸다 (외부 의존 없음, 브라우저로 바로 열림).
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent


def _load(path: Path, per_fmt: int, seed: int):
    """포맷별 per_fmt 건씩 저수지 표본 + 전량 상향-가중 비율 분포. 한 번만 읽는다."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    seen: Counter = Counter()
    dist: dict[str, list[float]] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            fmt = rec.get("fmt", "action")
            seen[fmt] += 1
            w = rec.get("token_weights") or []
            if w:
                # per-sample min 을 baseline 으로 쓰면 **전부 상향인 샘플**이 0 으로
                # 뒤집힌다 (min 이 1.0 이 되어 "초과" 가 0 건). baseline 은 파일 전체에서
                # 정해야 하므로 여기서는 값별 개수만 모으고 비율은 나중에 계산한다.
                dist.setdefault(fmt, []).append(Counter(round(x, 4) for x in w))
            b = buckets.setdefault(fmt, [])
            if len(b) < per_fmt:
                b.append(rec)
            else:  # 저수지 표본 — 앞쪽 편향 방지
                j = rng.randrange(seen[fmt])
                if j < per_fmt:
                    b[j] = rec
    print(f"[site] 입력 {sum(seen.values())}행  포맷 {dict(seen)}")
    return [r for fmt in sorted(buckets) for r in buckets[fmt]], dist


def _msg(rec: dict, role: str) -> str:
    for m in rec["messages"]:
        if m.get("from") == role or m.get("role") == role:
            return m.get("value") or m.get("content") or ""
    return ""


def _spans(tok, text: str, weights: list[float]):
    """토큰 offset 을 (start, end, weight) 로. 길이가 어긋나면 그 사실을 그대로 돌려준다."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    ok = len(offs) == len(weights)
    n = min(len(offs), len(weights))
    return [(offs[i][0], offs[i][1], weights[i]) for i in range(n)], ok, len(offs), len(weights)


def _render_weighted(text: str, spans, base: float) -> str:
    """가중치 구간을 색으로 칠한 HTML. baseline 은 **파일 전체 기준**으로 받는다 —
    샘플 안에서 min 을 잡으면 전부 상향인 샘플이 통째로 흐리게 칠해진다."""
    out, cur = [], 0
    for s, e, w in spans:
        if s >= e:
            continue
        if s > cur:
            out.append(html.escape(text[cur:s]))
        cls = "w-hi" if w > base else "w-lo"
        out.append(f'<span class="{cls}" title="w={w}">{html.escape(text[s:e])}</span>')
        cur = e
    out.append(html.escape(text[cur:]))
    return "".join(out)


def _current_state(user: str) -> str:
    m = re.search(r"Current UI State[^\n:]*:\n", user)
    if not m:
        return "(current state 머리글을 못 찾음)"
    rest = user[m.end() :]
    for stop in ("\n\nAction:", "\n\n[Screenshot]"):
        if stop in rest:
            rest = rest.split(stop)[0]
    return rest


CSS = """
:root{--bg:#fff;--fg:#111;--dim:#8a8f98;--hi:#fde68a;--hi-b:#b45309;--lo:#eef1f4;--line:#e3e6ea}
@media (prefers-color-scheme:dark){:root{--bg:#0f1113;--fg:#e7e9ea;--dim:#8a8f98;--hi:#4a3a12;--hi-b:#fbbf24;--lo:#1a1d21;--line:#272b30}}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:28px 0 8px}
.sub{color:var(--dim);margin-bottom:20px}
.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:12px 0 20px;font-size:13px}
.chip{padding:2px 8px;border-radius:4px}
.card{border:1px solid var(--line);border-radius:8px;margin:16px 0;overflow:hidden}
.hd{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:14px;flex-wrap:wrap;align-items:baseline}
.hd b{font-size:14px} .hd span{color:var(--dim);font-size:12.5px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0}
@media (max-width:1000px){.cols{grid-template-columns:1fr}}
.col{padding:12px 14px;min-width:0}
.col+.col{border-left:1px solid var(--line)}
@media (max-width:1000px){.col+.col{border-left:0;border-top:1px solid var(--line)}}
.col h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:460px;overflow:auto;background:var(--lo);padding:10px;border-radius:6px}
.w-hi{background:var(--hi);border-bottom:2px solid var(--hi-b)}
.w-lo{opacity:.55}
.bad{color:#dc2626;font-weight:600}
table{border-collapse:collapse;font-size:13px;margin:8px 0 0}
td,th{border:1px solid var(--line);padding:5px 10px;text-align:left}
th{background:var(--lo)}
"""


def build(args) -> int:
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    recs, dist = _load(args.weighted, args.per_fmt, args.seed)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    _GLOBAL_BASE = [min((v for cs in dist.values() for c in cs for v in c), default=0.0)]
    cards, mismatch = [], 0
    agg: Counter = Counter()
    for rec in recs:
        asst = _msg(rec, "gpt")
        w = rec.get("token_weights") or []
        spans, ok, n_tok, n_w = _spans(tok, asst, w)
        if not ok:
            mismatch += 1
        hi = sum(1 for _, _, x in spans if x > _GLOBAL_BASE[0])
        agg[rec.get("fmt", "action")] += 1
        dc = rec.get("_diff_counts", {})
        cards.append(f"""
<div class="card">
  <div class="hd">
    <b>fmt = {html.escape(str(rec.get('fmt','action')))}</b>
    <span>sample_id {rec.get('sample_id','-')}</span>
    <span>diff {html.escape(json.dumps(dc, ensure_ascii=False))}</span>
    <span>토큰 {n_w}개 · 상향 가중 {hi} ({hi/max(n_w,1):.0%})</span>
    {'' if ok else f'<span class="bad">길이 불일치! 토크나이즈 {n_tok} vs 저장 {n_w}</span>'}
  </div>
  <div class="cols">
    <div class="col"><h3>입력 — 모델이 실제로 본 current state</h3><pre>{html.escape(_current_state(_msg(rec,'human')))}</pre></div>
    <div class="col"><h3>타깃 — next state (색칠 = 상향 가중)</h3><pre>{_render_weighted(asst, spans, _GLOBAL_BASE[0])}</pre></div>
  </div>
</div>""")

    vals = sorted({round(x, 4) for r in recs for x in (r.get("token_weights") or [])})
    def _q(xs, p):
        xs = sorted(xs)
        return xs[int(p * (len(xs) - 1))] if xs else float("nan")

    # 파일 전체에서 baseline(=가장 낮은 가중치)을 정하고, 그 초과분을 "상향" 으로 센다.
    global_base = min((v for cs in dist.values() for c in cs for v in c), default=0.0)
    frac = {
        fmt: [
            sum(n for val, n in c.items() if val > global_base) / max(sum(c.values()), 1)
            for c in cs
        ]
        for fmt, cs in dist.items()
    }
    dist_rows = "".join(
        f"<tr><td>{html.escape(fmt)}</td><td>{len(v)}</td><td>{_q(v,.10):.3f}</td>"
        f"<td>{_q(v,.50):.3f}</td><td>{_q(v,.90):.3f}</td>"
        f"<td>{sum(1 for x in v if x > 0.999)} ({sum(1 for x in v if x > 0.999)/max(len(v),1):.2%})</td></tr>"
        for fmt, v in sorted(frac.items())
    )
    doc = f"""<!doctype html><meta charset="utf-8"><title>AC_EXP08 token_weights 감사</title>
<style>{CSS}</style>
<h1>AC_EXP08 — token_weights 시각 감사</h1>
<div class="sub">입력: <code>{html.escape(str(args.weighted))}</code> · 토크나이저 <code>{html.escape(args.model)}</code>
{('· revision <code>'+html.escape(args.revision)+'</code>') if args.revision else ''}</div>
<div class="legend">
  <span><span class="chip w-hi">색칠</span> = 상향 가중 (ADDED/MODIFIED 요소)</span>
  <span><span class="chip w-lo">흐림</span> = baseline (UNCHANGED · 태그 문법 · 들여쓰기)</span>
  <span>등장한 가중치 값: <b>{html.escape(str(vals))}</b></span>
</div>
<table>
  <tr><th>포맷</th><th>표본 수</th></tr>
  {''.join(f'<tr><td>{html.escape(k)}</td><td>{v}</td></tr>' for k, v in sorted(agg.items()))}
  <tr><th>길이 불일치</th><th class="{'bad' if mismatch else ''}">{mismatch}</th></tr>
</table>
<h2>diff 를 raw 로 계산했다는 증거 (전량 집계)</h2>
<p class="sub">한 샘플은 세 포맷 중 하나에만 속하므로 같은 sample_id 를 포맷끼리 대조할 수는 없다. 대신 <b>분포가 포맷 간에 같아야 한다</b>는 것이 증거다 — 만약 diff 를 applied(가려진 current)로 계산했다면 <code>dropped</code> 는 current 가 비어 next 전 요소가 ADDED 가 되므로 상향 비율이 1.0 에 몰려야 한다.</p>
<table>
  <tr><th>포맷</th><th>n</th><th>p10</th><th>p50</th><th>p90</th><th>전부 상향인 샘플</th></tr>
  {dist_rows}
</table>
<h2>무엇을 봐야 하나</h2>
<ul>
<li><b>색칠이 문서 전체를 덮으면 안 된다.</b> 컨테이너 하나의 판정이 서브트리를 통째로 덮는 실패 모드다 (UNCHANGED 요소가 있는데 100% 색칠이면 의심).</li>
<li><b>구조 <code>div</code> 의 <code>data-bbox</code> 좌표도 변화 시 색칠돼야 한다.</b> 안 되면 구조축이 빠진 것이다.</li>
<li><b>길이 불일치는 0 이어야 한다.</b> 어긋나면 LF collator 가 뒤에서부터 정렬해 가중치가 통째로 밀린다.</li>
</ul>
{''.join(cards)}
"""
    idx = out_dir / "index.html"
    idx.write_text(doc, encoding="utf-8")
    print(f"[site] {idx}  ({len(recs)}건, 길이 불일치 {mismatch})")
    return 1 if mismatch else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weighted", type=Path, required=True, help="token_weights 가 붙은 jsonl")
    p.add_argument("--out", type=Path, default=PROJ / "outputs" / "AndroidControl_EXP08" / "_weight_site")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--revision", default="66285546d2b821cf421d4f5eb2576359d3770cd3")
    p.add_argument("--per-fmt", type=int, default=6, help="포맷별 표본 수")
    p.add_argument("--seed", type=int, default=8)
    return build(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
