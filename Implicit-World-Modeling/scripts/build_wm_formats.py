#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_wm_formats.py
===================
Mobile GUI world-model (NEXT_STATE_PREDICTION) 학습 데이터를 세 가지 관측성
(observability) 포맷으로 분할/변환한다.

  full    (기본 25%) : 입력 XML 전체 유지
  masked  (기본 55%) : 입력 XML 일부를 <MASK/> / [MASK] 로 치환
  dropped (기본 20%) : 입력 XML 완전 제거 (스크린샷만)

출력은 두 파일이다.
  *_split_raw.jsonl      원본 messages 그대로 + sample_id / content_hash / fmt
  *_split_applied.jsonl  변환된 messages + 위 메타 + meta.masked_elements

두 파일은 같은 순서, 같은 sample_id 로 1:1 대응한다.
diff-loss(헝가리안) 파이프라인은 raw 를 읽어 타깃 diff 구간을 구하고,
applied 의 meta.masked_elements 와 sample_id 로 조인하여 3단 가중치를 만든다.

사용법:
  python build_wm_formats.py \
      --input  all_samples_state_pred.jsonl \
      --outdir out/ \
      --seed 42

주요 옵션은 --help 참조.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 상수
# --------------------------------------------------------------------------- #

MASK_ELEM = "<MASK/>"      # 요소 전체 / inner text 위치용 센티넬
MASK_ATTR = "[MASK]"       # 속성 값 내부용 센티넬 (well-formed 유지)

HDR_UI = "Current UI State:"
HDR_SHOT = "[Screenshot]"
HDR_ACTION = "Action:"

# 우선 마스킹 대상 태그
TIER1_TAGS = {"button", "p", "input"}
TIER2_TAGS = {"button", "p", "input", "img"}

# 시스템 프롬프트 앵커
ANCHOR_OUTPUT_FMT = "# Output Format (STRICT)"
OLD_GIVEN_LINE = "- Current UI State is provided as html-style XML and a screenshot."
NEW_GIVEN_LINE = (
    "- Current UI State is provided as a screenshot, optionally accompanied by "
    "html-style XML (see # Observability)."
)

OBSERVABILITY = {
    "full": (
        "# Observability\n"
        "The current UI state is given in FULL: the screenshot and the complete XML.\n"
        "The XML describes the state BEFORE the action. Identify which elements the\n"
        "action affects, then emit the state AFTER it."
    ),
    "masked": (
        "# Observability\n"
        "The current UI state is given as a screenshot plus a PARTIAL XML.\n"
        "Hidden content is replaced by " + MASK_ELEM + " (an element or its inner text)\n"
        "or " + MASK_ATTR + " (an attribute value). These are placeholders, never real\n"
        "screen content. Resolve each placeholder from the screenshot first, then\n"
        "predict the resulting state. Never emit " + MASK_ELEM + " or " + MASK_ATTR + "\n"
        "in your output."
    ),
    "dropped": (
        "# Observability\n"
        "The current UI state is given as a SCREENSHOT ONLY. No XML is provided.\n"
        "Read the screen from the image, then predict the resulting state."
    ),
}

INLINE_MARKER = {
    "full": "Current UI State (FULL):",
    "masked": "Current UI State (PARTIAL — hidden content is marked "
              + MASK_ELEM + " / " + MASK_ATTR + "):",
    "dropped": "Current UI State (NOT PROVIDED):",
}

DROPPED_BODY = "(none)"


# --------------------------------------------------------------------------- #
# Step 0 — XML 스캔 파서
# --------------------------------------------------------------------------- #

# 속성 값 안에 '>' 가 들어와도 안전하도록 큰따옴표 구간을 통째로 소비한다.
TAG_RE = re.compile(r'<\s*(/?)\s*([A-Za-z][\w:-]*)((?:"[^"]*"|[^<>"])*?)(/?)\s*>')
ARIA_RE = re.compile(r'aria-label\s*=\s*"([^"]*)"')
BBOX_RE = re.compile(r'data-bbox\s*=\s*"([^"]*)"')


@dataclass
class Node:
    """XML 요소 하나. 모든 span 은 파싱 대상 XML 문자열 기준 (start, end) 이다."""
    tag: str
    depth: int
    open_start: int
    open_end: int
    close_start: Optional[int] = None   # 닫는 태그 시작
    close_end: Optional[int] = None     # 닫는 태그 끝
    self_closing: bool = False
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    attrs_raw: str = ""
    bbox: str = ""
    aria: Optional[str] = None
    aria_span: Optional[Tuple[int, int]] = None   # 값(따옴표 안) 절대 구간

    # ---- 파생 span ----
    @property
    def span_full(self) -> Tuple[int, int]:
        """서브트리 전체(여는 태그 ~ 닫는 태그) 구간."""
        return (self.open_start, self.close_end if self.close_end is not None else self.open_end)

    @property
    def span_text(self) -> Optional[Tuple[int, int]]:
        """자식 태그가 없는 요소의 inner text 구간. 없으면 None."""
        if self.self_closing or self.close_start is None:
            return None
        if self.children:                      # 컨테이너의 '내용'은 자식 요소이므로 제외
            return None
        if self.close_start <= self.open_end:
            return None
        return (self.open_end, self.close_start)

    def text_of(self, xml: str) -> str:
        s = self.span_text
        return xml[s[0]:s[1]] if s else ""


class ParseError(RuntimeError):
    pass


def parse_xml(xml: str) -> List[Node]:
    """XML 문자열을 스캔하여 Node 리스트(문서 순서)를 만든다."""
    nodes: List[Node] = []
    stack: List[Node] = []

    for m in TAG_RE.finditer(xml):
        closing, tag, attrs, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)

        if closing:                                    # </tag>
            if not stack:
                raise ParseError(f"unmatched close tag </{tag}> at {m.start()}")
            node = stack.pop()
            if node.tag != tag:
                raise ParseError(f"tag mismatch: <{node.tag}> ... </{tag}> at {m.start()}")
            node.close_start, node.close_end = m.start(), m.end()
            continue

        node = Node(
            tag=tag,
            depth=len(stack),
            open_start=m.start(),
            open_end=m.end(),
            self_closing=bool(selfclose),
            parent=stack[-1] if stack else None,
            attrs_raw=attrs,
        )
        bm = BBOX_RE.search(attrs)
        if bm:
            node.bbox = bm.group(1)
        am = ARIA_RE.search(attrs)
        if am:
            node.aria = am.group(1)
            # attrs = m.group(3). m.start(3) 로 절대 오프셋을 정확히 얻는다.
            attrs_abs = m.start(3)
            node.aria_span = (attrs_abs + am.start(1), attrs_abs + am.end(1))

        if node.parent is not None:
            node.parent.children.append(node)
        nodes.append(node)

        if not node.self_closing:
            stack.append(node)

    if stack:
        raise ParseError(f"unclosed tags: {[n.tag for n in stack]}")
    return nodes


def verify_roundtrip(xml: str, nodes: Sequence[Node]) -> bool:
    """파서가 구조를 올바로 잡았는지: 루트 span 이 문자열 전체를 덮는지 확인."""
    roots = [n for n in nodes if n.parent is None]
    if not roots:
        return False
    lo = min(n.span_full[0] for n in roots)
    hi = max(n.span_full[1] for n in roots)
    return xml[:lo].strip() == "" and xml[hi:].strip() == ""


# --------------------------------------------------------------------------- #
# Step 2 — 마스킹
# --------------------------------------------------------------------------- #

_ARIA_BAD_EXT = re.compile(r"\.\w{2,4}\b")


def aria_maskable(v: str) -> bool:
    """aria-label 값이 스크린샷에서 복원 가능한 짧은 라벨인지 휴리스틱 판정.

    제외 대상: 합성 접근성 서술("Inbox, tab 4 out of 5"), UUID, 이메일,
    인코딩된 개행(&#10;), 파일명. 실측 통과율 약 79%.
    """
    v = v.strip()
    if not v:
        return False
    if len(v) > 25:
        return False
    if len(v.split()) > 3:
        return False
    if "," in v or "@" in v or "&#" in v:
        return False
    if _ARIA_BAD_EXT.search(v):
        return False
    return True


def tier_of(node: Node) -> int:
    has_text = node.span_text is not None and node.text_of_cache.strip() != ""
    has_aria = node.aria is not None
    if node.tag in TIER1_TAGS and (has_text or has_aria):
        return 1
    if node.tag in TIER2_TAGS:
        return 2
    return 3


def choose_kind(node: Node, rng: random.Random) -> str:
    """요소가 가진 필드에 따라 마스킹 방식을 확률적으로 고른다.

    누수 방지 규칙
    --------------
    inner text 와 aria-label 은 대개 같은 정보를 담는다. 한쪽만 가리면
    다른 쪽에서 답이 새므로,
      - 둘 다 있고 aria 가 마스킹 허용이면  -> 함께 가린다 ("text+aria")
      - 둘 다 있지만 aria 가 마스킹 불허면  -> 부분 마스킹 금지, whole 만
    """
    has_text = node.span_text is not None and node.text_of_cache.strip() != ""
    has_aria = node.aria is not None
    aria_ok = has_aria and aria_maskable(node.aria or "")

    if has_text and has_aria:
        if aria_ok:
            return rng.choices(["whole", "text+aria"], weights=[0.60, 0.40])[0]
        return "whole"                      # aria 로 누수되므로 부분 마스킹 불가
    if has_text:
        return rng.choices(["whole", "text"], weights=[0.70, 0.30])[0]
    if aria_ok:
        return rng.choices(["whole", "aria"], weights=[0.70, 0.30])[0]
    return "whole"


def spans_for_kind(node: Node, kind: str) -> List[Tuple[Tuple[int, int], str]]:
    """마스킹 방식에 대응하는 (span, 치환문자열) 목록. 빈 목록이면 적용 불가."""
    out: List[Tuple[Tuple[int, int], str]] = []
    if kind == "whole":
        out.append((node.span_full, MASK_ELEM))
    else:
        if "text" in kind and node.span_text is not None:
            out.append((node.span_text, MASK_ELEM))
        if "aria" in kind and node.aria_span is not None:
            out.append((node.aria_span, MASK_ATTR))
    return out


def _relation(span: Tuple[int, int], taken: List[Tuple[int, int]]) -> Tuple[str, List[int]]:
    """새 span 과 기존 마스킹 구간들의 관계를 판정한다.

    returns ("free", [])            겹치는 것이 없음
            ("subsume", [idx...])   기존 구간들을 완전히 포함 (흡수 가능)
            ("conflict", [])        부분 겹침 또는 기존 구간에 포함됨 (불가)
    """
    a, b = span
    contained: List[int] = []
    for i, (c, d) in enumerate(taken):
        if a < d and c < b:                 # 겹침
            if a <= c and d <= b:           # 새 span 이 기존을 완전히 포함
                contained.append(i)
            else:
                return ("conflict", [])
    return (("subsume", contained) if contained else ("free", []))


def sample_ratio(rng: random.Random, lo: float, hi: float, dist: str) -> float:
    if dist == "uniform":
        return rng.uniform(lo, hi)
    # beta(2, 1.6): 중간~높은 쪽으로 완만히 치우친 분포 (평균 ≈ 0.56 스케일 후)
    return lo + (hi - lo) * rng.betavariate(2.0, 1.6)


def mask_xml(
    xml: str,
    rng: random.Random,
    ratio_lo: float,
    ratio_hi: float,
    dist: str,
    min_depth: int = 2,
    overshoot: float = 1.3,
) -> Tuple[str, float, float, List[dict], Counter]:
    """XML 문자열에 마스킹을 적용한다.

    내부 자료구조는 세 개의 병렬 리스트다 (index 정렬 유지).
      taken[i]   : 마스킹 대상 문자 구간 (원본 XML 기준)
      repls[i]   : 치환 문자열 (MASK_ELEM 또는 MASK_ATTR)
      records[i] : 조인용 메타 (kind / tag / bbox / key / group)

    text+aria 동시 마스킹은 두 개의 span 을 같은 group 번호로 묶어 표현한다.

    Returns
    -------
    masked_xml, ratio_target, ratio_actual, masked_elements, tier_counter
    """
    nodes = parse_xml(xml)
    for n in nodes:                       # tier_of / choose_kind 에서 재사용
        n.text_of_cache = n.text_of(xml)  # type: ignore[attr-defined]

    ratio_target = sample_ratio(rng, ratio_lo, ratio_hi, dist)
    budget = ratio_target * len(xml)

    buckets: Dict[int, List[Node]] = defaultdict(list)
    for n in nodes:
        if n.parent is None:
            continue                       # 루트 제외
        buckets[tier_of(n)].append(n)
    for t in buckets:
        rng.shuffle(buckets[t])

    taken: List[Tuple[int, int]] = []
    repls: List[str] = []
    records: List[dict] = []
    tier_used: Counter = Counter()
    group_seq = [0]

    def _saved_of(i: int) -> int:
        a, b = taken[i]
        return (b - a) - len(repls[i])

    def _commit(node: Node, pairs, kinds, tier: int, contained: List[int]) -> None:
        for i in sorted(contained, reverse=True):
            taken.pop(i); repls.pop(i); records.pop(i)
        g = group_seq[0]; group_seq[0] += 1
        for (span, repl), kind in zip(pairs, kinds):
            taken.append(span); repls.append(repl)
            records.append({
                "group": g,
                "kind": kind,
                "tag": node.tag,
                "bbox": node.bbox,
                "key": ((node.aria or "") if kind == "aria"
                        else (node.text_of_cache.strip() or (node.aria or "")))[:80],
                "src_span": [span[0], span[1]],
            })
        tier_used[tier] += 1

    def _plan(node: Node, kind: str):
        """(pairs, kinds) 또는 None. pairs = [(span, repl), ...]"""
        pairs = spans_for_kind(node, kind)
        if not pairs:
            return None
        if kind == "text+aria" and len(pairs) < 2:
            return None                       # 한쪽만 가리면 누수 -> 취소
        kinds = ["whole"] if kind == "whole" else (
            ["text", "aria"] if kind == "text+aria" else [kind])
        return pairs, kinds

    def _feasible(pairs):
        """겹침 검사 + 절약량 계산. 불가하면 None."""
        contained_all: List[int] = []
        for span, _ in pairs:
            rel, contained = _relation(span, taken)
            if rel == "conflict":
                return None
            contained_all.extend(contained)
        contained_all = sorted(set(contained_all))
        refund = sum(_saved_of(i) for i in contained_all)
        saved = sum((b - a) - len(r) for (a, b), r in pairs) - refund
        if saved <= 0 or saved > budget * overshoot:
            return None
        return contained_all, saved

    # --- 1차: 티어 순 (의미 있는 요소 우선) ---
    for tier in (1, 2, 3):
        if budget <= 0:
            break
        for node in buckets.get(tier, []):
            if budget <= 0:
                break
            kind = choose_kind(node, rng)
            if kind == "whole" and node.depth < min_depth:
                continue
            plan = _plan(node, kind)
            if plan is None:
                continue
            pairs, kinds = plan
            feas = _feasible(pairs)
            if feas is None:
                continue
            contained, saved = feas
            _commit(node, pairs, kinds, tier, contained)
            budget -= saved

    # --- 2차: 예산이 남으면 큰 컨테이너부터 흡수 ---
    if budget > 0:
        rest = sorted((n for ns in buckets.values() for n in ns),
                      key=lambda n: -(n.span_full[1] - n.span_full[0]))
        for node in rest:
            if budget <= 0:
                break
            if node.depth < min_depth:
                continue
            pairs = [(node.span_full, MASK_ELEM)]
            feas = _feasible(pairs)
            if feas is None:
                continue
            contained, saved = feas
            _commit(node, pairs, ["whole"], tier_of(node), contained)
            budget -= saved

    # --- 치환 ---
    order = sorted(range(len(taken)), key=lambda i: taken[i][0])
    out, cur = [], 0
    for i in order:
        a, b = taken[i]
        out.append(xml[cur:a]); out.append(repls[i]); cur = b
    out.append(xml[cur:])
    masked = "".join(out)

    ratio_actual = 1.0 - len(masked) / max(len(xml), 1)
    records.sort(key=lambda r: r["src_span"][0])
    return masked, ratio_target, ratio_actual, records, tier_used


# --------------------------------------------------------------------------- #
# Step 3/4 — 프롬프트 재조립
# --------------------------------------------------------------------------- #

@dataclass
class HumanParts:
    xml: str
    shot_block: str      # "[Screenshot]\n<image>"
    action_block: str    # "Action:\n<action>{...}</action>"
    xml_span: Tuple[int, int]


def split_human(human: str) -> HumanParts:
    i = human.find(HDR_UI)
    j = human.find(HDR_SHOT)
    k = human.find(HDR_ACTION)
    if i < 0 or j < 0 or k < 0 or not (i < j < k):
        raise ParseError("human turn layout mismatch (expected UI State -> Screenshot -> Action)")

    xml_start = i + len(HDR_UI)
    xml = human[xml_start:j].strip()
    # strip() 으로 잘린 만큼을 보정해 절대 구간을 정확히 잡는다
    lead = len(human[xml_start:j]) - len(human[xml_start:j].lstrip())
    xml_span = (xml_start + lead, xml_start + lead + len(xml))

    shot_block = human[j:k].strip()
    action_block = human[k:].strip()
    return HumanParts(xml=xml, shot_block=shot_block,
                      action_block=action_block, xml_span=xml_span)


def build_human(parts: HumanParts, fmt: str, xml_out: str, image_first: bool) -> str:
    marker = INLINE_MARKER[fmt]
    body = DROPPED_BODY if fmt == "dropped" else xml_out
    ui_block = f"{marker}\n{body}"

    if image_first:
        blocks = [parts.shot_block, ui_block, parts.action_block]
    else:
        blocks = [ui_block, parts.shot_block, parts.action_block]
    return "\n\n".join(blocks)


def build_system(system: str, fmt: str) -> str:
    s = system.replace(OLD_GIVEN_LINE, NEW_GIVEN_LINE)
    if ANCHOR_OUTPUT_FMT not in s:
        raise ParseError(f"system prompt anchor not found: {ANCHOR_OUTPUT_FMT!r}")
    return s.replace(ANCHOR_OUTPUT_FMT, OBSERVABILITY[fmt] + "\n\n" + ANCHOR_OUTPUT_FMT, 1)


# --------------------------------------------------------------------------- #
# Step 1 — 분할
# --------------------------------------------------------------------------- #

def assign_formats(n: int, ratios: Tuple[float, float, float], seed: int) -> List[str]:
    """정확한 개수를 보장하는 셔플-절단 분할."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_full = int(round(n * ratios[0]))
    n_masked = int(round(n * ratios[1]))
    n_full = min(n_full, n)
    n_masked = min(n_masked, n - n_full)
    out = ["dropped"] * n
    for i in idx[:n_full]:
        out[i] = "full"
    for i in idx[n_full:n_full + n_masked]:
        out[i] = "masked"
    return out


# --------------------------------------------------------------------------- #
# 메인 파이프라인
# --------------------------------------------------------------------------- #

def get_turn(msgs: List[dict], role: str) -> dict:
    for m in msgs:
        if m.get("from") == role:
            return m
    raise ParseError(f"missing turn: {role}")


def process(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix or inp.stem

    samples: List[dict] = []
    with inp.open(encoding="utf-8") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    fmts = assign_formats(len(samples), (args.ratio_full, args.ratio_masked, args.ratio_dropped),
                          args.seed)

    raw_path = outdir / f"{stem}_split_raw.jsonl"
    app_path = outdir / f"{stem}_split_applied.jsonl"
    rep_path = outdir / f"{stem}_report.json"

    stats: dict = {
        "n_input": len(samples), "n_written": 0, "n_skipped": 0,
        "fmt_count": Counter(), "tier_used": Counter(),
        "kind_count": Counter(), "ratio_actual": [], "ratio_target": [],
        "xml_len_orig": [], "xml_len_out": [],
        "parse_fail": 0, "roundtrip_fail": 0, "layout_fail": 0, "wellformed_fail": 0,
        "mask_leak_in_target": 0, "errors": [],
    }

    f_raw = raw_path.open("w", encoding="utf-8")
    f_app = app_path.open("w", encoding="utf-8")

    for sid, (sample, fmt) in enumerate(zip(samples, fmts)):
        try:
            msgs = sample["messages"]
            sys_turn = get_turn(msgs, "system")
            hum_turn = get_turn(msgs, "human")
            gpt_turn = get_turn(msgs, "gpt")
            parts = split_human(hum_turn["value"])
        except ParseError as e:
            stats["n_skipped"] += 1
            stats["layout_fail"] += 1
            stats["errors"].append({"sample_id": sid, "stage": "layout", "msg": str(e)})
            continue

        content_hash = hashlib.sha1(gpt_turn["value"].encode("utf-8")).hexdigest()[:12]
        rng = random.Random(f"{args.seed}:{sid}")

        xml_out = parts.xml
        meta: dict = {
            "image_first": bool(args.image_first),
            "xml_len_orig": len(parts.xml),
            "xml_len_out": len(parts.xml),
            "assistant_char_len": len(gpt_turn["value"]),
            "masked_elements": [],
        }

        if fmt == "masked":
            try:
                nodes = parse_xml(parts.xml)
                if not verify_roundtrip(parts.xml, nodes):
                    raise ParseError("roundtrip verification failed")
                xml_out, r_tgt, r_act, records, tiers = mask_xml(
                    parts.xml, rng, args.ratio_lo, args.ratio_hi, args.ratio_dist,
                    min_depth=args.min_depth, overshoot=args.overshoot,
                )
                meta.update({
                    "mask_ratio_target": round(r_tgt, 4),
                    "mask_ratio_actual": round(r_act, 4),
                    "n_whole": sum(1 for r in records if r["kind"] == "whole"),
                    "n_text": sum(1 for r in records if r["kind"] == "text"),
                    "n_aria": sum(1 for r in records if r["kind"] == "aria"),
                    "tier_used": {str(k): v for k, v in sorted(tiers.items())},
                    "masked_elements": records,
                    "xml_len_out": len(xml_out),
                })
                # 마스킹 결과가 여전히 well-formed 인지 확인
                try:
                    parse_xml(xml_out)
                except ParseError as e:
                    stats["wellformed_fail"] += 1
                    stats["errors"].append(
                        {"sample_id": sid, "stage": "wellformed", "msg": str(e)})
                stats["ratio_target"].append(r_tgt)
                stats["ratio_actual"].append(r_act)
                stats["tier_used"].update(tiers)
                for r in records:
                    stats["kind_count"][r["kind"]] += 1
            except ParseError as e:
                # 파싱 실패 시 안전하게 full 로 강등 (데이터 손실 방지)
                stats["parse_fail"] += 1
                stats["errors"].append({"sample_id": sid, "stage": "parse", "msg": str(e)})
                fmt = "full"
                xml_out = parts.xml
        elif fmt == "dropped":
            meta["xml_len_out"] = 0

        # ---- 출력 조립 ----
        new_sys = build_system(sys_turn["value"], fmt)
        new_hum = build_human(parts, fmt, xml_out, args.image_first)

        if MASK_ELEM in gpt_turn["value"] or MASK_ATTR in gpt_turn["value"]:
            stats["mask_leak_in_target"] += 1

        raw_rec = {
            "messages": msgs,
            "images": sample.get("images", []),
            "sample_id": sid,
            "content_hash": content_hash,
            "fmt": fmt,
        }
        new_msgs = []
        for m in msgs:
            if m["from"] == "system":
                new_msgs.append({"from": "system", "value": new_sys})
            elif m["from"] == "human":
                new_msgs.append({"from": "human", "value": new_hum})
            else:
                new_msgs.append(dict(m))
        app_rec = {
            "messages": new_msgs,
            "images": sample.get("images", []),
            "sample_id": sid,
            "content_hash": content_hash,
            "fmt": fmt,
            "meta": meta,
        }

        f_raw.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
        f_app.write(json.dumps(app_rec, ensure_ascii=False) + "\n")

        stats["n_written"] += 1
        stats["fmt_count"][fmt] += 1
        stats["xml_len_orig"].append(meta["xml_len_orig"])
        stats["xml_len_out"].append(meta["xml_len_out"])

    f_raw.close()
    f_app.close()

    report = summarize(stats, args, str(raw_path), str(app_path))
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    return 0 if report["ok"] else 1


def _q(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return round(s[min(int(p * len(s)), len(s) - 1)], 4)


def summarize(stats: dict, args: argparse.Namespace, raw_path: str, app_path: str) -> dict:
    ra = stats["ratio_actual"]
    return {
        "ok": (stats["n_skipped"] == 0 and stats["mask_leak_in_target"] == 0
               and stats["wellformed_fail"] == 0),
        "input": args.input,
        "outputs": {"raw": raw_path, "applied": app_path},
        "config": {
            "seed": args.seed,
            "ratios": [args.ratio_full, args.ratio_masked, args.ratio_dropped],
            "mask_ratio_range": [args.ratio_lo, args.ratio_hi],
            "mask_ratio_dist": args.ratio_dist,
            "image_first": bool(args.image_first),
            "min_depth": args.min_depth,
            "overshoot": args.overshoot,
        },
        "counts": {
            "n_input": stats["n_input"],
            "n_written": stats["n_written"],
            "n_skipped": stats["n_skipped"],
            "fmt": dict(stats["fmt_count"]),
            "mask_kind": dict(stats["kind_count"]),
            "tier_used": {str(k): v for k, v in sorted(stats["tier_used"].items())},
        },
        "mask_ratio_actual": {
            "n": len(ra),
            "p10": _q(ra, 0.10), "p25": _q(ra, 0.25), "p50": _q(ra, 0.50),
            "p75": _q(ra, 0.75), "p90": _q(ra, 0.90),
            "mean": round(sum(ra) / len(ra), 4) if ra else 0.0,
        },
        "xml_len": {
            "orig_median": _q(stats["xml_len_orig"], 0.5),
            "out_median": _q(stats["xml_len_out"], 0.5),
        },
        "failures": {
            "layout_fail": stats["layout_fail"],
            "parse_fail_downgraded_to_full": stats["parse_fail"],
            "roundtrip_fail": stats["roundtrip_fail"],
            "wellformed_fail": stats["wellformed_fail"],
            "mask_leak_in_target": stats["mask_leak_in_target"],
        },
        "errors_head": stats["errors"][:20],
    }


def print_report(rep: dict) -> None:
    c = rep["counts"]
    m = rep["mask_ratio_actual"]
    f = rep["failures"]
    print("=" * 62)
    print(f"  input   : {rep['input']}")
    print(f"  raw     : {rep['outputs']['raw']}")
    print(f"  applied : {rep['outputs']['applied']}")
    print("-" * 62)
    print(f"  written {c['n_written']} / input {c['n_input']}  (skipped {c['n_skipped']})")
    print(f"  fmt        : {c['fmt']}")
    print(f"  mask kind  : {c['mask_kind']}")
    print(f"  tier used  : {c['tier_used']}")
    print(f"  mask ratio : p10 {m['p10']}  p50 {m['p50']}  p90 {m['p90']}  mean {m['mean']}")
    print(f"  xml len    : {rep['xml_len']['orig_median']} -> {rep['xml_len']['out_median']} (median chars)")
    print("-" * 62)
    print(f"  layout_fail={f['layout_fail']}  parse_fail={f['parse_fail_downgraded_to_full']}  "
          f"wellformed_fail={f['wellformed_fail']}  mask_leak={f['mask_leak_in_target']}")
    print(f"  RESULT: {'OK' if rep['ok'] else 'CHECK ERRORS'}")
    print("=" * 62)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="입력 JSONL (ShareGPT, NEXT_STATE_PREDICTION)")
    ap.add_argument("--outdir", required=True, help="출력 디렉터리")
    ap.add_argument("--prefix", default=None, help="출력 파일 접두사 (기본: 입력 파일명)")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ratio-full", type=float, default=0.25)
    ap.add_argument("--ratio-masked", type=float, default=0.55)
    ap.add_argument("--ratio-dropped", type=float, default=0.20)

    ap.add_argument("--ratio-lo", type=float, default=0.2, help="샘플별 마스킹 비율 하한")
    ap.add_argument("--ratio-hi", type=float, default=0.8, help="샘플별 마스킹 비율 상한")
    ap.add_argument("--ratio-dist", choices=["beta", "uniform"], default="beta")

    ap.add_argument("--min-depth", type=int, default=2,
                    help="요소 전체 마스킹을 허용하는 최소 depth (루트=0)")
    ap.add_argument("--overshoot", type=float, default=1.3,
                    help="남은 예산 대비 허용 초과 배수")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--image-first", dest="image_first", action="store_true", default=True,
                   help="스크린샷을 XML 앞에 배치 (기본값)")
    g.add_argument("--no-image-first", dest="image_first", action="store_false",
                   help="원본 순서(XML -> 스크린샷) 유지")

    args = ap.parse_args(argv)
    tot = args.ratio_full + args.ratio_masked + args.ratio_dropped
    if abs(tot - 1.0) > 1e-6:
        ap.error(f"ratio 합이 1이 아닙니다: {tot}")
    return process(args)


if __name__ == "__main__":
    sys.exit(main())
