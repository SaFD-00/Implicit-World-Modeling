#!/usr/bin/env python3
"""Filter mm-expanded length > threshold 인 샘플을 사전 제거하는 도구.

배경
----
Qwen3-VL ``get_rope_index`` 는 ``cutoff_len`` 으로 input_ids 가 잘리기 *전* 의
``image_grid_thw`` 로 ``llm_positions`` 를 만들기 때문에, 원본 mm-expanded
길이 > cutoff_len 인 샘플이 학습 dataloader 에 들어오면

    RuntimeError: shape mismatch: value tensor of shape [3, X]
                  cannot be broadcast to indexing result of shape [3, cutoff_len]

로 죽는다. cutoff 만 올리는 식의 회피는 long-tail (max ≈ 39K) 때문에
보장 불가 → 학습 전에 길이 초과 샘플을 데이터셋에서 빼는 것이 근본책이다.

대상 (필터): AC_EXP01 / AC_EXP02 (원본 ``data/AndroidControl/``) Stage 1 두 source
(``implicit-world-modeling_stage1_{state,action}_pred.jsonl``) + Stage 2 source
(``implicit-world-modeling_stage2.jsonl``).
출력: 같은 디렉토리 (``data/AndroidControl/``) 에 ``*_filtered.jsonl`` (3 파일).

대상 (측정, ``--report-only``): AC_EXP03 (좌표 미러 ``data/AndroidControl_EXP03/``) train
산출물. EXP03 는 무손실 정책상 필터링하지 않고, 선택한 ``cutoff_len`` 이 전 샘플을
덮는지 (over-threshold=0) 확인하는 용도로만 측정한다.

길이 계산
---------
1. 각 sample 의 image 를 PIL 로 열어 (W, H) 를 얻고,
2. Qwen3-VL ``smart_resize`` (factor = patch_size * merge_size, min/max pixels
   는 stage1 yaml 과 동일한 값) 로 grid_thw 를 결정,
3. ``vision_tokens = (grid_h * grid_w) // merge_size**2``,
4. chat template wrapper (``<|im_start|>...``) 토큰과 system/user/gpt 본문을
   별도로 tokenize, image placeholder 자리에 ``<|vision_start|>`` +
   N×``<|image_pad|>`` + ``<|vision_end|>`` 를 합산해 mm-expanded 총 길이를
   얻는다 (학습 시 collator 가 만드는 input_ids 길이와 일치).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

# split_data.py 의 IO helper 재사용 (같은 scripts/ 디렉토리).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_data import load_jsonl, write_jsonl  # noqa: E402


# AC_EXP01 / AC_EXP02 의 source 자산은 모두 원본 폴더 data/AndroidControl/ 에 있다
# (split_data.py 와 동일한 source 분리 정책). _filtered.jsonl 산출물도 같은 폴더에 쓴다.
# AC_EXP03 (좌표 미러) 는 산출물 폴더 data/AndroidControl_EXP03/ 를 가리키며,
# 무손실 정책상 필터링이 아니라 --report-only 측정 (cutoff_len 이 전 샘플을 덮는지 확인) 용도다.
DATASET_TO_DIR = {
    "AC_EXP01": "AndroidControl",
    "AndroidControl_EXP01": "AndroidControl",
    "AC_EXP02": "AndroidControl",
    "AndroidControl_EXP02": "AndroidControl",
    "AC_EXP03": "AndroidControl_EXP03",
    "AndroidControl_EXP03": "AndroidControl_EXP03",
}

# 데이터셋 디렉토리별 측정/필터 대상 source 목록.
# AndroidControl: AC_EXP01/02 의 원본 3 종 (필터 → _filtered).
# AndroidControl_EXP03: 좌표 미러 train 산출물 2 종 (측정 전용 — 10%/크래시 통계의 출처).
SOURCES_BY_DIR = {
    "AndroidControl": [
        "implicit-world-modeling_stage1_state_pred.jsonl",
        "implicit-world-modeling_stage1_action_pred.jsonl",
        "implicit-world-modeling_stage2.jsonl",
    ],
    "AndroidControl_EXP03": [
        "implicit-world-modeling_stage1_train.jsonl",
        "implicit-world-modeling_stage2_train.jsonl",
    ],
}

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Qwen3-VL chat template wrapper 토큰 (qwen3_vl_nothink template 와 동일).
SYS_PREFIX = "<|im_start|>system\n"
USR_PREFIX = "<|im_start|>user\n"
ASS_PREFIX = "<|im_start|>assistant\n"
TURN_SUFFIX = "<|im_end|>\n"
VBOS = "<|vision_start|>"
VEOS = "<|vision_end|>"


# ── smart_resize (Qwen3-VL 공식 알고리즘 단순 포팅) ───────────────────────
def smart_resize(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def vision_tokens_for_size(
    w: int, h: int, *, patch_size: int, merge_size: int, min_pixels: int, max_pixels: int
) -> int:
    factor = patch_size * merge_size
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    return (grid_h * grid_w) // (merge_size ** 2)


# ── Length computation ────────────────────────────────────────────────────
def build_length_fn(processor, *, image_max_pixels: int, image_min_pixels: int):
    tok = processor.tokenizer
    ip = processor.image_processor
    patch_size = ip.patch_size
    merge_size = ip.merge_size

    def encode_len(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    wrap = {
        "sys_pre": encode_len(SYS_PREFIX),
        "usr_pre": encode_len(USR_PREFIX),
        "ass_pre": encode_len(ASS_PREFIX),
        "suf": encode_len(TURN_SUFFIX),
        "vbos": encode_len(VBOS),
        "veos": encode_len(VEOS),
    }
    print(f"[len-fn] wrapper token costs: {wrap}", flush=True)

    size_cache: dict[tuple[int, int], int] = {}

    def vt_for(size: tuple[int, int]) -> int:
        if size not in size_cache:
            size_cache[size] = vision_tokens_for_size(
                *size,
                patch_size=patch_size,
                merge_size=merge_size,
                min_pixels=image_min_pixels,
                max_pixels=image_max_pixels,
            )
        return size_cache[size]

    def length_of(entry: dict, media_dir: Path) -> int | None:
        msgs = {m["from"]: m["value"] for m in entry["messages"]}
        sys_text = msgs.get("system", "")
        usr_text = msgs.get("human", "")
        gpt_text = msgs.get("gpt", "")

        rels = entry.get("images") or []
        if not rels:
            # 이미지 없으면 텍스트만 길이로 계산 (placeholder 도 없을 것).
            total = (
                (wrap["sys_pre"] + encode_len(sys_text) + wrap["suf"] if sys_text else 0)
                + wrap["usr_pre"] + encode_len(usr_text) + wrap["suf"]
                + wrap["ass_pre"] + encode_len(gpt_text) + wrap["suf"]
            )
            return total

        full = media_dir / rels[0]
        try:
            with Image.open(full) as im:
                size = im.size  # (W, H)
        except Exception as exc:
            print(f"[warn] image open failed: {full} ({exc}) — skipping", file=sys.stderr)
            return None

        n_img = usr_text.count("<image>")
        usr_minus_img = usr_text.replace("<image>", "")
        vt = vt_for(size)

        total = (
            (wrap["sys_pre"] + encode_len(sys_text) + wrap["suf"] if sys_text else 0)
            + wrap["usr_pre"]
            + encode_len(usr_minus_img)
            + n_img * (wrap["vbos"] + vt + wrap["veos"])
            + wrap["suf"]
            + wrap["ass_pre"] + encode_len(gpt_text) + wrap["suf"]
        )
        return total

    return length_of


# ── Filter driver ─────────────────────────────────────────────────────────
def filter_jsonl(
    src: Path,
    dst: Path,
    media_dir: Path,
    length_of,
    threshold: int,
    *,
    report_only: bool = False,
) -> dict:
    rows = load_jsonl(src)
    kept: list[dict] = []
    dropped_lengths: list[int] = []
    all_lengths: list[int] = []
    skipped = 0
    for i, entry in enumerate(rows):
        if i % 5000 == 0:
            print(f"  [{src.name}] {i}/{len(rows)} ... kept={len(kept)} dropped={len(dropped_lengths)}", flush=True)
        L = length_of(entry, media_dir)
        if L is None:
            skipped += 1
            continue
        all_lengths.append(L)
        if L > threshold:
            dropped_lengths.append(L)
        else:
            kept.append(entry)

    # report-only 모드는 측정만 — _filtered 산출물을 쓰지 않는다 (무손실 정책).
    if not report_only:
        write_jsonl(kept, dst)

    info = {
        "src": str(src),
        "dst": str(dst),
        "in": len(rows),
        "out": len(kept),
        "dropped": len(dropped_lengths),
        "skipped": skipped,
        "drop_pct": (len(dropped_lengths) / max(len(rows), 1)) * 100,
        "max": max(all_lengths) if all_lengths else 0,
    }
    if dropped_lengths:
        dropped_lengths.sort()
        info["dropped_min"] = dropped_lengths[0]
        info["dropped_p50"] = dropped_lengths[len(dropped_lengths) // 2]
        info["dropped_max"] = dropped_lengths[-1]
    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AC_EXP01 source jsonl 에서 mm-expanded length > threshold 샘플을 제거.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_TO_DIR), default="AC_EXP01")
    parser.add_argument("--data-dir", default=None, help="Data root (default: <repo>/data)")
    parser.add_argument("--threshold", type=int, default=10000,
                        help="mm-expanded length 기준 컷 (default 10000 = stage1/2 cutoff_len).")
    parser.add_argument("--image-max-pixels", type=int, default=2097152,
                        help="stage1/2 yaml 의 image_max_pixels 와 일치해야 함. "
                             "Default 2097152 는 Qwen3-VL family (factor 32, 2048 tokens) 기준. "
                             "Qwen2/2.5-VL (factor 28) 학습 시 1605632 등으로 override.")
    parser.add_argument("--image-min-pixels", type=int, default=4096,
                        help="stage1 yaml 의 image_min_pixels 와 일치해야 함 (default 4096).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Processor / tokenizer model id.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="대응하는 _filtered.jsonl 이 이미 있으면 그 source 를 처리하지 않음.")
    parser.add_argument("--report-only", action="store_true",
                        help="길이 분포만 측정/출력하고 _filtered.jsonl 은 쓰지 않음 "
                             "(AC_EXP03 무손실 검증용: threshold 가 전 샘플을 덮는지 확인).")
    args = parser.parse_args()

    if args.data_dir:
        data_root = Path(args.data_dir)
    else:
        data_root = Path(__file__).resolve().parent.parent / "data"
    ds_dir = data_root / DATASET_TO_DIR[args.dataset]
    if not ds_dir.exists():
        print(f"[ERROR] dataset dir not found: {ds_dir}", file=sys.stderr)
        return 1

    sources = [ds_dir / name for name in SOURCES_BY_DIR[DATASET_TO_DIR[args.dataset]]]
    for p in sources:
        if not p.exists():
            print(f"[ERROR] source not found: {p}", file=sys.stderr)
            return 1

    pending: list[Path] = []
    for src in sources:
        dst = src.with_name(src.stem + "_filtered" + src.suffix)
        # report-only 는 산출물을 쓰지 않으므로 _filtered 존재 여부로 skip 하지 않는다.
        if args.skip_existing and not args.report_only and dst.exists():
            print(f"[skip] {dst.name} already exists (--skip-existing).")
            continue
        pending.append(src)
    if not pending:
        print("[done] 모든 source 의 _filtered.jsonl 이 이미 존재합니다.")
        return 0

    print(f"Dataset: {args.dataset} ({ds_dir})")
    print(f"Threshold: {args.threshold}")
    print(f"image_max_pixels={args.image_max_pixels} image_min_pixels={args.image_min_pixels}")
    print(f"Loading processor: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    length_of = build_length_fn(
        processor,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
    )

    if args.report_only:
        print("[report-only] 측정만 수행 — _filtered.jsonl 을 쓰지 않습니다.")
    print()
    summaries: list[dict] = []
    for src in pending:
        dst = src.with_name(src.stem + "_filtered" + src.suffix)
        label = "report" if args.report_only else "filter"
        arrow = "(측정)" if args.report_only else f"→ {dst.name}"
        print(f"[{label}] {src.name} {arrow}", flush=True)
        info = filter_jsonl(src, dst, data_root, length_of, args.threshold,
                            report_only=args.report_only)
        summaries.append(info)
        verb = "over-threshold" if args.report_only else "dropped"
        print(
            f"  in={info['in']} {verb}={info['dropped']} "
            f"({info['drop_pct']:.2f}%) max={info['max']} skipped={info['skipped']}",
            flush=True,
        )
        if info["dropped"] > 0:
            print(
                f"  over-threshold length: min={info['dropped_min']} "
                f"p50={info['dropped_p50']} max={info['dropped_max']}",
                flush=True,
            )
        print()

    print("=== Summary ===")
    overall_max = max((info["max"] for info in summaries), default=0)
    overall_over = sum(info["dropped"] for info in summaries)
    for info in summaries:
        print(
            f"  {Path(info['src']).name}: in={info['in']} "
            f"over-threshold={info['dropped']} ({info['drop_pct']:.2f}%) max={info['max']}"
        )
    if args.report_only:
        verdict = "OK (전 샘플 ≤ threshold — 무손실)" if overall_over == 0 else "초과 샘플 존재 → cutoff 상향 필요"
        print(f"  overall: max={overall_max} threshold={args.threshold} over={overall_over} → {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
