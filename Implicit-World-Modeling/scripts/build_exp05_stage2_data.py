#!/usr/bin/env python3
"""AC_EXP05 stage2 학습 데이터 정본 빌더 — drive 원천 → 저장소 이미지 경로 관례 변환.

이 스크립트가 **AC_EXP05 stage2 3개 jsonl 의 유일한 커밋된 생성 경로**다.
drive-download 원천은 gitignored(재클론 시 사라짐)이므로, 산출물의 재현 근거를
``<train>.jsonl.meta.json`` sidecar 에 원천 경로/레코드 수/home 해시로 남긴다.

파이프라인
----------
1. ``home.png`` (RGBA) → ``home.jpg`` (RGB JPEG) 로 변환해 공용 이미지 폴더에 주입
   (리사이즈 금지 — 원본 1080×2400 그대로).
2. stage2 3파일의 각 레코드 ``images[]`` 원소만 저장소 경로 관례로 치환:
     - ``myset/images/home.jpg``                     → ``AndroidControl/images/home.jpg``
     - ``myset/images/episode_(\\d+)_step_(\\d+).jpg`` → ``AndroidControl/images/episode_{ep:06d}_step_{step}.jpg``
       (episode 는 int 파싱 후 6자리 zero-pad, step 문자열은 원형 유지)
     - 그 외 패턴은 **즉시 abort (fail-closed)**.
   메시지 value/순서/바이트는 불변 — ``myset`` 은 오직 ``images[]`` 에만 있으므로
   raw 라인의 해당 ref 문자열만 국소 치환한다 (json 재직렬화로 인한 바이트 표류 회피).

**stage2 는 무가중**: diff-loss/token_weights 는 stage1 전용이다. 이 스크립트는
어떤 가중치도 부여하지 않으며, 산출 레코드 키는 원천 그대로 ``{messages, images}`` 뿐이다.

Usage
-----
  python scripts/build_exp05_stage2_data.py
  python scripts/build_exp05_stage2_data.py --quality 95
  python scripts/build_exp05_stage2_data.py --src <drive-dir> --data-root <repo/data>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJ = SCRIPTS.parent

# ── 경로 상수 ────────────────────────────────────────────────────────────────
DEFAULT_SRC = Path(
    "/home/seungwoo.baek/projects/Implicit-World-Modeling/"
    ".claude/drive-download-20260715T072828Z-1-001"
)
DEFAULT_DATA_ROOT = PROJ / "data"

# drive 원본명 → 정본 산출명 매핑 (레코드 수는 fail-closed 검증에서 대조)
FILE_MAP = {
    "stage2_train.jsonl": "implicit-world-modeling_stage2_train.jsonl",
    "stage2_test_id.jsonl": "implicit-world-modeling_stage2_test_id.jsonl",
    "stage2_test_ood.jsonl": "implicit-world-modeling_stage2_test_ood.jsonl",
}
EXPECTED_COUNTS = {
    "implicit-world-modeling_stage2_train.jsonl": 15000,
    "implicit-world-modeling_stage2_test_id.jsonl": 3000,
    "implicit-world-modeling_stage2_test_ood.jsonl": 3000,
}
TRAIN_OUT = "implicit-world-modeling_stage2_train.jsonl"

OUT_SUBDIR = "AndroidControl_EXP05"
IMG_SUBDIR = "AndroidControl/images"

HOME_SRC = "myset/images/home.jpg"
HOME_DST = f"{IMG_SUBDIR}/home.jpg"
_EP_RE = re.compile(r"^myset/images/episode_(\d+)_step_(\d+)\.jpg$")


def convert_ref(ref: str) -> str:
    """단일 이미지 ref 를 저장소 경로 관례로 변환. 미지 패턴은 fail-closed abort."""
    if ref == HOME_SRC:
        return HOME_DST
    m = _EP_RE.match(ref)
    if m:
        return f"{IMG_SUBDIR}/episode_{int(m.group(1)):06d}_step_{m.group(2)}.jpg"
    raise SystemExit(f"[ERROR] 미지의 이미지 ref 패턴 (fail-closed abort): {ref!r}")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_home_jpg(src_png: Path, dst_jpg: Path, quality: int) -> None:
    """home.png(RGBA) → home.jpg(RGB JPEG). 리사이즈 없음, 존재 시 덮어쓰기."""
    from PIL import Image  # noqa: PLC0415

    if not src_png.is_file():
        raise SystemExit(f"[ERROR] 없음: {src_png}")
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_png) as im:
        im.convert("RGB").save(dst_jpg, "JPEG", quality=quality)
    print(f"[home] {src_png.name} → {dst_jpg}")


def convert_file(src_path: Path, dst_path: Path) -> int:
    """한 파일을 변환해 산출. messages 바이트 불변(raw-line 국소 치환), 레코드 순서 보존.

    반환: 산출 레코드 수. images 는 파싱해 새 ref 를 계산하되, 실제 라인 편집은
    원 ref 문자열의 국소 치환으로 수행해 messages value 를 바이트 그대로 남긴다.
    """
    n = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open(encoding="utf-8") as fin, dst_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            keys = set(rec.keys())
            if keys != {"messages", "images"}:
                raise SystemExit(
                    f"[ERROR] {src_path.name}:{lineno} 예상외 키 {sorted(keys)} "
                    "(stage2 는 messages+images 만 허용)"
                )
            if "token_weights" in rec:  # stage2 무가중 계약 방어
                raise SystemExit(f"[ERROR] {src_path.name}:{lineno} token_weights 존재 — stage2 금지")

            new_line = line
            for old_ref in rec["images"]:
                new_ref = convert_ref(old_ref)
                cnt = new_line.count(old_ref)
                if cnt != 1:
                    raise SystemExit(
                        f"[ERROR] {src_path.name}:{lineno} ref {old_ref!r} 출현 {cnt}회 "
                        "(레코드당 정확히 1회 기대 — 국소 치환 안전성 위반)"
                    )
                new_line = new_line.replace(old_ref, new_ref)
            fout.write(new_line)
            n += 1
    return n


# ── fail-closed 검증 (실패 시 sys.exit 비정상 종료) ──────────────────────────
def verify_outputs(
    data_root: Path,
    out_dir: Path,
    src_counts: dict[str, int],
    home_dst: Path,
) -> None:
    from PIL import Image  # noqa: PLC0415

    problems: list[str] = []
    all_paths: set[str] = set()

    for out_name, expected in EXPECTED_COUNTS.items():
        out_path = out_dir / out_name
        n = 0
        n_myset = 0
        n_badtok = 0
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                n += 1
                if "myset" in line:
                    n_myset += 1
                rec = json.loads(line)
                imgs = rec["images"]
                for im in imgs:
                    all_paths.add(im)
                tok = sum(m.get("value", "").count("<image>") for m in rec["messages"])
                if tok != len(imgs):
                    n_badtok += 1

        # (a) 입력 대비 출력 레코드 수 동일 + 기대치 일치
        src_name = next(k for k, v in FILE_MAP.items() if v == out_name)
        if n != expected or n != src_counts[src_name]:
            problems.append(
                f"(a) {out_name}: 출력 {n} != 기대 {expected} / 입력 {src_counts[src_name]}"
            )
        # (b) myset 0건
        if n_myset:
            problems.append(f"(b) {out_name}: 'myset' {n_myset}건 잔존")
        # (d) <image> 토큰 수 == len(images)
        if n_badtok:
            problems.append(f"(d) {out_name}: <image>!=len(images) {n_badtok}건")

    # (c) home.jpg 주입 후 — 모든 변환 경로가 실파일로 resolve (누락 0)
    missing = [p for p in all_paths if not (data_root / p).is_file()]
    if missing:
        problems.append(f"(c) 미존재 이미지 {len(missing)}건 (예: {missing[:3]})")

    # (e) home.jpg 가 PIL 로 열리고 mode==RGB, format==JPEG
    try:
        with Image.open(home_dst) as im:
            if im.mode != "RGB" or im.format != "JPEG":
                problems.append(f"(e) home.jpg mode={im.mode} format={im.format} (RGB/JPEG 아님)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"(e) home.jpg PIL open 실패: {e}")

    print("\n── fail-closed 검증 ─────────────────────────────────────")
    print(f"  (a) 레코드 수: {[EXPECTED_COUNTS[k] for k in EXPECTED_COUNTS]} 대조")
    print(f"  (b) myset 잔존: 0")
    print(f"  (c) 고유 이미지 경로 {len(all_paths)}건 전량 resolve (home 포함), 누락 0")
    print(f"  (d) <image> 토큰 수 == len(images)")
    print(f"  (e) home.jpg = RGB JPEG")
    if problems:
        print("\n  [FAIL] 검증 위반:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(1)
    print("  [OK] 검증 5종 전부 통과")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="AC_EXP05 stage2 데이터 빌드 (경로 변환 + home.jpg 주입, 무가중)"
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="drive-download 원천 디렉토리")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="data 루트 (기본: repo/data)")
    p.add_argument("--quality", type=int, default=95, help="home.jpg JPEG 품질 (기본 95)")
    args = p.parse_args(argv)

    src: Path = args.src
    data_root: Path = args.data_root
    out_dir = data_root / OUT_SUBDIR
    home_src = src / "home.png"
    home_dst = data_root / HOME_DST

    if not src.is_dir():
        print(f"[ERROR] 원천 디렉토리 없음: {src}", file=sys.stderr)
        return 1

    # 1) home.jpg 주입 (검증 (c) 가 home ref 를 resolve 하려면 먼저 존재해야 함)
    build_home_jpg(home_src, home_dst, args.quality)

    # 2) 3파일 변환
    src_counts: dict[str, int] = {}
    for src_name, out_name in FILE_MAP.items():
        src_path = src / src_name
        if not src_path.is_file():
            print(f"[ERROR] 원천 없음: {src_path}", file=sys.stderr)
            return 1
        n = convert_file(src_path, out_dir / out_name)
        src_counts[src_name] = n
        print(f"[conv] {src_name} → {out_name}  ({n} 행)")

    # 3) fail-closed 검증
    verify_outputs(data_root, out_dir, src_counts, home_dst)

    # 4) provenance sidecar (train 에만; 타임스탬프 없음 — 재현성)
    meta = {
        "src_path": str(src),
        "per_file_record_counts": {
            out_name: EXPECTED_COUNTS[out_name] for out_name in EXPECTED_COUNTS
        },
        "home_png_sha256": sha256_of(home_src),
        "home_jpg_sha256": sha256_of(home_dst),
    }
    meta_path = out_dir / (TRAIN_OUT + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[meta] {meta_path}")

    print(f"\nDone. → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
