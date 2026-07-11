#!/usr/bin/env python3
"""EXP01 ratio73 멤버십을 EXP05 (픽셀 좌표 + 해상도 정렬 + 프롬프트 수정) 으로 미러링한다.

EXP05 는 AndroidWorld 평가 환경과 **해상도를 정렬**한 실험군이다. AndroidControl
원본 해상도(base 1080×2400) 를 Qwen2.5-VL 의 ``smart_resize`` (factor 28) 로
``--image-budget 1605632`` 제약 하에 축소하면 좌표 공간이 840×1876 **절대 픽셀**이
된다 — 이 값은 학습 시 이미지 프로세서에 넘기는 ``max_pixels=1605632`` 와 반드시
일치해야 한다 (불일치 시 좌표-이미지 grounding 이 붕괴한다). min_pixels 는 3136.
**Qwen2.5-VL 전용 좌표 규약** 이다 — EXP03/EXP04 의 0–1000 정규화 좌표와는 다른
규약이므로 서로 바꿔 쓸 수 없다.

EXP05 원천 2종 (data/AndroidControl/) 은 EXP01 좌표 pool 을 픽셀 좌표 + 해상도
정렬로 재산출하고 프롬프트를 수정한 버전이다:
  - implicit-world-modeling_stage1_action_xy_pixel-aligned.jsonl   (action_pred)
  - implicit-world-modeling_stage1_state_xy_pixel-aligned.jsonl    (state_pred)

본 스크립트는 EXP01 ratio73 산출 파일들의 멤버십을 그대로 따라가며, 각 EXP01 레코드에
대응하는 EXP05 좌표 레코드를 동일 순서로 골라 출력한다 — EXP01/EXP02/EXP03/EXP04 와
**행 단위 1:1 대응** (표현만 픽셀 좌표 + 해상도 정렬). 출력은 EXP02 스타일 파일명으로
data/AndroidControl_EXP05/ 에 stage1 train + test 6종 (총 7 파일) 생성.
**Stage 2 는 보류** — 이번 미러 대상이 아니다 (stage1 만).

EXP05 pool 대비 누락(drop) 수치는 원천 데이터 미도착으로 미기재 — 실측 후 기입.

규칙
----
- 매칭 키 = ``images[0]`` 의 ``(episode, step)`` (int() 가 zero-pad 정규화 겸함;
  split_data.py 의 ``EPISODE_RE`` + ``_norm_ep`` 와 동일 규약).
- stage1 task 종류는 이미지 개수로 판별: 2개 = action_pred, 1개 = state_pred.
- 이미지 경로: 매칭된 EXP01 레코드의 ``images`` 를 그대로 채택 (AndroidControl/images/...).
- 본문(messages): EXP05 원천 레코드 (픽셀 좌표 + 해상도 정렬 프롬프트 유지).
- EXP05 원천에 키가 없는 EXP01 레코드는 제외한다 (EXP05 가 EXP01 대비 작을 수 있음).

Usage
-----
  python scripts/mirror_exp05.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_data import EPISODE_RE, load_jsonl, write_jsonl  # scripts/ 코드 재사용

STEP_RE = re.compile(r"step_(\d+)")

# (EXP01 입력 파일, EXP05 출력 파일, stage1 여부)
# stage1 train 은 ratio73(train_7_3) 만 미러 → EXP02 스타일 단일 train 명으로 출력.
# 나머지 test 는 EXP01 과 동일 파일명 (EXP02/EXP03/EXP04 와도 byte 구조 동일).
# Stage 2 (stage2_train/test_id/test_ood) 는 보류 — JOBS 에 포함하지 않는다.
JOBS = [
    ("implicit-world-modeling_stage1_train_7_3.jsonl",
     "implicit-world-modeling_stage1_train.jsonl", True),
    ("implicit-world-modeling_stage1_test_id_action.jsonl",
     "implicit-world-modeling_stage1_test_id_action.jsonl", True),
    ("implicit-world-modeling_stage1_test_id_state.jsonl",
     "implicit-world-modeling_stage1_test_id_state.jsonl", True),
    ("implicit-world-modeling_stage1_test_id_state_without_open_app.jsonl",
     "implicit-world-modeling_stage1_test_id_state_without_open_app.jsonl", True),
    ("implicit-world-modeling_stage1_test_ood_action.jsonl",
     "implicit-world-modeling_stage1_test_ood_action.jsonl", True),
    ("implicit-world-modeling_stage1_test_ood_state.jsonl",
     "implicit-world-modeling_stage1_test_ood_state.jsonl", True),
    ("implicit-world-modeling_stage1_test_ood_state_without_open_app.jsonl",
     "implicit-world-modeling_stage1_test_ood_state_without_open_app.jsonl", True),
]


def key_of(images: list[str]) -> tuple[int, int]:
    """images[0] → (episode, step). int() 가 zero-pad 정규화를 겸한다."""
    img = images[0]
    e = EPISODE_RE.search(img)
    s = STEP_RE.search(img)
    if not (e and s):
        raise ValueError(f"episode/step 추출 실패: {img!r}")
    return int(e.group(1)), int(s.group(1))


def build_index(path: Path) -> dict[tuple[int, int], dict]:
    """(episode, step) → 원천 레코드. 키 유일성 검증 포함."""
    idx: dict[tuple[int, int], dict] = {}
    for r in load_jsonl(path):
        k = key_of(r["images"])
        if k in idx:
            raise ValueError(f"중복 키 {k} in {path.name}")
        idx[k] = r
    return idx


def mirror(
    rows: list[dict],
    *,
    action_idx: dict[tuple[int, int], dict],
    state_idx: dict[tuple[int, int], dict],
) -> tuple[list[dict], int]:
    out: list[dict] = []
    dropped = 0
    for rec in rows:
        imgs = rec["images"]
        idx = action_idx if len(imgs) == 2 else state_idx  # 2개=action, 1개=state
        e05 = idx.get(key_of(imgs))
        if e05 is None:  # EXP05 원천에 없는 데이터 → 제외
            dropped += 1
            continue
        out.append({**e05, "images": imgs})  # 본문=EXP05 픽셀 좌표/프롬프트, 경로=EXP01
    return out, dropped


def main() -> int:
    base = Path(__file__).resolve().parent.parent / "data"
    src = base / "AndroidControl"
    exp01 = base / "AndroidControl_EXP01"
    out_dir = base / "AndroidControl_EXP05"
    out_dir.mkdir(parents=True, exist_ok=True)

    action_src_path = src / "implicit-world-modeling_stage1_action_xy_pixel-aligned.jsonl"
    state_src_path = src / "implicit-world-modeling_stage1_state_xy_pixel-aligned.jsonl"
    for p in (action_src_path, state_src_path):
        if not p.exists():
            print(
                f"[ERROR] EXP05 source 없음: {p} — Google Drive '0710_버젼' 폴더에서 "
                f"받아 위 이름으로 배치하세요.",
                file=sys.stderr,
            )
            return 1

    action_idx = build_index(action_src_path)
    state_idx = build_index(state_src_path)
    print(f"EXP05 index: action={len(action_idx)} state={len(state_idx)}")
    print(f"{'output file':58s} {'in':>6} {'out':>6} {'drop':>5}")

    total_out = 0
    for in_name, out_name, _stage1 in JOBS:
        in_path = exp01 / in_name
        if not in_path.exists():
            print(f"[ERROR] EXP01 source 없음: {in_path}", file=sys.stderr)
            return 1
        rows = load_jsonl(in_path)
        out, dropped = mirror(rows, action_idx=action_idx, state_idx=state_idx)
        write_jsonl(out, out_dir / out_name)
        total_out += len(out)
        print(f"{out_name:58s} {len(rows):6d} {len(out):6d} {dropped:5d}")

    print(f"\nDone. {len(JOBS)} files, {total_out} rows → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
