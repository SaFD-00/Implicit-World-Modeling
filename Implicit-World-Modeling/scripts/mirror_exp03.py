#!/usr/bin/env python3
"""EXP01 ratio73 멤버십을 EXP03 (좌표/point 표현) 으로 미러링한다.

EXP03 원천 3종 (data/AndroidControl/) 은 EXP01 과 **동일한 (episode, step) 전이의
좌표 표현 변형** 이다:
  - implicit-world-modeling_stage1_action_xy.jsonl   (action_pred)
  - implicit-world-modeling_stage1_state_xy.jsonl    (state_pred)
  - implicit-world-modeling_stage2_xy.jsonl          (stage2)
UI 트리가 ``index="N"`` 대신 ``bounds="[..]" point="[cx,cy]"`` 이고, 액션이 ``index``
대신 ``point`` 이며, 이미지 경로 prefix 가 ``myset/images/`` (zero-pad 없음) 이다.

본 스크립트는 EXP01 ratio73 산출 파일들의 멤버십을 그대로 따라가며, 각 EXP01 레코드에
대응하는 EXP03 좌표 레코드를 동일 순서로 골라 출력한다 — EXP01/EXP02 와 **행 단위
1:1 대응** (표현만 좌표). 출력은 EXP02 스타일 파일명으로 data/AndroidControl_EXP03/ 에
stage1 train + test 6종 + stage2 train/test 3종 (총 10 파일) 생성.

규칙
----
- 매칭 키 = ``images[0]`` 의 ``(episode, step)`` (int() 가 zero-pad 정규화 겸함;
  split_data.py 의 ``EPISODE_RE`` + ``_norm_ep`` 와 동일 규약).
- stage1 task 종류는 이미지 개수로 판별: 2개 = action_pred, 1개 = state_pred.
- 이미지 경로: 매칭된 EXP01 레코드의 ``images`` 를 그대로 채택 (AndroidControl/images/...).
- 본문(messages): EXP03 원천 레코드 (좌표 point 표현 유지).
- EXP03 원천에 키가 없는 EXP01 레코드는 제외한다 (EXP03 가 ~0.8~1.7% 작음).

Usage
-----
  python scripts/mirror_exp03.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_data import EPISODE_RE, load_jsonl, write_jsonl  # scripts/ 코드 재사용

STEP_RE = re.compile(r"step_(\d+)")

# (EXP01 입력 파일, EXP03 출력 파일, stage1 여부)
# stage1 train 은 ratio73(train_7_3) 만 미러 → EXP02 스타일 단일 train 명으로 출력.
# 나머지 test / stage2 는 EXP01 과 동일 파일명 (EXP02 와도 byte 구조 동일).
JOBS = [
    ("implicit-world-modeling_stage1_train_7_3.jsonl",
     "implicit-world-modeling_stage1_train.jsonl", True),
    ("implicit-world-modeling_stage1_test_id_action_pred.jsonl",
     "implicit-world-modeling_stage1_test_id_action_pred.jsonl", True),
    ("implicit-world-modeling_stage1_test_id_state_pred.jsonl",
     "implicit-world-modeling_stage1_test_id_state_pred.jsonl", True),
    ("implicit-world-modeling_stage1_test_id_state_pred_without_open_app.jsonl",
     "implicit-world-modeling_stage1_test_id_state_pred_without_open_app.jsonl", True),
    ("implicit-world-modeling_stage1_test_ood_action_pred.jsonl",
     "implicit-world-modeling_stage1_test_ood_action_pred.jsonl", True),
    ("implicit-world-modeling_stage1_test_ood_state_pred.jsonl",
     "implicit-world-modeling_stage1_test_ood_state_pred.jsonl", True),
    ("implicit-world-modeling_stage1_test_ood_state_pred_without_open_app.jsonl",
     "implicit-world-modeling_stage1_test_ood_state_pred_without_open_app.jsonl", True),
    ("implicit-world-modeling_stage2_train.jsonl",
     "implicit-world-modeling_stage2_train.jsonl", False),
    ("implicit-world-modeling_stage2_test_id.jsonl",
     "implicit-world-modeling_stage2_test_id.jsonl", False),
    ("implicit-world-modeling_stage2_test_ood.jsonl",
     "implicit-world-modeling_stage2_test_ood.jsonl", False),
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
    stage1: bool,
    action_idx: dict[tuple[int, int], dict],
    state_idx: dict[tuple[int, int], dict],
    stage2_idx: dict[tuple[int, int], dict],
) -> tuple[list[dict], int]:
    out: list[dict] = []
    dropped = 0
    for rec in rows:
        imgs = rec["images"]
        if stage1:
            idx = action_idx if len(imgs) == 2 else state_idx
        else:
            idx = stage2_idx
        e03 = idx.get(key_of(imgs))
        if e03 is None:  # EXP03 원천에 없는 데이터 → 제외
            dropped += 1
            continue
        out.append({**e03, "images": imgs})  # 본문=EXP03 좌표, 경로=EXP01
    return out, dropped


def main() -> int:
    base = Path(__file__).resolve().parent.parent / "data"
    src = base / "AndroidControl"
    exp01 = base / "AndroidControl_EXP01"
    out_dir = base / "AndroidControl_EXP03"
    out_dir.mkdir(parents=True, exist_ok=True)

    action_idx = build_index(src / "implicit-world-modeling_stage1_action_xy.jsonl")
    state_idx = build_index(src / "implicit-world-modeling_stage1_state_xy.jsonl")
    stage2_idx = build_index(src / "implicit-world-modeling_stage2_xy.jsonl")
    print(
        f"EXP03 index: action={len(action_idx)} "
        f"state={len(state_idx)} stage2={len(stage2_idx)}"
    )
    print(f"{'output file':58s} {'in':>6} {'out':>6} {'drop':>5}")

    total_out = 0
    for in_name, out_name, stage1 in JOBS:
        in_path = exp01 / in_name
        if not in_path.exists():
            print(f"[ERROR] EXP01 source 없음: {in_path}", file=sys.stderr)
            return 1
        rows = load_jsonl(in_path)
        out, dropped = mirror(
            rows,
            stage1=stage1,
            action_idx=action_idx,
            state_idx=state_idx,
            stage2_idx=stage2_idx,
        )
        write_jsonl(out, out_dir / out_name)
        total_out += len(out)
        print(f"{out_name:58s} {len(rows):6d} {len(out):6d} {dropped:5d}")

    print(f"\nDone. {len(JOBS)} files, {total_out} rows → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
