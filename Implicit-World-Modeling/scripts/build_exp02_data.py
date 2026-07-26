#!/usr/bin/env python3
"""AC_EXP02 stage1 train 데이터 정본 빌더 — diff loss token_weights 부여 (v1).

이 스크립트가 **AC_EXP02 데이터의 유일한 커밋된 생성 경로**다 (예전엔 노트북
Cell 7 이 그 역할이었고, 노트북 은퇴와 함께 그대로 이관됐다). AC_EXP02 는
AC_EXP01 ratio73 과 **동일한 학습 데이터/하이퍼파라미터**이되, Stage 1 에만
diff token weighted loss 를 얹은 대조 실험군이다.

두 부분으로 나뉜다 (둘 다 멱등):

  1. stage1_train.jsonl — AC_EXP01 ratio73 train 에 ``scripts/diff_loss/preprocess_dataset.py``
     (**v1**) 로 ``token_weights`` 를 부여한 것. 가중치 W = (added 2.0, modified 2.0,
     unchanged 1.0).
     ⚠ **v1 고정 불변식**: 저장된 EXP02 train 의 token_weights 와 v1 재실행은 40/40
     일치한다 (v2 로 돌리면 17/40). v1 의 경계 비대칭 버그도 **EXP02 재현성 보존을
     위해 의도적으로 고치지 않는다** (ARCHITECTURE §3 함정 10). 따라서 이 빌더는
     preprocess_dataset**_v2**.py 를 절대 쓰지 않는다.

  2. test / Stage 2 파일 (7종) — AC_EXP01 에서 복사한다. AC_EXP01 ratio73 과 동일
     평가셋이어야 공정 비교가 성립하기 때문이다.

Diff loss 자체(LlamaFactory 패치)는 이 스크립트가 하지 않는다 —
``scripts/setup_llamafactory.sh`` 가 ``patches/llamafactory/*.patch`` 로 적용한다.

Usage
-----
  .venv/bin/python scripts/build_exp02_data.py            # 없을 때만 생성 (멱등)
  .venv/bin/python scripts/build_exp02_data.py --force    # train 재생성 강제
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from implicit_world_modeling.lf_registry import BASE_DIR, _MODEL_CONFIG

# W 상수 — v1 preprocess 에 넘길 diff token 가중치 (added/modified/unchanged).
# ⚠ 값을 바꾸면 저장된 EXP02 train 과 어긋난다 (재현성 파손). 함정 10 참조.
_W_ADDED = "2.0"
_W_MODIFIED = "2.0"
_W_UNCHANGED = "1.0"

# AC_EXP01 → AC_EXP02 로 그대로 복사하는 test / Stage 2 파일 (동일 평가셋).
_SHARED_FILES = [
    "stage1_test_id_state.jsonl",
    "stage1_test_ood_state.jsonl",
    "stage1_test_id_action.jsonl",
    "stage1_test_ood_action.jsonl",
    "stage2_train.jsonl",
    "stage2_test_id.jsonl",
    "stage2_test_ood.jsonl",
]


def build(base_dir: Path, force: bool) -> None:
    ac1_dir = base_dir / "data" / "AndroidControl_EXP01"
    ac2_dir = base_dir / "data" / "AndroidControl_EXP02"
    ac2_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = base_dir / "scripts" / "diff_loss"

    # 무효 셸 HF_TOKEN 회피 (stored 토큰 사용) — 노트북 Cell 7 과 동일.
    env = {**os.environ, "HF_TOKEN": ""}

    # 1. stage1_train: AC_EXP01 ratio73 train 에 token_weights 부여 (v1).
    ac2_train = ac2_dir / "stage1_train.jsonl"
    ac1_ratio73 = ac1_dir / "stage1_train_ratio73.jsonl"
    if ac2_train.exists() and not force:
        print(f"[skip] {ac2_train.name} 이미 존재 (--force 로 재생성)")
    else:
        if not ac1_ratio73.exists():
            sys.exit(
                f"[!] 입력 없음: {ac1_ratio73}\n"
                f"    AC_EXP01 ratio73 train 을 먼저 생성하세요 "
                f"(scripts/split_data.py --dataset AC_EXP01)."
            )
        model_id = _MODEL_CONFIG["qwen3-vl-8b"]["model_id"]
        print(f"[preprocess] AC_EXP01 ratio73 train -> AC_EXP02 (diff loss token_weights, v1) ...")
        print(f"             model={model_id}  W=(added {_W_ADDED}, modified {_W_MODIFIED}, unchanged {_W_UNCHANGED})")
        subprocess.run(
            [
                sys.executable, str(dl_dir / "preprocess_dataset.py"),
                "--input", str(ac1_ratio73),
                "--output", str(ac2_train),
                "--model", model_id,
                "--w-added", _W_ADDED,
                "--w-modified", _W_MODIFIED,
                "--w-unchanged", _W_UNCHANGED,
            ],
            check=True,
            env=env,
        )

    # 2. test / Stage 2 파일: AC_EXP01 에서 복사 (동일 평가셋 → 공정 비교).
    for fname in _SHARED_FILES:
        dst = ac2_dir / fname
        if dst.exists() and not force:
            print(f"[skip] {fname}")
        else:
            src = ac1_dir / fname
            if not src.exists():
                sys.exit(f"[!] 복사 원본 없음: {src}")
            shutil.copy2(src, dst)
            print(f"[copy] {fname}")

    print("[완료] AC_EXP02 데이터 준비 — dataset_info 정본은 configs/lf_dataset/dataset_info.json (커밋됨)")


def main() -> None:
    parser = argparse.ArgumentParser(description="AC_EXP02 stage1 데이터 정본 빌더 (diff loss v1).")
    parser.add_argument(
        "--base-dir",
        default=str(BASE_DIR),
        help="repo 루트 (기본: lf_registry.BASE_DIR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 존재해도 train + 복사 파일을 재생성한다.",
    )
    args = parser.parse_args()
    build(Path(args.base_dir), args.force)


if __name__ == "__main__":
    main()
