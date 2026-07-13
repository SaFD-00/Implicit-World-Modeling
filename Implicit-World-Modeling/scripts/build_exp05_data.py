#!/usr/bin/env python3
"""AC_EXP05 학습 데이터 정본 빌더 — mirror → diff-loss 가중치 부여까지 한 번에.

이 스크립트가 **AC_EXP05 `stage1_train.jsonl` 의 유일한 커밋된 생성 경로**다.
(이전에는 v2 diff-loss 체인의 호출자가 저장소에 없어 out-of-band 실행에 의존했고,
fresh clone 이 EXP05 학습 데이터를 재현할 수 없었다.)

파이프라인
----------
1. ``mirror_experiment.py --experiment exp05``
   → EXP01 ratio73 멤버십을 EXP05 픽셀 좌표 원천으로 미러 (stage1 7 파일)
2. ``diff_loss/preprocess_dataset_v2.py``
   → 미러된 train 에 ``token_weights`` / ``_diff_counts`` 인라인 부여 후 **원자 교체**

**v2 고정**: EXP05 HTML 에는 ``index`` 속성이 없어 v1 builder 를 쓰면 pos_map 이 비어
모든 토큰이 baseline 으로 방치된다 (에러 없이 diff loss 가 조용히 무력화). 따라서
metric/builder 는 v2 로 하드 고정하며 CLI 로 낮출 수 없다.

**tokenizer**: Qwen2.5-VL 3B/7B 는 동일 tokenizer 이므로 3B 로 가중치를 만들어 두 모델에
공용한다. 재현성을 위해 ``--revision`` 으로 commit SHA 를 고정할 수 있고, 실제 사용된
revision 은 산출물 sidecar (``<train>.meta.json``) 에 기록된다.

Usage
-----
  python scripts/build_exp05_data.py
  python scripts/build_exp05_data.py --skip-mirror          # 가중치만 다시 부여
  python scripts/build_exp05_data.py --revision <commit-sha>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "diff_loss"))

# ── EXP05 불변식 (CLI 로 바꿀 수 없다) ──────────────────────────────────────
# 가중 체계: diff(ADDED/MODIFIED) 1.0 / non-diff(UNCHANGED) 0.25 → diff 가 실질 4배.
# action 샘플은 preprocess 가 images 개수로 판별해 uniform 1.0 을 준다.
W_ADDED = 1.0
W_MODIFIED = 1.0
W_UNCHANGED = 0.25
METRIC_VERSION = "v2"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

TRAIN_NAME = "implicit-world-modeling_stage1_train.jsonl"


def run_mirror(data_root: Path) -> None:
    cmd = [
        sys.executable,
        str(SCRIPTS / "mirror_experiment.py"),
        "--experiment",
        "exp05",
        "--data-root",
        str(data_root),
    ]
    print(f"[1/2] mirror: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def build_weights(
    train_path: Path, model: str, revision: str | None, on_error: str
) -> dict:
    from preprocess_dataset_v2 import preprocess  # noqa: PLC0415 (sys.path 주입 후)

    # 입력이 이미 가중치를 가진 산출물이면 중복 부여를 막는다.
    with train_path.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    if "token_weights" in first:
        raise SystemExit(
            f"[ERROR] {train_path.name} 에 이미 token_weights 가 있다 (가중 산출물). "
            "mirror 로 원본을 재생성한 뒤 실행하라 (--skip-mirror 를 빼면 된다)."
        )

    weighted = train_path.with_name(train_path.name + ".weighted")
    print(f"\n[2/2] diff-loss 가중치 부여 → {train_path.name}\n")
    meta = preprocess(
        input_jsonl=str(train_path),
        output_jsonl=str(weighted),
        model_name=model,
        w_added=W_ADDED,
        w_modified=W_MODIFIED,
        w_unchanged=W_UNCHANGED,
        metric_version=METRIC_VERSION,
        revision=revision,
        on_error=on_error,
    )
    # 가중 산출물을 train 정본 자리에 원자 교체 (sidecar 도 함께)
    os.replace(weighted, train_path)
    os.replace(str(weighted) + ".meta.json", str(train_path) + ".meta.json")
    return meta


def verify(train_path: Path) -> int:
    """계약 검증: state 는 weight ⊆ {0.25, 1.0}, action 은 정확히 {1.0}."""
    n = n_state = n_action = 0
    bad: list[str] = []
    state_tok = state_low = 0
    with train_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            w = set(r["token_weights"])
            n += 1
            if len(r["images"]) == 2:  # action_pred
                n_action += 1
                if w and w != {1.0}:
                    bad.append(f"  line {i}: action 인데 weights={sorted(w)}")
            else:  # state_pred
                n_state += 1
                if not w <= {W_UNCHANGED, W_ADDED, W_MODIFIED}:
                    bad.append(f"  line {i}: state weights 이탈 {sorted(w)}")
                state_tok += len(r["token_weights"])
                state_low += sum(1 for x in r["token_weights"] if x == W_UNCHANGED)

    print("\n── 검증 ──────────────────────────────────────────────")
    print(f"  총 {n} 행 = state {n_state} + action {n_action}")
    if state_tok:
        print(f"  state 출력 토큰 중 {W_UNCHANGED}배 감쇠 비율: {state_low / state_tok:.1%}")
    if bad:
        print(f"  [FAIL] 계약 위반 {len(bad)}건")
        for b in bad[:10]:
            print(b)
        return 1
    print("  [OK] weight 계약 만족")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AC_EXP05 학습 데이터 빌드 (mirror + diff loss v2)")
    p.add_argument(
        "--data-root",
        type=Path,
        default=SCRIPTS.parent / "data",
        help="data 루트 (기본: repo/data)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"tokenizer (기본: {DEFAULT_MODEL})")
    p.add_argument("--revision", default=None, help="tokenizer commit SHA 고정")
    p.add_argument("--skip-mirror", action="store_true", help="mirror 를 건너뛴다")
    p.add_argument(
        "--on-error",
        choices=["fail", "uniform", "skip"],
        default="fail",
        help="diff/weight 실패 처리 (기본 fail = fail-closed)",
    )
    args = p.parse_args(argv)

    if not args.skip_mirror:
        run_mirror(args.data_root)

    train_path = args.data_root / "AndroidControl_EXP05" / TRAIN_NAME
    if not train_path.is_file():
        print(f"[ERROR] 없음: {train_path}", file=sys.stderr)
        return 1

    meta = build_weights(train_path, args.model, args.revision, args.on_error)
    rc = verify(train_path)

    fallback = meta["counts"]["diff_fail"] + meta["counts"]["weight_fail"]
    if fallback:
        print(f"\n[주의] uniform fallback {fallback}건 — sidecar 참조")
    print(f"\nDone. {train_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
