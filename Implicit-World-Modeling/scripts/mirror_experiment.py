#!/usr/bin/env python3
"""EXP01 ratio73 멤버십을 EXP03/EXP04/EXP05 의 좌표 표현으로 미러링한다 (파라미터화 통합).

세 실험(EXP03/04/05)은 EXP01 과 **동일한 (episode, step) 전이** 를 서로 다른 좌표
표현으로 담은 원천 pool 을 갖는다. 본 스크립트는 ``--experiment`` 로 변형을 선택해,
EXP01 ratio73 산출 파일들의 멤버십을 그대로 따라가며 대응하는 좌표 레코드를 동일
순서로 골라 출력한다 (구 ``mirror_exp03.py`` / ``mirror_exp04.py`` / ``mirror_exp05.py``
통합본; 성공 실행의 출력 데이터는 셋과 byte-identical).

- EXP03: 0–1000 정규화 좌표 (point). stage1 7 + stage2 3 = 10 파일.
- EXP04: EXP03 좌표 pool 의 stage1 프롬프트 업그레이드본
  (``scroll(direction, point)`` → ``swipe(start, end)`` 등). stage1 7 파일 (stage2 보류).
- EXP05: 픽셀 좌표 + 해상도 정렬 (Qwen2.5-VL 전용). stage1 7 파일 (stage2 보류).

매칭·출력 규칙 (세 실험 공통)
----
- 매칭 키 = ``images[0]`` 의 ``(episode, step)`` (int() 가 zero-pad 정규화 겸함;
  split_data.py 의 ``EPISODE_RE`` + ``_norm_ep`` 와 동일 규약).
- stage1 task 종류는 이미지 개수로 판별: 2개 = action_pred, 1개 = state_pred.
  stage2 는 stage2 pool 로 매칭 (EXP03 만).
- 이미지 경로: 매칭된 EXP01 레코드의 ``images`` 를 그대로 채택 (AndroidControl/images/...).
- 본문(messages): 각 실험 원천 레코드 (좌표 표현 유지).
- 매칭되는 EXP01 행만 **순서보존 1:1** 로 출력하고, 원천 pool 에 키가 없는 EXP01 행은
  제외한다 (drop). 즉 전 구간 1:1 이 아니라 matched key 에 한한 순서보존 1:1 이다.

EXP05 참고
----
EXP05 의 좌표 공간(840×1876 절대 픽셀)과 이미지 예산(max_pixels=1605632, min_pixels=3136)
은 **upstream 원천 생성 시 확정된 불변식**이며 본 스크립트의 CLI 옵션이 아니다. 학습 시
이미지 프로세서의 ``max_pixels`` 와 반드시 일치해야 한다 (불일치 시 좌표-이미지 grounding
붕괴). EXP03/04 의 0–1000 정규화 좌표와는 다른 규약이라 서로 바꿔 쓸 수 없다.

Usage
-----
  python scripts/mirror_experiment.py --experiment exp03
  python scripts/mirror_experiment.py --experiment exp05 --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_data import EPISODE_RE, load_jsonl, write_jsonl  # scripts/ 코드 재사용

STEP_RE = re.compile(r"step_(\d+)")

Kind = Literal["stage1", "stage2"]

# (EXP01 입력 파일, 출력 파일, kind). 세 실험 공통.
# stage1 train 은 ratio73(train_7_3) 만 미러 → EXP02 스타일 단일 train 명으로 출력.
# 나머지 test / stage2 는 EXP01 과 동일 파일명 (EXP02 와도 byte 구조 동일).
STAGE1_JOBS: list[tuple[str, str, Kind]] = [
    (
        "implicit-world-modeling_stage1_train_7_3.jsonl",
        "implicit-world-modeling_stage1_train.jsonl",
        "stage1",
    ),
    (
        "implicit-world-modeling_stage1_test_id_action.jsonl",
        "implicit-world-modeling_stage1_test_id_action.jsonl",
        "stage1",
    ),
    (
        "implicit-world-modeling_stage1_test_id_state.jsonl",
        "implicit-world-modeling_stage1_test_id_state.jsonl",
        "stage1",
    ),
    (
        "implicit-world-modeling_stage1_test_id_state_without_open_app.jsonl",
        "implicit-world-modeling_stage1_test_id_state_without_open_app.jsonl",
        "stage1",
    ),
    (
        "implicit-world-modeling_stage1_test_ood_action.jsonl",
        "implicit-world-modeling_stage1_test_ood_action.jsonl",
        "stage1",
    ),
    (
        "implicit-world-modeling_stage1_test_ood_state.jsonl",
        "implicit-world-modeling_stage1_test_ood_state.jsonl",
        "stage1",
    ),
    (
        "implicit-world-modeling_stage1_test_ood_state_without_open_app.jsonl",
        "implicit-world-modeling_stage1_test_ood_state_without_open_app.jsonl",
        "stage1",
    ),
]

# stage2 3-job. EXP03 에만 append (EXP04/05 는 stage2 보류).
STAGE2_JOBS: list[tuple[str, str, Kind]] = [
    (
        "implicit-world-modeling_stage2_train.jsonl",
        "implicit-world-modeling_stage2_train.jsonl",
        "stage2",
    ),
    (
        "implicit-world-modeling_stage2_test_id.jsonl",
        "implicit-world-modeling_stage2_test_id.jsonl",
        "stage2",
    ),
    (
        "implicit-world-modeling_stage2_test_ood.jsonl",
        "implicit-world-modeling_stage2_test_ood.jsonl",
        "stage2",
    ),
]


@dataclass(frozen=True)
class VariantConfig:
    """실험별 차이만 파라미터화. (공유 로직은 함수로 고정.)"""

    experiment: str
    action_source: str  # data/AndroidControl/ 아래 action_pred 원천 파일명
    state_source: str  # data/AndroidControl/ 아래 state_pred 원천 파일명
    out_subdir: str  # data/ 아래 출력 서브디렉토리
    has_stage2: bool  # True 면 STAGE2_JOBS + stage2 index 사용 (EXP03)
    stage2_source: str | None = None  # has_stage2=True 일 때만 사용
    missing_source_hint: str | None = None  # 원천 누락 시 stderr 안내 (EXP05)

    def jobs(self) -> list[tuple[str, str, Kind]]:
        return STAGE1_JOBS + (STAGE2_JOBS if self.has_stage2 else [])


VARIANTS: dict[str, VariantConfig] = {
    "exp03": VariantConfig(
        experiment="exp03",
        action_source="implicit-world-modeling_stage1_action_xy.jsonl",
        state_source="implicit-world-modeling_stage1_state_xy.jsonl",
        stage2_source="implicit-world-modeling_stage2_xy.jsonl",
        out_subdir="AndroidControl_EXP03",
        has_stage2=True,
    ),
    "exp04": VariantConfig(
        experiment="exp04",
        action_source="implicit-world-modeling_stage1_action_xy_prompt-enhanced.jsonl",
        state_source="implicit-world-modeling_stage1_state_xy_prompt-enhanced.jsonl",
        out_subdir="AndroidControl_EXP04",
        has_stage2=False,
    ),
    "exp05": VariantConfig(
        experiment="exp05",
        action_source="implicit-world-modeling_stage1_action_xy_pixel-aligned.jsonl",
        state_source="implicit-world-modeling_stage1_state_xy_pixel-aligned.jsonl",
        out_subdir="AndroidControl_EXP05",
        has_stage2=False,
        missing_source_hint=(
            "Google Drive '0710_버젼' 폴더에서 받아 위 이름으로 배치하세요."
        ),
    ),
}


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
    kind: Kind,
    action_idx: dict[tuple[int, int], dict],
    state_idx: dict[tuple[int, int], dict],
    stage2_idx: dict[tuple[int, int], dict] | None = None,
) -> tuple[list[dict], int]:
    out: list[dict] = []
    dropped = 0
    for rec in rows:
        imgs = rec["images"]
        if kind == "stage1":
            idx = action_idx if len(imgs) == 2 else state_idx  # 2개=action, 1개=state
        else:  # stage2
            idx = stage2_idx
        src = idx.get(key_of(imgs))
        if src is None:  # 원천에 없는 데이터 → 제외
            dropped += 1
            continue
        out.append({**src, "images": imgs})  # 본문=실험 좌표, 경로=EXP01
    return out, dropped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXP01 ratio73 멤버십을 EXP03/04/05 좌표 표현으로 미러링",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(VARIANTS),
        help="미러링할 변형 (필수, 기본값 없음)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="data 루트 (기본: repo/data). AndroidControl/ 와 AndroidControl_EXP01/ 포함",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = VARIANTS[args.experiment]

    base: Path = args.data_root
    src = base / "AndroidControl"
    exp01 = base / "AndroidControl_EXP01"
    out_dir = base / cfg.out_subdir
    jobs = cfg.jobs()

    action_src = src / cfg.action_source
    state_src = src / cfg.state_source
    stage2_src = src / cfg.stage2_source if cfg.has_stage2 else None

    # ── resolved variant / source / 목적지 / job 목록 출력 ──────────────────
    print(f"[{cfg.experiment}] mirror EXP01 ratio73 → {out_dir}")
    print(f"  source root : {src}")
    print(f"  exp01 root  : {exp01}")
    print(f"  action src  : {cfg.action_source}")
    print(f"  state  src  : {cfg.state_source}")
    if cfg.has_stage2:
        print(f"  stage2 src  : {cfg.stage2_source}")
    print(f"  jobs ({len(jobs)}):")
    for in_name, out_name, kind in jobs:
        print(f"    [{kind}] {in_name} → {out_name}")

    # ── 필요한 입력 일괄 preflight (원천 + EXP01 입력) ─────────────────────
    src_paths = [action_src, state_src] + ([stage2_src] if stage2_src else [])
    input_paths = [exp01 / in_name for in_name, _out, _kind in jobs]
    missing_src = [p for p in src_paths if not p.exists()]
    missing_input = [p for p in input_paths if not p.exists()]
    if missing_src or missing_input:
        print("[ERROR] 필요한 입력 없음:", file=sys.stderr)
        for p in missing_src + missing_input:
            print(f"  - {p}", file=sys.stderr)
        if missing_src and cfg.missing_source_hint:  # 원천 누락 안내 (EXP05)
            print(f"  힌트: {cfg.missing_source_hint}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    action_idx = build_index(action_src)
    state_idx = build_index(state_src)
    stage2_idx = build_index(stage2_src) if stage2_src else None

    idx_line = f"index: action={len(action_idx)} state={len(state_idx)}"
    if stage2_idx is not None:
        idx_line += f" stage2={len(stage2_idx)}"
    print(idx_line)
    print(f"{'output file':58s} {'in':>6} {'out':>6} {'drop':>5}")

    total_out = 0
    for in_name, out_name, kind in jobs:
        rows = load_jsonl(exp01 / in_name)
        out, dropped = mirror(
            rows,
            kind=kind,
            action_idx=action_idx,
            state_idx=state_idx,
            stage2_idx=stage2_idx,
        )
        write_jsonl(out, out_dir / out_name)
        total_out += len(out)
        print(f"{out_name:58s} {len(rows):6d} {len(out):6d} {dropped:5d}")

    print(f"\nDone. {len(jobs)} files, {total_out} rows → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
