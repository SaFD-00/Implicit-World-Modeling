"""Differential test: scripts/mirror_experiment.py 가 구 mirror_exp03/04/05.py 와
동작 보존(출력 데이터 byte-identical)임을 증명한다.

3층 구조
----
1. Config 동등성: mirror_experiment 의 각 variant 가 만드는 (job in/out/kind, source
   파일명, out_subdir) 가 원본 상수(아래 하드코딩; 원본에서 verbatim 복사)와 정확히
   일치. 원본 import 를 요구하지 않으므로 원본 삭제 후에도 항상 실행된다.
2. 로직 동등성(합성 데이터): 원본 mirror_exp03/04/05.mirror 와 새 mirror 를 동일
   합성 rows+idx 에 돌려 출력 레코드·drop 수가 동일. 모든 분기 커버 —
   stage1 action(2 imgs)·stage1 state(1 img)·stage2·매칭 실패(drop).
3. End-to-end byte-identity(소형 합성 data 트리): 원본 main()(data-root 를 tmp 로)과
   mirror_experiment 의 exp03/04/05 실행을 각각 별도 out-dir 에 돌려 생성된 jsonl
   파일 집합의 바이트가 동일함을 assert.

Layer 2·3 은 원본 모듈을 import 하므로 원본이 삭제되면 skip 된다 (skipif). 원본이
존재하는 상태에서의 통과가 byte-identity 증명이며, 삭제 후 실행에서는 Layer 1 만
남고 2·3 은 skipped 로 보고된다.

Run:
    pytest tests/test_mirror_experiment.py -v
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

mirror_experiment = importlib.import_module("mirror_experiment")

# 원본 3모듈은 통합 후 삭제 대상 → import 실패해도 Layer 1 은 살아있어야 한다.
_ORIG_MODS: dict[str, object] = {}
try:
    _ORIG_MODS = {
        "exp03": importlib.import_module("mirror_exp03"),
        "exp04": importlib.import_module("mirror_exp04"),
        "exp05": importlib.import_module("mirror_exp05"),
    }
    ORIGINALS_PRESENT = True
except ImportError:
    ORIGINALS_PRESENT = False

needs_originals = pytest.mark.skipif(
    not ORIGINALS_PRESENT,
    reason="원본 mirror_exp03/04/05.py 삭제됨 — byte-identity 는 삭제 전 통과로 증명됨",
)


# ── 원본에서 verbatim 복사한 기대 상수 (Layer 1 비교 기준) ────────────────────
# 원본 세 파일의 JOBS 튜플 3번째 원소(stage1 bool)를 kind 로 사상: True→"stage1",
# False→"stage2". source 파일명은 원본 main() 의 build_index 인자에서 그대로 복사.

_STAGE1_JOBS = [
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
_STAGE2_JOBS = [
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

# variant → (기대 jobs, action_source, state_source, stage2_source, out_subdir)
_EXPECTED = {
    "exp03": {
        "jobs": _STAGE1_JOBS + _STAGE2_JOBS,
        "action_source": "implicit-world-modeling_stage1_action_xy.jsonl",
        "state_source": "implicit-world-modeling_stage1_state_xy.jsonl",
        "stage2_source": "implicit-world-modeling_stage2_xy.jsonl",
        "out_subdir": "AndroidControl_EXP03",
    },
    "exp04": {
        "jobs": _STAGE1_JOBS,
        "action_source": "implicit-world-modeling_stage1_action_xy_prompt-enhanced.jsonl",
        "state_source": "implicit-world-modeling_stage1_state_xy_prompt-enhanced.jsonl",
        "stage2_source": None,
        "out_subdir": "AndroidControl_EXP04",
    },
    "exp05": {
        "jobs": _STAGE1_JOBS,
        "action_source": "implicit-world-modeling_stage1_action_xy_pixel-aligned.jsonl",
        "state_source": "implicit-world-modeling_stage1_state_xy_pixel-aligned.jsonl",
        "stage2_source": None,
        "out_subdir": "AndroidControl_EXP05",
    },
}


# ── Layer 1: Config 동등성 (원본 import 불요, 항상 실행) ──────────────────────
@pytest.mark.parametrize("exp", ["exp03", "exp04", "exp05"])
def test_config_matches_original_constants(exp: str) -> None:
    cfg = mirror_experiment.VARIANTS[exp]
    want = _EXPECTED[exp]

    assert cfg.jobs() == want["jobs"], f"{exp} jobs (in/out/kind·순서) 불일치"
    assert cfg.action_source == want["action_source"]
    assert cfg.state_source == want["state_source"]
    assert cfg.stage2_source == want["stage2_source"]
    assert cfg.out_subdir == want["out_subdir"]
    assert cfg.has_stage2 is (want["stage2_source"] is not None)


def test_job_counts() -> None:
    # 원본 exp03 JOBS=10 (stage1 7 + stage2 3), exp04/05 JOBS=7.
    assert len(mirror_experiment.VARIANTS["exp03"].jobs()) == 10
    assert len(mirror_experiment.VARIANTS["exp04"].jobs()) == 7
    assert len(mirror_experiment.VARIANTS["exp05"].jobs()) == 7
    assert len(mirror_experiment.STAGE1_JOBS) == 7
    assert len(mirror_experiment.STAGE2_JOBS) == 3


@needs_originals
@pytest.mark.parametrize("exp", ["exp03", "exp04", "exp05"])
def test_config_matches_imported_original_jobs(exp: str) -> None:
    """원본 모듈 JOBS 를 직접 import 해 (in/out/kind) 사상 일치까지 확인."""
    orig = _ORIG_MODS[exp]
    orig_jobs = [(i, o, "stage1" if s1 else "stage2") for i, o, s1 in orig.JOBS]
    assert mirror_experiment.VARIANTS[exp].jobs() == orig_jobs


# ── 합성 데이터 헬퍼 ─────────────────────────────────────────────────────────
def _img(prefix: str, ep: int, step: int, n: int) -> list[str]:
    """episode/step 이 추출 가능한 합성 image 경로 n개."""
    return [
        f"{prefix}/images/episode_{ep:05d}/step_{step}/img_{k}.png" for k in range(n)
    ]


def _make_indices():
    """action(2img)·state(1img)·stage2 원천 index 를 합성 생성."""
    action_rows = [
        {"messages": f"action-{ep}", "images": _img("SRC", ep, 0, 2), "kind": "A"}
        for ep in (1, 2)
    ]
    state_rows = [
        {"messages": f"state-{ep}", "images": _img("SRC", ep, 0, 1), "kind": "S"}
        for ep in (3, 4)
    ]
    stage2_rows = [
        {"messages": f"stage2-{ep}", "images": _img("SRC", ep, 0, 1), "kind": "T"}
        for ep in (5, 6)
    ]
    return action_rows, state_rows, stage2_rows


def _stage1_exp01_rows() -> list[dict]:
    """stage1 EXP01 rows: action match·state match·둘 다 drop 커버. 이미지 경로는
    원천과 다르게(EXP01 prefix) 주어 '경로=EXP01, 본문=원천' 병합을 검증한다."""
    return [
        {"images": _img("EXP01", 1, 0, 2)},  # → action match (ep1)
        {"images": _img("EXP01", 3, 0, 1)},  # → state match (ep3)
        {"images": _img("EXP01", 2, 0, 2)},  # → action match (ep2)
        {"images": _img("EXP01", 9, 9, 2)},  # → drop (2img, 키 없음)
        {"images": _img("EXP01", 8, 8, 1)},  # → drop (1img, 키 없음)
    ]


def _stage2_exp01_rows() -> list[dict]:
    return [
        {"images": _img("EXP01", 5, 0, 1)},  # → stage2 match (ep5)
        {"images": _img("EXP01", 7, 7, 1)},  # → drop
    ]


# ── Layer 2: 로직 동등성 (합성 데이터) ───────────────────────────────────────
@needs_originals
def test_mirror_logic_exp03_stage1_and_stage2() -> None:
    orig = _ORIG_MODS["exp03"]
    action_rows, state_rows, stage2_rows = _make_indices()
    a_idx = {orig.key_of(r["images"]): r for r in action_rows}
    s_idx = {orig.key_of(r["images"]): r for r in state_rows}
    t_idx = {orig.key_of(r["images"]): r for r in stage2_rows}

    # stage1 분기
    rows1 = _stage1_exp01_rows()
    o1, od1 = orig.mirror(
        rows1, stage1=True, action_idx=a_idx, state_idx=s_idx, stage2_idx=t_idx
    )
    n1, nd1 = mirror_experiment.mirror(
        rows1, kind="stage1", action_idx=a_idx, state_idx=s_idx, stage2_idx=t_idx
    )
    assert (o1, od1) == (n1, nd1)
    assert od1 == 2 and len(o1) == 3  # 2 drop, 3 match (action·state·action)

    # stage2 분기
    rows2 = _stage2_exp01_rows()
    o2, od2 = orig.mirror(
        rows2, stage1=False, action_idx=a_idx, state_idx=s_idx, stage2_idx=t_idx
    )
    n2, nd2 = mirror_experiment.mirror(
        rows2, kind="stage2", action_idx=a_idx, state_idx=s_idx, stage2_idx=t_idx
    )
    assert (o2, od2) == (n2, nd2)
    assert od2 == 1 and len(o2) == 1


@needs_originals
@pytest.mark.parametrize("exp", ["exp04", "exp05"])
def test_mirror_logic_exp04_05_stage1(exp: str) -> None:
    orig = _ORIG_MODS[exp]
    action_rows, state_rows, _ = _make_indices()
    a_idx = {orig.key_of(r["images"]): r for r in action_rows}
    s_idx = {orig.key_of(r["images"]): r for r in state_rows}

    rows1 = _stage1_exp01_rows()
    o, od = orig.mirror(rows1, action_idx=a_idx, state_idx=s_idx)
    n, nd = mirror_experiment.mirror(
        rows1, kind="stage1", action_idx=a_idx, state_idx=s_idx
    )
    assert (o, od) == (n, nd)
    assert od == 2 and len(o) == 3
    # 병합 검증: 본문=원천, 경로=EXP01
    assert o[0]["images"] == rows1[0]["images"]
    assert o[0]["messages"] == action_rows[0]["messages"]


# ── Layer 3: End-to-end byte-identity (소형 합성 data 트리) ──────────────────
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mirror_experiment.write_jsonl(rows, path)


def _build_data_tree(data_root: Path, exp: str) -> None:
    """축소된 data/AndroidControl/ + data/AndroidControl_EXP01/ 합성 트리 생성."""
    cfg = mirror_experiment.VARIANTS[exp]
    src = data_root / "AndroidControl"
    exp01 = data_root / "AndroidControl_EXP01"

    action_rows, state_rows, stage2_rows = _make_indices()
    _write_jsonl(src / cfg.action_source, action_rows)
    _write_jsonl(src / cfg.state_source, state_rows)
    if cfg.has_stage2:
        _write_jsonl(src / cfg.stage2_source, stage2_rows)

    # EXP01 입력: 모든 stage1 job 은 동일 stage1 rows, stage2 job 은 stage2 rows.
    for in_name, _out, kind in cfg.jobs():
        rows = _stage2_exp01_rows() if kind == "stage2" else _stage1_exp01_rows()
        _write_jsonl(exp01 / in_name, rows)


@needs_originals
@pytest.mark.parametrize("exp", ["exp03", "exp04", "exp05"])
def test_end_to_end_byte_identity(exp: str, tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path.resolve()
    orig_root = base_dir / "orig"
    new_root = base_dir / "new"
    orig_data = orig_root / "data"
    new_data = new_root / "data"
    _build_data_tree(orig_data, exp)
    _build_data_tree(new_data, exp)

    orig = _ORIG_MODS[exp]
    # 원본 main() 의 base = Path(__file__).resolve().parent.parent / "data" 리다이렉트.
    fake_file = orig_root / "scripts" / f"mirror_{exp}.py"
    monkeypatch.setattr(orig, "__file__", str(fake_file))
    assert orig.main() == 0

    assert (
        mirror_experiment.main(["--experiment", exp, "--data-root", str(new_data)]) == 0
    )

    out_subdir = mirror_experiment.VARIANTS[exp].out_subdir
    orig_out = orig_data / out_subdir
    new_out = new_data / out_subdir

    orig_files = sorted(p.name for p in orig_out.glob("*.jsonl"))
    new_files = sorted(p.name for p in new_out.glob("*.jsonl"))
    assert orig_files == new_files, "출력 파일 집합 불일치"
    assert orig_files, "출력 파일이 하나도 없음 (합성 트리 오류)"

    for name in orig_files:
        ob = (orig_out / name).read_bytes()
        nb = (new_out / name).read_bytes()
        assert ob == nb, f"byte mismatch in {name} ({exp})"
