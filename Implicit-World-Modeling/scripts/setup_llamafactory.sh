#!/usr/bin/env bash
# setup_llamafactory.sh — LlamaFactory 워킹트리 부트스트랩 (clone → pin → patch → install → verify)
#
# 노트북 Cell 3(pin 없는 clone) + Cell 7(anchor 문자열 치환 패치) 의식을 대체한다.
# 멱등: 이미 pin + 패치 적용 상태면 완전한 no-op.
#
# 사용법:
#   bash scripts/setup_llamafactory.sh              # clone → pin → patch
#   bash scripts/setup_llamafactory.sh --install    # + pip editable 설치 (활성 conda env 필요)
#   bash scripts/setup_llamafactory.sh --verify     # + 상태 표 출력 (실패 시 비0 exit)
#
# 환경변수:
#   LF_DIR      LlamaFactory 워킹트리 경로 (기본: <프로젝트루트>/LlamaFactory)
#   LF_GIT_URL  clone 원격 (기본: hiyouga/LlamaFactory)
#   PYTHON      verify/install 에 쓸 python (기본: python)
#
# ★ 안전 규칙: 이 스크립트는 LF 워킹트리에 대해 git reset / git checkout -- <file> /
#   git clean / 파일 삭제를 절대 하지 않는다. 상태가 기대와 다르면 진단만 출력하고 종료한다.
#   (LF 는 gitignore 된 서드파티 트리 = 사용자의 유일본일 수 있다.)

set -euo pipefail

# --- 상수 -------------------------------------------------------------------
LF_PIN="99464b3d034fd19fa73486f05e3b64b963e1b423"
DEFAULT_LF_GIT_URL="https://github.com/hiyouga/LlamaFactory.git"
REQUIRED_ENV="implicit-world-modeling"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="$PROJECT_ROOT/patches/llamafactory"
DATASET_INFO="$PROJECT_ROOT/configs/lf_dataset/dataset_info.json"

LF_DIR="${LF_DIR:-$PROJECT_ROOT/LlamaFactory}"
LF_GIT_URL="${LF_GIT_URL:-$DEFAULT_LF_GIT_URL}"
PYTHON="${PYTHON:-python}"
[[ "$LF_DIR" == /* ]] || LF_DIR="$PWD/$LF_DIR"

# LF 워킹트리의 "기지(旣知) dirty" — 실패 사유로 삼지 않는다.
#   data/dataset_info.json : IWM-* 데이터셋 엔트리 머지 (별도 단위의 승인 게이트 사항)
#   untracked (data/ 심링크, examples/custom/) : status --untracked-files=no 로 애초에 무시
KNOWN_DIRTY_RE='^data/dataset_info\.json$'

# --- 출력 헬퍼 ---------------------------------------------------------------
say()  { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }
err()  { printf '\n[FAIL] %s\n' "$1" >&2; shift; local l; for l in "$@"; do printf '       %s\n' "$l" >&2; done; }
die()  { err "$@"; exit 1; }

TMPWORK=""
cleanup() { [[ -n "$TMPWORK" && -d "$TMPWORK" ]] && rm -rf "$TMPWORK"; return 0; }
trap cleanup EXIT

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# --- 인자 --------------------------------------------------------------------
DO_INSTALL=0
DO_VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --install) DO_INSTALL=1 ;;
    --verify)  DO_VERIFY=1 ;;
    -h|--help) usage 0 ;;
    *) err "알 수 없는 인자: $arg" "사용 가능: --install, --verify, --help"; exit 2 ;;
  esac
done

# --- 패치 목록 (파일명 하드코딩 금지: *.patch 정렬 순회) ------------------------
PATCHES=()
if [[ -d "$PATCH_DIR" ]]; then
  while IFS= read -r p; do PATCHES+=("$p"); done \
    < <(find "$PATCH_DIR" -maxdepth 1 -type f -name '*.patch' | LC_ALL=C sort)
fi
NPATCH=${#PATCHES[@]}
if (( NPATCH == 0 )); then
  die "패치를 찾지 못했다: $PATCH_DIR/*.patch" \
      "이 저장소의 patches/llamafactory/ 가 비어 있거나 경로가 잘못됐다."
fi

# 패치가 건드리는 파일들 (a/<path> → <path>), 중복 제거
TOUCHED=()
_seen=" "
for p in "${PATCHES[@]}"; do
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ "$_seen" == *" $f "* ]] && continue
    _seen+="$f "
    TOUCHED+=("$f")
  done < <(awk '/^diff --git /{ t=$3; sub(/^a\//, "", t); print t }' "$p")
done
if (( ${#TOUCHED[@]} == 0 )); then
  die "패치에서 대상 파일을 파싱하지 못했다 ($PATCH_DIR/*.patch)" \
      "'diff --git a/... b/...' 헤더가 있는 git 형식 패치인지 확인하라."
fi

say "LlamaFactory bootstrap"
say "  LF_DIR     : $LF_DIR"
say "  LF_GIT_URL : $LF_GIT_URL"
say "  pin        : $LF_PIN"
say "  patches    : ${NPATCH}개 (${TOUCHED[*]} 대상)"

# --- 1) clone ----------------------------------------------------------------
step "체크아웃 확인"
if [[ ! -e "$LF_DIR" ]]; then
  say "[clone] $LF_GIT_URL → $LF_DIR"
  mkdir -p "$(dirname "$LF_DIR")"
  git clone "$LF_GIT_URL" "$LF_DIR"
  git -C "$LF_DIR" checkout --quiet --detach "$LF_PIN"
  say "[pin]   HEAD → $LF_PIN (detached)"
elif [[ ! -d "$LF_DIR/.git" ]]; then
  die "$LF_DIR 가 존재하지만 git 저장소가 아니다." \
      "직접 확인 후 옮기거나 지우고 다시 실행하라 (이 스크립트는 삭제하지 않는다):" \
      "  ls -la '$LF_DIR'"
else
  say "[keep]  기존 체크아웃 사용 (건드리지 않음)"
fi

# --- 2) pin 확인 --------------------------------------------------------------
HEAD_SHA="$(git -C "$LF_DIR" rev-parse HEAD)"
if [[ "$HEAD_SHA" != "$LF_PIN" ]]; then
  DIRTY="$(git -C "$LF_DIR" status --porcelain --untracked-files=no)"
  if [[ -n "$DIRTY" ]]; then
    err "LF HEAD 가 pin 이 아닌데 워킹트리에 로컬 변경이 있다 → 아무것도 건드리지 않고 중단한다." \
        "HEAD : $HEAD_SHA" \
        "pin  : $LF_PIN" \
        "" \
        "로컬 변경 (이 내용이 유일본일 수 있다):"
    printf '%s\n' "$DIRTY" | sed 's/^/         /' >&2
    err "다음을 직접 확인한 뒤 수동으로 정리하라 (스크립트는 실행하지 않는다):" \
        "  git -C '$LF_DIR' diff                 # 무엇이 바뀌었는지 확인" \
        "  git -C '$LF_DIR' stash push -m lf-wip # 보존하려면" \
        "  git -C '$LF_DIR' checkout --detach $LF_PIN"
    exit 1
  fi
  say "[pin]   HEAD($HEAD_SHA) != pin, 워킹트리 clean → pin 으로 checkout"
  if ! git -C "$LF_DIR" rev-parse --verify --quiet "${LF_PIN}^{commit}" >/dev/null; then
    say "[fetch] pin 커밋이 로컬에 없다 → origin 에서 가져온다"
    git -C "$LF_DIR" fetch origin "$LF_PIN" \
      || die "pin 커밋을 가져오지 못했다: $LF_PIN" \
             "원격이 올바른지 확인하라: git -C '$LF_DIR' remote -v"
  fi
  git -C "$LF_DIR" checkout --quiet --detach "$LF_PIN"
fi
say "[ok]    HEAD == pin ($LF_PIN)"

# 패치 대상도 아니고 기지 dirty 도 아닌 변경 → 경고만 (패치 적용과 무관하므로 중단하지 않음)
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  [[ "$_seen" == *" $f "* ]] && continue
  [[ "$f" =~ $KNOWN_DIRTY_RE ]] && continue
  warn "패치와 무관한 로컬 변경: $f (그대로 둔다)"
done < <(git -C "$LF_DIR" diff --name-only HEAD || true)

# --- 3) 기지 상태 게이트 ------------------------------------------------------
# pin 기준 pristine(state0) 에 패치를 1..N 누적 적용한 기대 상태를 임시 디렉토리에 만들고,
# 현재 LF 워킹트리가 그중 어느 state 와 byte-exact 로 일치하는지 찾는다.
#   match_k == N  → 전부 적용됨 (no-op)
#   0 <= k < N    → k개까지 적용됨 → 나머지만 적용
#   match_k == -1 → 우리 패치로 설명되지 않는 상태(오염) → 진단 후 중단
# (git apply --reverse --check 만으로는 패치 hunk 바깥의 오염을 탐지하지 못한다.)
step "워킹트리 상태 판정 (기지 상태 게이트)"
TMPWORK="$(mktemp -d)"
mkdir -p "$TMPWORK/state0"
for f in "${TOUCHED[@]}"; do
  if git -C "$LF_DIR" cat-file -e "${LF_PIN}:${f}" 2>/dev/null; then
    mkdir -p "$TMPWORK/state0/$(dirname "$f")"
    git -C "$LF_DIR" show "${LF_PIN}:${f}" > "$TMPWORK/state0/$f"
  fi   # pin 에 없는 파일 = 패치가 새로 만드는 파일 → state0 에 없음이 정답
done
# 임시 디렉토리를 git 저장소로 만들어 `git apply` 의 경로 기준(toplevel)을 고정한다.
git init -q -b main "$TMPWORK/state0"

for (( i = 1; i <= NPATCH; i++ )); do
  cp -a "$TMPWORK/state$((i - 1))" "$TMPWORK/state$i"
  if ! git -C "$TMPWORK/state$i" apply "${PATCHES[i-1]}" 2>"$TMPWORK/apply_err"; then
    err "패치가 pristine pin 에 적용되지 않는다: ${PATCHES[i-1]}" \
        "pin($LF_PIN)과 패치 버전이 어긋났다. patches/llamafactory/README.md 의 pin 을 확인하라."
    sed 's/^/       /' "$TMPWORK/apply_err" >&2
    exit 1
  fi
done

state_matches() {  # $1 = k
  local k="$1" f a b
  for f in "${TOUCHED[@]}"; do
    a="$LF_DIR/$f"; b="$TMPWORK/state$k/$f"
    if [[ -e "$a" && -e "$b" ]]; then
      cmp -s "$a" "$b" || return 1
    elif [[ -e "$a" || -e "$b" ]]; then
      return 1
    fi
  done
  return 0
}

MATCH_K=-1
for (( k = NPATCH; k >= 0; k-- )); do
  if state_matches "$k"; then MATCH_K=$k; break; fi
done

if (( MATCH_K < 0 )); then
  err "LF 워킹트리가 '$LF_PIN + 우리 패치' 로 설명되지 않는 상태다 → 아무것도 건드리지 않고 중단한다."
  {
    printf '       패치 대상 파일 대조 (pristine=pin, patched=pin+%d개 패치):\n' "$NPATCH"
    for f in "${TOUCHED[@]}"; do
      state="알 수 없는 내용(오염)"
      for (( k = NPATCH; k >= 0; k-- )); do
        if [[ -e "$LF_DIR/$f" && -e "$TMPWORK/state$k/$f" ]] && cmp -s "$LF_DIR/$f" "$TMPWORK/state$k/$f"; then
          if (( k == 0 )); then state="pristine (패치 미적용)"; else state="패치 ${k}개까지 적용됨"; fi
          break
        fi
      done
      printf '         %-52s %s\n' "$f" "$state"
    done
    printf '\n       무엇이 다른지 확인하고 직접 정리하라 (스크립트는 되돌리지 않는다):\n'
    printf "         git -C '%s' diff -- src/\n" "$LF_DIR"
    printf "         git -C '%s' stash push -m lf-wip   # 보존하려면\n" "$LF_DIR"
    printf "         git -C '%s' checkout --detach %s   # 그 뒤 이 스크립트를 다시 실행\n" "$LF_DIR" "$LF_PIN"
  } >&2
  exit 1
fi
say "[ok]    ${MATCH_K}/${NPATCH} 패치가 적용된 기지 상태"

# --- 4) 패치 적용 (멱등) ------------------------------------------------------
step "패치 적용"
for (( i = 1; i <= NPATCH; i++ )); do
  p="${PATCHES[i-1]}"
  name="$(basename "$p")"
  if (( i <= MATCH_K )); then
    say "[skip]  $name — 이미 적용됨"
    continue
  fi
  if git -C "$LF_DIR" apply --check "$p" 2>"$TMPWORK/apply_err"; then
    git -C "$LF_DIR" apply "$p"
    say "[apply] $name"
  else
    err "패치를 적용할 수 없다 (버전/상태 불일치): $name"
    sed 's/^/       /' "$TMPWORK/apply_err" >&2
    err "LF pin 과 워킹트리 상태를 확인하라:" \
        "  git -C '$LF_DIR' rev-parse HEAD   # == $LF_PIN 이어야 한다" \
        "  git -C '$LF_DIR' status --porcelain"
    exit 1
  fi
done

# --- 5) --install ------------------------------------------------------------
if (( DO_INSTALL )); then
  step "pip 설치 (editable)"
  # conda activate 는 시도하지 않는다: 비대화형 셸에서 무동작이라 조용히 엉뚱한 env 에 설치된다.
  # 대신 현재 활성 python 이 올바른 env 인지 검사하고, 아니면 안내 후 중단한다.
  if ! PY_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)' 2>/dev/null)"; then
    die "python 을 찾을 수 없다 ('$PYTHON')." \
        "conda env 를 활성화한 뒤 다시 실행하라:" \
        "  conda activate $REQUIRED_ENV" \
        "  bash scripts/setup_llamafactory.sh --install"
  fi
  if [[ "$(basename "$PY_PREFIX")" != "$REQUIRED_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$REQUIRED_ENV" ]]; then
    die "활성 python env 가 '$REQUIRED_ENV' 가 아니다 (sys.prefix=$PY_PREFIX)." \
        "잘못된 env 에 설치하지 않기 위해 중단한다. 다음을 실행하라:" \
        "  conda activate $REQUIRED_ENV" \
        "  bash scripts/setup_llamafactory.sh --install"
  fi
  say "[env]   $PY_PREFIX"
  ( cd "$PROJECT_ROOT" && "$PYTHON" -m pip install -e ".[llamafactory]" )
  ( cd "$PROJECT_ROOT" && "$PYTHON" -m pip install -e "$LF_DIR" )
  say "[ok]    editable 설치 완료"
fi

# --- 6) --verify -------------------------------------------------------------
if (( DO_VERIFY )); then
  step "검증"
  VERIFY_FAIL=0
  row() {  # $1 = ok|FAIL, $2 = 항목, $3 = 상세
    if [[ "$1" == "ok" ]]; then printf '  [ ok ] %-34s %s\n' "$2" "$3"
    else printf '  [FAIL] %-34s %s\n' "$2" "$3"; VERIFY_FAIL=1; fi
  }

  # 1. HEAD == pin
  HEAD_NOW="$(git -C "$LF_DIR" rev-parse HEAD)"
  if [[ "$HEAD_NOW" == "$LF_PIN" ]]; then
    row ok "LF HEAD == pin" "${LF_PIN:0:12}"
  else
    row FAIL "LF HEAD == pin" "HEAD=${HEAD_NOW:0:12} != pin=${LF_PIN:0:12}"
  fi

  # 2. 모든 패치가 적용된 상태인가 — 위 기지 상태 게이트의 MATCH_K 를 재사용한다.
  # 패치를 하나씩 `git apply --reverse --check` 하면 안 된다: 패치는 스택이라
  # 뒤 패치가 앞 패치의 추가 라인을 수정하므로(0002 가 0001 의 라인을 고친다),
  # 최종 트리에서 앞 패치만 단독 역적용하는 것은 원리적으로 불가능하다.
  if (( MATCH_K == NPATCH )); then
    row ok "패치 적용 상태 (${NPATCH}개)" "$(for p in "${PATCHES[@]}"; do printf '%s ' "$(basename "$p")"; done)"
  else
    row FAIL "패치 적용 상태 (${NPATCH}개)" "${MATCH_K}/${NPATCH} 적용됨 (기지 상태 게이트 기준)"
  fi

  # 3. import 레벨 증명: 패치된 필드가 실제로 설치된 llamafactory 에 존재하는가
  if PY_OUT="$("$PYTHON" -c 'import os, llamafactory
from llamafactory.hparams.finetuning_args import FinetuningArguments
FinetuningArguments(use_diff_token_weighted_loss=True)
print(os.path.dirname(llamafactory.__file__))' 2>"$TMPWORK/py_err")"; then
    row ok "import: use_diff_token_weighted_loss" "$PY_OUT"
  else
    row FAIL "import: use_diff_token_weighted_loss" "$(tail -1 "$TMPWORK/py_err" 2>/dev/null || echo "python 실행 실패 ('$PYTHON')")"
    say "         → conda activate $REQUIRED_ENV 후 재실행, 또는 --install 로 설치하라."
  fi

  # 4. dataset_info.json 존재
  if [[ -f "$DATASET_INFO" ]]; then
    row ok "configs/lf_dataset/dataset_info.json" "존재"
  else
    row FAIL "configs/lf_dataset/dataset_info.json" "없음: $DATASET_INFO"
  fi

  if (( VERIFY_FAIL )); then
    printf '\n[FAIL] 검증 실패 — 위 표의 [FAIL] 항목을 해결하라.\n' >&2
    exit 1
  fi
  say ""
  say "[ok]    검증 통과"
fi

step "완료"
say "LF_DIR=$LF_DIR (pin $LF_PIN, 패치 ${NPATCH}/${NPATCH} 적용)"
