#!/usr/bin/env bash
# remote_launch.sh — configs/remote/run.template.yaml 을 렌더하고 (가능하면) 원격 GPU 클러스터에 제출한다.
#
# 제공자 중립: 이 스크립트에는 특정 플랫폼 이름이 하드코딩돼 있지 않다. 제출 커맨드는 .env 의
#             REMOTE_SUBMIT_CMD 로 주입하고, run spec 스키마는 configs/remote/run.template.yaml 을
#             교체하면 바뀐다. 플랫폼을 갈아타도 이 코드는 그대로다.
#
# UNVALIDATED: 이 스크립트는 제출 CLI 가 없는 머신에서 작성됐다. 검증된 것은 렌더(치환) 경로뿐이며,
#              제출 커맨드의 플래그/스키마는 실제로 실행해 확인하지 않았다.
#
# 사용법:
#   bash scripts/remote_launch.sh --dry-run            # 렌더 결과를 stdout 으로 출력하고 종료 (제출 안 함)
#   bash scripts/remote_launch.sh --dry-run -o out.yaml
#   bash scripts/remote_launch.sh                      # 렌더 후 제출 시도 (REMOTE_SUBMIT_CMD 없으면 렌더만)
#
# 입력 변수 (프로세스 환경 > .env 우선순위) — .env.example 의 REMOTE_* 블록 참고:
#   필수: REMOTE_ORG REMOTE_PROJECT REMOTE_CLUSTER REMOTE_GPU_TYPE REMOTE_GPU_COUNT
#         REMOTE_IMAGE REMOTE_ARTIFACT_URI IWM_DATA_MOUNT IWM_GIT_URL IWM_GIT_REF
#   선택: REMOTE_SUBMIT_CMD  — 제출 커맨드 템플릿. {spec} {org} {project} 가 치환된다.
#                              예) '<cli> run create -f {spec} --organization {org} --project {project}'
#                              비어 있으면 렌더만 하고 제출하지 않는다 (거짓 성공을 주장하지 않기 위해).
#
# 이 스크립트는 .env 값을 화면에 출력하지 않는다 (HF_TOKEN 평문 보호). 단 --dry-run 의 렌더 결과에는
# REMOTE_* / IWM_* 값이 포함된다 (템플릿에 토큰은 들어가지 않는다 — 플랫폼 secret 으로 주입).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$PROJECT_ROOT/configs/remote/run.template.yaml"
ENV_FILE="$PROJECT_ROOT/.env"

REQUIRED_VARS=(
  REMOTE_ORG REMOTE_PROJECT REMOTE_CLUSTER REMOTE_GPU_TYPE REMOTE_GPU_COUNT
  REMOTE_IMAGE REMOTE_ARTIFACT_URI IWM_DATA_MOUNT IWM_GIT_URL IWM_GIT_REF
)
# envsubst 에 넘길 shell-format: 여기 나열된 변수만 치환된다.
# (전체 치환 모드로 돌리면 run 커맨드의 $HF_TOKEN/$PATH 같은 셸 변수까지 빈 문자열로 날아간다.)
SHELL_FORMAT='${REMOTE_ORG} ${REMOTE_PROJECT} ${REMOTE_CLUSTER} ${REMOTE_GPU_TYPE} ${REMOTE_GPU_COUNT} ${REMOTE_IMAGE} ${REMOTE_ARTIFACT_URI} ${IWM_DATA_MOUNT} ${IWM_GIT_URL} ${IWM_GIT_REF}'

DRY_RUN=0
OUT_PATH=""

usage() {
  sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    -o|--out)
      if [[ -z "${2:-}" ]]; then echo "[!] $1 requires a path." >&2; exit 2; fi
      OUT_PATH="$2"; shift ;;
    -h|--help) usage 0 ;;
    *) echo "[!] Unknown argument '$1'. Use --help." >&2; exit 2 ;;
  esac
  shift
done

[[ -f "$TEMPLATE" ]] || { echo "[!] 템플릿이 없다: $TEMPLATE" >&2; exit 1; }
command -v envsubst >/dev/null 2>&1 || { echo "[!] envsubst 가 필요하다 (gettext 패키지)." >&2; exit 1; }

# --- 1) 변수 로드: 프로세스 환경이 .env 를 이긴다 -------------------------------
PRESET_KV=()
for v in "${REQUIRED_VARS[@]}" REMOTE_SUBMIT_CMD; do
  if [[ -n "${!v:-}" ]]; then PRESET_KV+=("$v=${!v}"); fi
done
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  source "$ENV_FILE"
  set +a
fi
for kv in ${PRESET_KV[@]+"${PRESET_KV[@]}"}; do
  export "$kv"
done

# --- 2) 검증: 누락 변수 전수 나열 ----------------------------------------------
MISSING=()
for v in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!v:-}" ]]; then MISSING+=("$v"); fi
done
if (( ${#MISSING[@]} > 0 )); then
  {
    echo "[!] run spec 렌더에 필요한 변수 ${#MISSING[@]}개가 비어 있다:"
    for v in "${MISSING[@]}"; do echo "      - $v"; done
    echo "    → $ENV_FILE 에 설정하거나 환경변수로 넘겨라. 형식은 .env.example 의 REMOTE_* 블록 참고."
  } >&2
  exit 1
fi

# 소프트 경고 (값은 출력하지 않는다) — scripts/gpu_policy.py 의 허용 범위 + EXP05 요구사항 기준.
case "$REMOTE_GPU_TYPE" in
  A100|H100) ;;
  *) echo "[warn] REMOTE_GPU_TYPE 이 A100/H100 이 아니다. EXP05 stage1 full FT 는 80GB GPU 가 필요하다." >&2 ;;
esac
case "$REMOTE_GPU_COUNT" in
  1|2|4|8) ;;
  *) echo "[warn] REMOTE_GPU_COUNT 는 1|2|4|8 이어야 한다 (scripts/gpu_policy.py)." >&2 ;;
esac

# --- 3) 렌더 -------------------------------------------------------------------
if [[ -n "$OUT_PATH" ]]; then
  RENDERED="$OUT_PATH"
  mkdir -p "$(dirname "$RENDERED")"
else
  RENDERED="$(mktemp -t iwm-remote-run.XXXXXX.yaml)"
fi
envsubst "$SHELL_FORMAT" < "$TEMPLATE" > "$RENDERED"

LEFTOVER="$(grep -c '\${' "$RENDERED" || true)"
if [[ "$LEFTOVER" != "0" ]]; then
  echo "[!] 렌더 결과에 미치환 placeholder 가 ${LEFTOVER}건 남았다: $RENDERED" >&2
  exit 1
fi

if (( DRY_RUN )); then
  echo "[+] dry-run: 렌더만 수행했다 (제출 안 함). 파일: $RENDERED" >&2
  cat "$RENDERED"
  exit 0
fi

# --- 4) 제출 -------------------------------------------------------------------
# 제출 커맨드는 플랫폼마다 다르므로 .env 의 REMOTE_SUBMIT_CMD 에서 받는다.
# 미설정이면 제출하지 않는다 — 검증되지 않은 커맨드를 추측해 실행하지 않기 위해서다.
if [[ -z "${REMOTE_SUBMIT_CMD:-}" ]]; then
  echo "[!] REMOTE_SUBMIT_CMD 가 비어 있다 — 렌더만 하고 제출을 생략한다." >&2
  echo "    렌더 결과: $RENDERED" >&2
  echo "    제출하려면 .env 에 플랫폼의 제출 커맨드를 넣어라. {spec} {org} {project} 가 치환된다. 예:" >&2
  echo "      REMOTE_SUBMIT_CMD='<cli> run create -f {spec} --organization {org} --project {project}'" >&2
  exit 0
fi

SUBMIT="${REMOTE_SUBMIT_CMD//\{spec\}/$RENDERED}"
SUBMIT="${SUBMIT//\{org\}/$REMOTE_ORG}"
SUBMIT="${SUBMIT//\{project\}/$REMOTE_PROJECT}"

SUBMIT_BIN="${SUBMIT%% *}"
if ! command -v "$SUBMIT_BIN" >/dev/null 2>&1; then
  echo "[!] 제출 CLI '$SUBMIT_BIN' 이 설치돼 있지 않다 — 제출 생략." >&2
  echo "    렌더 결과: $RENDERED" >&2
  echo "    설치 후 실행:  $SUBMIT" >&2
  exit 0
fi

echo "[+] 제출: $SUBMIT" >&2
exec bash -c "$SUBMIT"
