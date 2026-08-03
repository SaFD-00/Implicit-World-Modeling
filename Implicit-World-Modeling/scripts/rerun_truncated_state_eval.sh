#!/usr/bin/env bash
# 절단(max_new_tokens=1024) state leaf 재추론 — EXP07 과 동일한 12288 예산으로 다시 뽑는다.
#
# 왜 필요한가
# -----------
# 일부 stage1 state leaf 는 `build_infer_cmd` 가 `--max_new_tokens` 를 안 넘기던 시절에
# vllm 기본값 1024 로 생성돼 예측이 하드 컷됐다. 잘린 예측은 Hungarian F1 을 구조적으로
# 과소평가하고 `copy_rate_pred` 를 한쪽으로 틀리게 만들어 **채점 자체가 무의미**하다.
# 현재 `stage1_eval.sh` 는 state 에 12288 을 준다(98행) — EXP07 이 실제로 쓴 값과 같다
# (`.gen_done`: max_new_tokens=12288 / seed=42). 즉 **코드 수정 없이 재실행만 하면 된다.**
#
# ⚠️ 절단 판정은 반드시 "토큰" 으로 한다 — 날짜도, 문자 길이도 아니다
# --------------------------------------------------------------------
# (1) mtime 판정선("2026-07-28 23:38 UTC 이전 = 절단")은 틀렸다. 같은 시기 산출물이 갈린다.
# (2) **문자 길이 판정도 틀렸다 — 양방향으로.** 2026-08-03 전수 실측:
#     · false negative: EXP02 base 는 predict 최장이 **29,286자**인데 그 행의 토큰이 **정확히
#       1024** 였다(공백 수천 개 꼬리). 문자로 보면 "정상"으로 통과해 버린다.
#     · false positive 도 난다 — 모델이 그냥 짧게 생성한 leaf 를 절단으로 오분류한다.
#     ⇒ 판정식: **`count(tok == 1024) / n >= 0.05`** — 1024 는 고정 상수다.
#       "구 예산값에 대량으로 쌓인 모드"가 절단의 지문이다. 자유 생성은 특정 길이에 몰리지 않는다.
#   (3) `max_tok <= 1024` 를 조건에 넣지 마라. 저장된 텍스트를 **재토크나이즈**하면 길이가
#       보존되지 않아 절단 leaf 에도 1024 를 몇 토큰 넘긴 행이 하나씩 섞인다 — EXP01
#       `3-8b_r73/lora/ep3`=1060, EXP02 `lora/ep1`=1025·`ep2`=1030 이 그 사례이고, 셋 다
#       1024-토큰 행이 36~46% 라 **절단이 확실하다**. 상한만 보면 이 셋을 놓친다.
#
# 대상 17 leaf (토큰 실측 확정 · 1024 몰림 비율)
#   EXP01 2.5-7b_r73  lora_wm ep2   32.0%     EXP01 3-8b_r73  base          29.1%
#   EXP01 2.5-7b_r73  lora_wm ep3   31.2%     EXP01 3-8b_r73  lora_wm ep1   41.4%
#   EXP01 3-8b_r37    base          29.1%     EXP01 3-8b_r73  lora_wm ep2   45.5%
#   EXP01 3-8b_r37    lora_wm ep1   40.9%     EXP01 3-8b_r73  lora_wm ep3   45.7%
#   EXP01 3-8b_r37    lora_wm ep2   40.2%     EXP02 3-8b      base(오배치)   29.1%
#   EXP01 3-8b_r37    lora_wm ep3   37.7%     EXP02 3-8b      lora_wm ep1   37.9%
#   EXP02 3-8b        lora_wm ep2   36.6%     EXP02 3-8b      lora_wm ep3   48.8%
#   EXP03 2.5-7b      lora_wm ep3   68.1%     EXP03 3-8b      base          70.1%
#   EXP03 3-8b        lora_wm ep3   71.2%
#
# 재추론이 **불필요한** 정상 leaf 16개 (1024 몰림 0.0~0.1%)
#   EXP01 2.5-7b_r73 base·lora_wm ep1 · EXP03 2.5-7b base · EXP05 4 leaf · EXP07 9 leaf
#   ⇒ EXP01 은 12 중 10, EXP02 는 4 전부, EXP03 은 4 중 3 이 절단이다.
#
# ⚠️ EXP02 base 는 leaf 이름이 `on-AC_EXP01-state` 로 오배치돼 있다(디렉터리 EXP02 가 맞다).
#    재추론하면 `on-AC_EXP02-state` 로 새로 생기므로, 끝난 뒤 옛 오배치 leaf 를 정리할 것.
#
# 사용법
#   scripts/rerun_truncated_state_eval.sh              # dry-run (명령만 출력, 기본)
#   scripts/rerun_truncated_state_eval.sh --run        # 실제 실행
#   scripts/rerun_truncated_state_eval.sh --run g5     # 특정 그룹만 (g1..g6)
#
# 선행 조건
#   - GPU 여유. qwen3-vl-8b × EXP03 는 **2장(TP=2) 필수** — 프롬프트가 길어 32GB 1장에
#     `gpu_memory_utilization` 을 어떻게 줘도 양쪽으로 막힌다. EXP01/EXP02 의 8b 는
#     프롬프트가 짧아 1장으로 될 수 있으나 미검증이라 실패하면 VLLM_TP_SIZE=2 로 올려라.
#   - EXP02 는 local merged 가 있고, EXP01·EXP03 은 없어서 **HF Hub fallback** 이다.
#     HF repo 가 껍데기(파일 0개)로 선생성돼 있을 수 있으니 먼저 접근을 확인할 것.
#
# 채점은 분리한다
#   EVAL_SKIP_SCORE=1 로 추론만 하고 `.gen_done` 마커를 남긴다. 채점은 GPU 를 안 쓰므로
#   끝난 뒤 `scripts/rebuild_eval_metrics.sh` 로 배치 처리한다(마커가 대상 선정 기준이다).
set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

RUN=0
GROUP="all"
for a in "$@"; do
  case "$a" in
    --run) RUN=1 ;;
    g[1-6]) GROUP="$a" ;;
    -h|--help) sed -n '1,60p' "$0"; exit 0 ;;
    *) echo "[!] 알 수 없는 인자: $a" >&2; exit 2 ;;
  esac
done

COMMON_ENV="EVAL_TASKS=state EVAL_SKIP_WOA=1 EVAL_SKIP_SCORE=1 VLLM_SEED=42"

# 카드 하나를 "쓸 수 있다"고 보는 최소 여유(MiB). 7B/8B + 긴 state 프롬프트의 KV cache 까지
# 감안한 값이다. 이 값을 낮추면 엔진은 뜨는데 KV cache 가 모자라 런타임 OOM 으로 죽는다.
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-20480}"

gpu_free_count() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v min="$GPU_MIN_FREE_MIB" '{ if (($2-$1) > min) n++ } END { print n+0 }'
}

emit() {  # $1=그룹 $2=필요GPU수 $3=설명 / 나머지=명령
  local g="$1" ngpu="$2" desc="$3"; shift 3
  [ "$GROUP" != "all" ] && [ "$GROUP" != "$g" ] && return 0
  echo "───────────────────────────────────────────────────────────────"
  echo "[$g] $desc  (GPU ${ngpu}장 필요)"
  echo "  $*"
  if [ "$RUN" -eq 1 ]; then
    local free; free=$(gpu_free_count)
    if [ "$free" -lt "$ngpu" ]; then
      echo "  [!] 건너뜀 — 여유 GPU ${free}장 < 필요 ${ngpu}장" >&2; return 0
    fi
    echo "  [>] 실행"; bash -c "$*"
  fi
}

echo "== 절단 state leaf 재추론 (대상 17 leaf) =="
echo "모드: $([ "$RUN" -eq 1 ] && echo 실행 || echo 'dry-run (실행하려면 --run)')  ·  그룹: $GROUP"
echo "현재 여유 GPU(${GPU_MIN_FREE_MIB}MiB 기준): $(gpu_free_count) 장"
echo

emit g1 1 "EXP01·ratio73 · qwen2.5-vl-7b · lora_wm ep2,3" \
  "CUDA_VISIBLE_DEVICES=0 $COMMON_ENV scripts/stage1_eval.sh \
     --model qwen2.5-vl-7b --train-dataset AC_EXP01_ratio73 --eval-datasets AC_EXP01 \
     --variants lora_world_model --epochs 2,3"

emit g2 1 "EXP01·ratio37 · qwen3-vl-8b · base + lora_wm ep1,2,3" \
  "CUDA_VISIBLE_DEVICES=0 $COMMON_ENV scripts/stage1_eval.sh \
     --model qwen3-vl-8b --train-dataset AC_EXP01_ratio37 --eval-datasets AC_EXP01 \
     --variants base,lora_world_model --epochs 1,2,3"

emit g3 1 "EXP01·ratio73 · qwen3-vl-8b · base + lora_wm ep1,2,3" \
  "CUDA_VISIBLE_DEVICES=0 $COMMON_ENV scripts/stage1_eval.sh \
     --model qwen3-vl-8b --train-dataset AC_EXP01_ratio73 --eval-datasets AC_EXP01 \
     --variants base,lora_world_model --epochs 1,2,3"

emit g4 1 "EXP02 · qwen3-vl-8b · base + lora_wm ep1,2,3 (4 leaf 전부 절단)" \
  "CUDA_VISIBLE_DEVICES=0 $COMMON_ENV scripts/stage1_eval.sh \
     --model qwen3-vl-8b --train-dataset AC_EXP02 --eval-datasets AC_EXP02 \
     --variants base,lora_world_model --epochs 1,2,3"

emit g5 1 "EXP03 · qwen2.5-vl-7b · lora_wm ep3  (base 는 정상이라 제외)" \
  "CUDA_VISIBLE_DEVICES=0 $COMMON_ENV scripts/stage1_eval.sh \
     --model qwen2.5-vl-7b --train-dataset AC_EXP03 --eval-datasets AC_EXP03 \
     --variants lora_world_model --epochs 3"

emit g6 2 "EXP03 · qwen3-vl-8b · base + lora_wm ep3  (TP=2 필수)" \
  "CUDA_VISIBLE_DEVICES=0,1 VLLM_TP_SIZE=2 $COMMON_ENV scripts/stage1_eval.sh \
     --model qwen3-vl-8b --train-dataset AC_EXP03 --eval-datasets AC_EXP03 \
     --variants base,lora_world_model --epochs 3"

echo "───────────────────────────────────────────────────────────────"
cat <<'EOF'

다음 단계
  1) 추론이 끝나면 각 leaf 에 `.gen_done` 이 생긴다. 절단이 풀렸는지 **토큰으로** 확인:
       python - <<'PY'
       import json,sys
       from collections import Counter
       from transformers import AutoTokenizer
       t=AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", local_files_only=True)
       L=[len(t.encode(json.loads(l).get("predict",""),add_special_tokens=False))
          for l in open(sys.argv[1])]
       c=Counter(L); print("tok_max",max(L),"모드비",f"{c[max(L)]/len(L):.1%}")
       PY
     tok_max 가 1024 를 넘고 모드비가 0% 에 가까우면 풀린 것이다.
  2) 채점은 GPU 없이 배치로:
       scripts/rebuild_eval_metrics.sh          # .gen_done 마커가 있는 leaf 만 집는다
  3) EXP02 base 재추론 후에는 오배치 leaf `on-AC_EXP01-state` 를 정리할 것
     (디렉터리 EXP02 가 규범이고, 정본 툴링은 `on-{디렉터리EXP}` 로 이름을 만든다).
EOF
