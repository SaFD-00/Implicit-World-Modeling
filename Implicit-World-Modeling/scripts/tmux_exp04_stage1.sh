#!/usr/bin/env bash
# EXP04 (EXP03 프롬프트 업그레이드) Stage 1 LORA tmux 학습 — GPU 0,1 (NPROC=2).
# 실행: bash scripts/tmux_exp04_stage1.sh
#
# 창 구성:
#   0:control — 모니터링 (학습은 1:s1lora 에서 진행)
#   1:s1lora  — qwen3-vl-8b → qwen2.5-vl-7b stage1 lora 순차 (GPU 0+1, NPROC=2)
#
# GPU 타임라인:
#   GPU 0+1  [--qwen3-vl-8b s1 lora--][--qwen2.5-vl-7b s1 lora--]
#
# global batch = per_device(1) × grad_accum(32) × world_size(2) = 64
#   (EXP03 single-GPU global batch 64 와 동일 — YAML grad_accum 을 64→32 로 baked).
#
# EXP04 는 EXP03 와 동일 멤버십·좌표 표현 + stage1 프롬프트 업그레이드(swipe/html-style-XML)
# 실험군이라, EXP03 가 학습한 LORA 2 모델과 직접 비교한다 (Stage 2 보류).

SESSION="iwm-exp04-stage1"
BASE="/data/seungwoo/Implicit-World-Modeling/Implicit-World-Modeling"
CENV="/data/seungwoo/Implicit-World-Modeling/.conda-env"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "control" -c "$BASE"

# --- Window 1: stage1 lora 2 모델 순차 (GPU 0,1) ---
tmux new-window -t "$SESSION" -n "s1lora" -c "$BASE"
tmux send-keys -t "$SESSION:s1lora" \
  "conda activate $CENV && \\
NPROC_PER_NODE_OVERRIDE=2 CUDA_VISIBLE_DEVICES=0,1 \\
  bash scripts/stage1_train.sh --model qwen3-vl-8b --dataset AC_EXP04 --stage1-mode lora && \\
NPROC_PER_NODE_OVERRIDE=2 CUDA_VISIBLE_DEVICES=0,1 \\
  bash scripts/stage1_train.sh --model qwen2.5-vl-7b --dataset AC_EXP04 --stage1-mode lora && \\
echo '[EXP04] stage1 lora 2 모델 DONE'" Enter

tmux select-window -t "$SESSION:control"
echo "[tmux] Session '$SESSION' 시작됨 (detached)."
echo "[tmux] 진행 확인:  tmux attach -t $SESSION   (창 1:s1lora)"
echo "[tmux] 학습 로그:  outputs/logs/ + LlamaFactory trainer_log.jsonl"
