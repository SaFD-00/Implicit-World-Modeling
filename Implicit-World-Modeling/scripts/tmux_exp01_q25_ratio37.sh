#!/usr/bin/env bash
# EXP01 qwen2.5-vl-7b ratio37 tmux 스케줄 학습
# 실행: bash scripts/tmux_exp01_q25_ratio37.sh
#
# 창 구성:
#   0:control  — 모니터링 + 학습 완료 후 post_train_q25_ratio37.sh 실행
#   1:s1+s2wm  — stage1(GPU 0+1) → s1-merge(CPU) → stage2 world-model(GPU 0)
#   2:s2base   — stage2 base (GPU 1, 즉시 시작)

SESSION="iwm-exp01-q25-r37"
BASE="/data/seungwoo/Implicit-World-Modeling/Implicit-World-Modeling"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "control" -c "$BASE"

# --- Window 1: stage1 → s1-merge → stage2 world-model ---
tmux new-window -t "$SESSION" -n "s1+s2wm" -c "$BASE"
tmux send-keys -t "$SESSION:s1+s2wm" \
  "conda activate implicit-world-modeling && \\
NPROC_PER_NODE_OVERRIDE=2 CUDA_VISIBLE_DEVICES=0,1 \\
  bash scripts/stage1_train.sh \\
    --model qwen2.5-vl-7b --dataset AC_EXP01 --exp01-ratios ratio37 && \\
bash scripts/stage1_merge.sh \\
  --model qwen2.5-vl-7b --dataset AC_EXP01 --exp01-ratios ratio37 --no-hf-upload && \\
NPROC_PER_NODE_OVERRIDE=1 CUDA_VISIBLE_DEVICES=0 \\
  bash scripts/stage2_train.sh \\
    --model qwen2.5-vl-7b --dataset AC_EXP01 \\
    --stage1-mode full --stage1-epoch 3 --stage2-mode lora \\
    --exp01-ratios ratio37 --variants world-model-full && \\
echo '[WIN1] stage1+s2wm DONE'" Enter

# --- Window 2: stage2 base (GPU 1, 즉시 시작) ---
tmux new-window -t "$SESSION" -n "s2base" -c "$BASE"
tmux send-keys -t "$SESSION:s2base" \
  "conda activate implicit-world-modeling && \\
NPROC_PER_NODE_OVERRIDE=1 CUDA_VISIBLE_DEVICES=1 \\
  bash scripts/stage2_train.sh \\
    --model qwen2.5-vl-7b --dataset AC_EXP01 \\
    --stage1-mode full --stage2-mode lora \\
    --exp01-ratios ratio37 --variants base && \\
echo '[WIN2] s2base DONE'" Enter

tmux select-window -t "$SESSION:control"
echo "[tmux] Session '$SESSION' 시작됨."
echo "[tmux] 두 창이 모두 DONE 출력하면 control 창에서 아래를 실행하세요:"
echo "         bash scripts/post_train_q25_ratio37.sh"
tmux attach-session -t "$SESSION"
