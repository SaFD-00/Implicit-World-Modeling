#!/usr/bin/env bash
# WIN1 + WIN2 모두 완료 확인 후 control 창에서 실행
# 순서: stage2 merge → stage1 eval → stage2 eval → HF ep3 push
set -euo pipefail

BASE="/data/seungwoo/Implicit-World-Modeling/Implicit-World-Modeling"
cd "$BASE"
source .env   # HF_TOKEN 로드

echo "=== [1/4] Stage2 Merge (local only) ==="
bash scripts/stage2_merge.sh \
  --model qwen2.5-vl-7b --dataset AC_EXP01 \
  --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
  --exp01-ratios ratio37 --no-hf-upload

echo "=== [2/4] Stage1 Eval ==="
bash scripts/stage1_eval.sh \
  --model qwen2.5-vl-7b --train-dataset AC_EXP01 --exp01-ratio ratio37 \
  --eval-datasets AC_EXP01,MB

echo "=== [3/4] Stage2 Eval ==="
bash scripts/stage2_eval.sh \
  --model qwen2.5-vl-7b --train-dataset AC_EXP01 --exp01-ratio ratio37 \
  --stage1-mode full --stage1-epoch 3 --stage2-mode lora \
  --eval-datasets AC_EXP01,MB

echo "=== [4/4] HF Push ep3 only ==="
OUT="$BASE/outputs/AndroidControl_EXP01/merged"
python3 - "$OUT" "$HF_TOKEN" <<'EOF'
import sys
from huggingface_hub import HfApi

out_base, token = sys.argv[1], sys.argv[2]
api = HfApi(token=token)

pushes = [
    ("qwen2.5-vl-7b_ratio37_stage1_full_world-model/epoch-3",
     "SaFD-00/qwen2.5-vl-7b-ac-exp01-ratio37-world-model-stage1-full-epoch3"),
    ("qwen2.5-vl-7b_ratio37_stage2_lora_base/epoch-3",
     "SaFD-00/qwen2.5-vl-7b-ac-exp01-ratio37-base-stage2-lora-epoch3"),
    ("qwen2.5-vl-7b_ratio37_stage2_lora_world-model_from_full-ep3/epoch-3",
     "SaFD-00/qwen2.5-vl-7b-ac-exp01-ratio37-world-model-stage1-full-epoch3-stage2-lora-epoch3"),
]

for rel, repo_id in pushes:
    folder = f"{out_base}/{rel}"
    print(f"→ {repo_id}")
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model")
    print(f"  Done: https://huggingface.co/{repo_id}")
EOF

echo "=== All Done ==="
