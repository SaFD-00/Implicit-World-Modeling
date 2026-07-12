#!/usr/bin/env python3
"""
apply_llamafactory_patch.py
────────────────────────────────────────────────────────────────
diff token weighted loss 를 위한 LlamaFactory 소스 6파일 패치를 멱등 적용한다.

LlamaFactory 는 Implicit-World-Modeling 메인 repo 에서 gitignore 된 별도 repo 이므로
diff loss 패치는 git 에 커밋하지 않고, 이 스크립트로 working tree 에 재적용한다.
notebook 환경 세팅 셀에서 호출된다.

- 이미 패치돼 있으면 skip (멱등).
- anchor 를 못 찾으면 에러로 중단 (LlamaFactory 버전 불일치).

패치 대상 6파일 / 9개 지점:
  data/converter.py            — _token_weights 필드 통과
  data/processor/supervised.py — token_weights 를 model_inputs 에 포함
  data/collator.py             — labels 기반 token_weights 복원 (3 지점)
  hparams/finetuning_args.py   — use_diff_token_weighted_loss 파라미터
  train/trainer_utils.py       — diff_token_weighted_loss_func 함수
  train/sft/trainer.py         — loss 분기 (2 지점)

패치 내용 레퍼런스: .claude/references/diff_loss/llamafactory_patch.py

usage:
  python scripts/diff_loss/apply_llamafactory_patch.py [--lf-root <LlamaFactory 경로>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = "src/llamafactory"

# 각 파일별 (old, new) 치환 쌍. new 가 이미 있으면 skip, old 가 있으면 치환.
PATCHES: dict[str, list[tuple[str, str]]] = {
    f"{SRC}/data/converter.py": [
        (
            '            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,\n'
            "        }",
            '            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,\n'
            '            "_token_weights": example.get("token_weights", None),  # diff token weighted loss\n'
            "        }",
        ),
    ],
    f"{SRC}/data/processor/supervised.py": [
        (
            '            model_inputs["audios"].append(examples["_audios"][i])\n'
            "\n"
            "        return model_inputs",
            '            model_inputs["audios"].append(examples["_audios"][i])\n'
            '            if "_token_weights" in examples:  # diff token weighted loss\n'
            '                model_inputs["token_weights"].append(examples["_token_weights"][i])\n'
            "\n"
            "        return model_inputs",
        ),
    ],
    f"{SRC}/data/collator.py": [
        (
            '    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:\n'
            "        batch_images, batch_videos, batch_audios = [], [], []\n"
            "        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []",
            '    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:\n'
            "        # diff token weighted loss: token_weights는 DataCollatorForSeq2Seq가 모르는 필드이므로 분리\n"
            '        token_weights_list = [feature.pop("token_weights", None) for feature in features]\n'
            "        has_token_weights = any(w is not None for w in token_weights_list)\n"
            "        batch_images, batch_videos, batch_audios = [], [], []\n"
            "        batch_imglens, batch_vidlens, batch_audlens, batch_input_ids = [], [], [], []",
        ),
        (
            '        if "image_bound" in features:  # for minicpmv inputs\n'
            '            bsz, seq_length = features["input_ids"].shape\n'
            '            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)\n'
            '            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}\n'
            "\n"
            "        return features",
            '        if "image_bound" in features:  # for minicpmv inputs\n'
            '            bsz, seq_length = features["input_ids"].shape\n'
            '            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)\n'
            '            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}\n'
            "\n"
            "        # diff token weighted loss: labels(-100) 마스크로 prefix/assistant 경계를 판단해 복원.\n"
            "        # n_asst >= n_w: template prefix(<think> 등)로 토큰이 늘면 뒤에서 정렬(offset).\n"
            "        # n_asst <  n_w: cutoff truncation으로 토큰이 줄면 앞에서부터 n_asst개만 사용.\n"
            "        if has_token_weights:\n"
            '            max_len = features["input_ids"].shape[1]\n'
            '            labels = features["labels"]\n'
            "            batch_weights = []\n"
            "            for i, w in enumerate(token_weights_list):\n"
            "                full_w = torch.ones(max_len, dtype=torch.float32)\n"
            "                full_w[labels[i] == IGNORE_INDEX] = 0.0\n"
            "                if w is not None:\n"
            "                    w = list(w)\n"
            "                    asst_indices = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]\n"
            "                    n_asst = len(asst_indices)\n"
            "                    n_w = len(w)\n"
            "                    if n_asst >= n_w:\n"
            "                        offset = n_asst - n_w\n"
            "                        for j in range(n_w):\n"
            "                            full_w[asst_indices[offset + j]] = w[j]\n"
            "                    else:\n"
            "                        for j in range(n_asst):\n"
            "                            full_w[asst_indices[j]] = w[j]\n"
            "                batch_weights.append(full_w)\n"
            '            features["token_weights"] = torch.stack(batch_weights)\n'
            "\n"
            "        return features",
        ),
        (
            '        keys_on_seq_dim_1 = {"input_ids", "labels", "attention_mask", "token_type_ids"}',
            '        keys_on_seq_dim_1 = {"input_ids", "labels", "attention_mask", "token_type_ids", "token_weights"}',
        ),
    ],
    f"{SRC}/hparams/finetuning_args.py": [
        (
            "    eaft_alpha: float = field(\n"
            "        default=1.0,\n"
            '        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},\n'
            "    )\n"
            "    freeze_vision_tower: bool = field(",
            "    eaft_alpha: float = field(\n"
            "        default=1.0,\n"
            '        metadata={"help": "The alpha parameter for EAFT loss to control the power of adaptive weight."},\n'
            "    )\n"
            "    use_diff_token_weighted_loss: bool = field(\n"
            "        default=False,\n"
            "        metadata={\n"
            '            "help": (\n'
            "                \"Whether to apply per-token loss weights supplied via the 'token_weights' \"\n"
            '                "field in the dataset. Diff tokens (ADDED/MODIFIED) receive higher weights "\n'
            '                "defined during preprocessing."\n'
            "            )\n"
            "        },\n"
            "    )\n"
            "    freeze_vision_tower: bool = field(",
        ),
    ],
    f"{SRC}/train/trainer_utils.py": [
        (
            "    if num_items_in_batch is not None:\n"
            "        total_loss = weighted_losses.sum()\n"
            "        if torch.is_tensor(num_items_in_batch):\n"
            "            num_items_in_batch = num_items_in_batch.to(total_loss.device)\n"
            "        loss = total_loss / num_items_in_batch\n"
            "    else:\n"
            "        loss = weighted_losses.mean()\n"
            "\n"
            "    return loss\n"
            "\n"
            "\n"
            "def nested_detach(",
            "    if num_items_in_batch is not None:\n"
            "        total_loss = weighted_losses.sum()\n"
            "        if torch.is_tensor(num_items_in_batch):\n"
            "            num_items_in_batch = num_items_in_batch.to(total_loss.device)\n"
            "        loss = total_loss / num_items_in_batch\n"
            "    else:\n"
            "        loss = weighted_losses.mean()\n"
            "\n"
            "    return loss\n"
            "\n"
            "\n"
            "def diff_token_weighted_loss_func(\n"
            '    outputs: "torch.Tensor",\n'
            '    labels: "torch.Tensor",\n'
            '    num_items_in_batch: Optional["torch.Tensor"] = None,\n'
            '    token_weights: Optional["torch.Tensor"] = None,\n'
            ') -> "torch.Tensor":\n'
            '    r"""Per-token weighted cross-entropy loss for the diff token weighted loss.\n'
            "\n"
            "    Tokens belonging to ADDED/MODIFIED diff elements carry weights > 1.0 (set during\n"
            "    preprocessing); all other tokens carry weight 1.0. When `token_weights` is None or\n"
            "    all-ones, this reduces exactly to the standard cross-entropy mean.\n"
            '    """\n'
            '    logits = outputs.get("logits")\n'
            "    if logits is None:\n"
            '        return outputs.get("loss", torch.tensor(0.0))\n'
            "\n"
            "    logits = logits.float()\n"
            "    vocab_size = logits.size(-1)\n"
            "\n"
            "    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)\n"
            "    shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)\n"
            "\n"
            "    per_token_loss = torch.nn.functional.cross_entropy(\n"
            '        shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none"\n'
            "    )\n"
            "\n"
            "    if token_weights is not None:\n"
            "        shift_weights = token_weights[..., 1:].contiguous().view(-1).to(logits.device)\n"
            "    else:\n"
            "        shift_weights = (shift_labels != IGNORE_INDEX).float()\n"
            "\n"
            "    weighted_loss = per_token_loss * shift_weights\n"
            "\n"
            "    valid_mask = shift_labels != IGNORE_INDEX\n"
            "    denom = shift_weights[valid_mask].sum().clamp(min=1.0)\n"
            "    loss = weighted_loss[valid_mask].sum() / denom\n"
            "\n"
            "    if num_items_in_batch is not None:\n"
            "        n_valid = valid_mask.sum().clamp(min=1)\n"
            "        if torch.is_tensor(num_items_in_batch):\n"
            "            num_items_in_batch = num_items_in_batch.to(loss.device)\n"
            "        loss = loss * (n_valid.float() / num_items_in_batch)\n"
            "\n"
            "    return loss\n"
            "\n"
            "\n"
            "def nested_detach(",
        ),
    ],
    f"{SRC}/train/sft/trainer.py": [
        (
            "        elif finetuning_args.use_asft_loss:\n"
            "            from ..trainer_utils import asft_loss_func\n"
            "\n"
            "            self.compute_loss_func = partial(\n"
            "                asft_loss_func,\n"
            "                asft_alpha=finetuning_args.asft_alpha,\n"
            "            )\n"
            "\n"
            '        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization',
            "        elif finetuning_args.use_asft_loss:\n"
            "            from ..trainer_utils import asft_loss_func\n"
            "\n"
            "            self.compute_loss_func = partial(\n"
            "                asft_loss_func,\n"
            "                asft_alpha=finetuning_args.asft_alpha,\n"
            "            )\n"
            "        elif finetuning_args.use_diff_token_weighted_loss:\n"
            "            from ..trainer_utils import diff_token_weighted_loss_func\n"
            "\n"
            "            self.compute_loss_func = diff_token_weighted_loss_func\n"
            "\n"
            '        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization',
        ),
        (
            "            outputs = model(**inputs)\n"
            '            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)\n'
            "        else:\n"
            "            return super().compute_loss(model, inputs, *args, **kwargs)",
            "            outputs = model(**inputs)\n"
            '            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)\n'
            "        elif self.finetuning_args.use_diff_token_weighted_loss:\n"
            '            token_weights = inputs.pop("token_weights", None)\n'
            "            outputs = model(**inputs)\n"
            "            return self.compute_loss_func(\n"
            "                outputs,\n"
            '                inputs["labels"],\n'
            '                num_items_in_batch=kwargs.get("num_items_in_batch"),\n'
            "                token_weights=token_weights,\n"
            "            )\n"
            "        else:\n"
            "            return super().compute_loss(model, inputs, *args, **kwargs)",
        ),
    ],
}


def _default_lf_root() -> Path:
    # scripts/diff_loss/apply_llamafactory_patch.py → parents[2] = Implicit-World-Modeling 루트
    return Path(__file__).resolve().parents[2] / "LlamaFactory"


def apply_patches(lf_root: Path) -> int:
    n_patched = n_skipped = 0
    errors: list[str] = []

    for rel_path, pairs in PATCHES.items():
        fpath = lf_root / rel_path
        if not fpath.is_file():
            errors.append(f"파일 없음: {fpath}")
            continue

        content = fpath.read_text(encoding="utf-8")
        original = content
        for idx, (old, new) in enumerate(pairs):
            if new in content:
                n_skipped += 1
            elif old in content:
                content = content.replace(old, new, 1)
                n_patched += 1
            else:
                errors.append(f"anchor 불일치: {rel_path} [지점 {idx}]")

        if content != original:
            fpath.write_text(content, encoding="utf-8")
            print(f"  [patched] {rel_path}")
        else:
            print(f"  [skip]    {rel_path}")

    print(f"\n[완료] 적용 {n_patched}개 / skip {n_skipped}개 / 에러 {len(errors)}개")
    for e in errors:
        print(f"  [ERROR] {e}")
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LlamaFactory diff loss 패치 멱등 적용"
    )
    parser.add_argument(
        "--lf-root",
        type=Path,
        default=_default_lf_root(),
        help="LlamaFactory 루트 경로 (기본: Implicit-World-Modeling/LlamaFactory)",
    )
    args = parser.parse_args()

    lf_root: Path = args.lf_root
    if not (lf_root / SRC).is_dir():
        print(f"[ERROR] LlamaFactory 소스를 찾을 수 없음: {lf_root}/{SRC}")
        sys.exit(1)

    print(f"[INFO] LlamaFactory diff loss 패치 적용: {lf_root}")
    sys.exit(apply_patches(lf_root))


if __name__ == "__main__":
    main()
