# LlamaFactory patches

LlamaFactory (`Implicit-World-Modeling/LlamaFactory/`) is a third-party working
tree excluded from this repo's git via `.gitignore`. It is not a submodule.
This directory is the git-tracked record of the local modifications made to
that working tree, so the state can be reproduced from a pristine clone.

## Pin

```
99464b3d034fd19fa73486f05e3b64b963e1b423
```
(`[misc] code lint (#10439)`, remote = hiyouga/LlamaFactory)

## Apply order

1. `git clone <hiyouga/LlamaFactory> LlamaFactory && cd LlamaFactory`
2. `git checkout 99464b3d034fd19fa73486f05e3b64b963e1b423`
3. `git apply patches/llamafactory/0001-diff-loss.patch` — applies the diff-loss
   changes to 6 files under `src/llamafactory/` (collator, converter,
   supervised processor, finetuning_args, sft trainer, trainer_utils).
4. Merge `configs/lf_dataset/dataset_info.json` (`IWM-*` entries only) into
   `data/dataset_info.json`.
5. Copy `configs/train/*` into `examples/custom/` to restore the 74 as-trained
   YAML configs.

## Why this exists

LlamaFactory is not part of this git history, so without this directory the
local patch (diff-loss weighting) and the as-trained run configs would only
exist in the gitignored working tree — a single point of failure. This
directory is the source of truth for those changes; the working tree is a
disposable build artifact of it.

## Regeneration

From the LF working tree, at the pin above:
```
git diff -- src/ > patches/llamafactory/0001-diff-loss.patch
```
`MANIFEST.sha256` is regenerated with `sha256sum` over the 74 YAMLs
(`configs/train/`), the 6 patched `src/` files (post-patch, i.e. current LF
working tree), and `configs/lf_dataset/dataset_info.json`; verify with
`sha256sum -c patches/llamafactory/MANIFEST.sha256`.
