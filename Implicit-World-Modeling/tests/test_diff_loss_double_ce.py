"""diff loss 이중 CE 제거 + chunked CE 회귀 테스트 (patches/llamafactory/0002-double-ce-fix.patch).

두 가지 수정을 고정한다:

1. **이중 CE 제거** (``train/sft/trainer.py``)
   ``model(**inputs)`` 에 ``labels`` 가 그대로 실려 HF 내부 CE(``ForCausalLMLoss``)가 한 번 더 돌던
   문제. 결과는 버려지지만 fp32 [B, S, V] autograd 그래프가 남아 activation 을 낭비했다
   (vocab=151936 기준 0.566 GiB / 1k token). → forward 전에 labels 를 pop 한다.

2. **chunked CE** (``train/trainer_utils.py::diff_token_weighted_loss_func``)
   logits 전량을 ``.float()`` 하던 것을 시퀀스 청크 단위 upcast 로 바꿨다.
   **가중합/정규화 순서를 그대로 보존하므로 결과는 bit-exact** 여야 한다 —
   loss 차이 0.0, max|grad 차이| 0.0 (tolerance 없음, ``torch.equal`` 수준).

Run:
    pytest tests/test_diff_loss_double_ce.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.trainer_utils import diff_token_weighted_loss_func

# diff loss v2 가중 체계 (scripts/diff_loss/token_weight_builder_v2.py)
W_DIFF = 1.0  # ADDED / MODIFIED
W_NON_DIFF = 0.25  # UNCHANGED


# ── 레퍼런스: 0002 이전의 구현 (0001 이 넣은 그대로) ──────────────────────────
# LF 워킹트리를 되돌려 비교하지 않기 위해 여기에 인라인 보존한다. 수정 금지.
def reference_diff_token_weighted_loss_func(
    outputs,
    labels,
    num_items_in_batch=None,
    token_weights=None,
):
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    logits = logits.float()
    vocab_size = logits.size(-1)

    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)

    per_token_loss = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none"
    )

    if token_weights is not None:
        shift_weights = token_weights[..., 1:].contiguous().view(-1).to(logits.device)
    else:
        shift_weights = (shift_labels != IGNORE_INDEX).float()

    weighted_loss = per_token_loss * shift_weights

    valid_mask = shift_labels != IGNORE_INDEX
    denom = shift_weights[valid_mask].sum().clamp(min=1.0)
    loss = weighted_loss[valid_mask].sum() / denom

    if num_items_in_batch is not None:
        n_valid = valid_mask.sum().clamp(min=1)
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss * (n_valid.float() / num_items_in_batch)

    return loss


# ── 픽스처 ────────────────────────────────────────────────────────────────────
def make_batch(batch_size=2, seq_len=768, vocab_size=1024, seed=0, dtype=torch.float32):
    """prompt 절반 IGNORE_INDEX 마스킹 + diff/non-diff 가중치 혼합 배치."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch_size, seq_len, vocab_size, generator=g, dtype=dtype)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)

    n_prompt = seq_len // 2
    labels[:, :n_prompt] = IGNORE_INDEX  # prompt 구간 마스킹

    # assistant 구간: 앞 1/3 은 diff(1.0), 나머지는 non-diff(0.25). prompt 구간은 0.0.
    weights = torch.full((batch_size, seq_len), W_NON_DIFF)
    weights[:, :n_prompt] = 0.0
    diff_end = n_prompt + (seq_len - n_prompt) // 3
    weights[:, n_prompt:diff_end] = W_DIFF
    return logits, labels, weights


def run_both(logits, labels, weights, num_items_in_batch=None, chunk_size=128):
    """구/신 두 경로를 각각 forward+backward 하고 (loss, grad) 를 돌려준다."""
    out = []
    for fn in (reference_diff_token_weighted_loss_func, diff_token_weighted_loss_func):
        x = logits.detach().clone().requires_grad_(True)
        kwargs = {} if fn is reference_diff_token_weighted_loss_func else {"chunk_size": chunk_size}
        loss = fn(
            {"logits": x},
            labels,
            num_items_in_batch=num_items_in_batch,
            token_weights=weights,
            **kwargs,
        )
        loss.backward()
        out.append((loss.detach(), x.grad.detach()))
    return out


# ── bit-exact: loss / grad ────────────────────────────────────────────────────
@pytest.mark.parametrize("num_items", [None, 512])
@pytest.mark.parametrize("chunk_size", [128, 333, 4096])  # 나누어떨어짐 / 안 떨어짐 / 청크 1개
def test_chunked_ce_is_bit_exact(num_items, chunk_size):
    logits, labels, weights = make_batch()
    n = None if num_items is None else torch.tensor(float(num_items))

    (old_loss, old_grad), (new_loss, new_grad) = run_both(
        logits, labels, weights, num_items_in_batch=n, chunk_size=chunk_size
    )

    loss_diff = (old_loss - new_loss).abs().item()
    grad_diff = (old_grad - new_grad).abs().max().item()
    print(f"\n[chunk={chunk_size} num_items={num_items}] loss={new_loss.item():.9f} "
          f"|Δloss|={loss_diff:.3e} max|Δgrad|={grad_diff:.3e}")

    assert torch.equal(old_loss, new_loss), f"loss가 bit-exact 하지 않다: |Δ|={loss_diff:.3e}"
    assert torch.equal(old_grad, new_grad), f"grad가 bit-exact 하지 않다: max|Δ|={grad_diff:.3e}"
    assert loss_diff == 0.0 and grad_diff == 0.0


def test_bit_exact_without_token_weights():
    """token_weights=None (uniform) 경로도 동일해야 한다."""
    logits, labels, _ = make_batch(seed=1)
    (old_loss, old_grad), (new_loss, new_grad) = run_both(logits, labels, None, chunk_size=200)
    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old_grad, new_grad)


def test_all_masked_chunk_is_handled():
    """청크 전체가 IGNORE_INDEX 인 경우에도 구현이 일치한다 (0으로 나누지 않음)."""
    logits, labels, weights = make_batch(seq_len=512, seed=2)
    labels[:, :400] = IGNORE_INDEX  # 첫 청크(128) 전체가 마스킹됨
    weights[:, :400] = 0.0
    (old_loss, old_grad), (new_loss, new_grad) = run_both(logits, labels, weights, chunk_size=128)
    assert torch.isfinite(new_loss)
    assert torch.equal(old_loss, new_loss)
    assert torch.equal(old_grad, new_grad)


# ── 가중 체계 회귀: 값 자체가 맞는가 (diff 1.0 / non-diff 0.25) ────────────────
def test_weighting_scheme_matches_manual_computation():
    logits, labels, weights = make_batch(batch_size=1, seq_len=256, seed=3)
    loss = diff_token_weighted_loss_func({"logits": logits}, labels, token_weights=weights)

    ce = torch.nn.functional.cross_entropy(
        logits[0, :-1, :].float(), labels[0, 1:], ignore_index=IGNORE_INDEX, reduction="none"
    )
    w = weights[0, 1:]
    m = labels[0, 1:] != IGNORE_INDEX
    expected = (ce * w)[m].sum() / w[m].sum().clamp(min=1.0)

    assert torch.allclose(loss, expected, rtol=0, atol=0)
    assert set(weights[0, 128:].tolist()) <= {W_DIFF, W_NON_DIFF}, "가중치는 1.0 / 0.25 만"


# ── 이중 CE 제거: forward 가 labels 를 보지 못해야 한다 ────────────────────────
def test_compute_loss_does_not_pass_labels_to_model():
    """labels 가 forward 로 흘러가면 HF 내부 CE 가 한 번 더 돈다 (이중 CE)."""
    seen = {}

    def fake_model(**kwargs):
        seen.update(kwargs)
        return {"logits": logits}

    got = {}

    def fake_loss_func(outputs, labels, num_items_in_batch=None, token_weights=None):
        got["labels"] = labels
        got["token_weights"] = token_weights
        got["num_items_in_batch"] = num_items_in_batch
        return torch.tensor(0.0)

    logits, labels, weights = make_batch(batch_size=1, seq_len=8, vocab_size=16, seed=4)
    trainer = SimpleNamespace(
        finetuning_args=SimpleNamespace(use_asft_loss=False, use_diff_token_weighted_loss=True),
        compute_loss_func=fake_loss_func,
    )
    inputs = {"input_ids": torch.zeros(1, 8, dtype=torch.long), "labels": labels, "token_weights": weights}

    CustomSeq2SeqTrainer.compute_loss(trainer, fake_model, inputs, num_items_in_batch=4)

    assert "labels" not in seen, "labels 가 model forward 로 전달됐다 → HF 내부 CE 가 중복 실행된다"
    assert "token_weights" not in seen, "token_weights 는 모델이 모르는 필드다"
    assert seen["input_ids"] is inputs["input_ids"]
    assert torch.equal(got["labels"], labels), "loss 함수는 pop 한 labels 를 그대로 받아야 한다"
    assert torch.equal(got["token_weights"], weights)
    assert got["num_items_in_batch"] == 4
