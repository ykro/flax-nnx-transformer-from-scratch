"""Experiment 04: Assembling and Compiling a Full Mini-Transformer in Flax NNX.

Combines Token Embeddings, stacked TransformerDecoderBlocks (RMSNorm + CausalRoPEAttention
+ RMSNorm + SwiGLUMLP), and a language modeling head, trained end-to-end with
`@nnx.jit`, `nnx.value_and_grad`, and `nnx.Optimizer(model, optax.adamw(...), wrt=nnx.Param)`.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def _load_exp03() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp03_causal_attention_rope",
        Path(__file__).resolve().parent / "03_causal_attention_rope.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_exp03 = _load_exp03()
CausalRoPEAttention = _exp03.CausalRoPEAttention


class SwiGLUMLP(nnx.Module):
    """Feed-forward SwiGLU network for a Transformer Decoder block."""

    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs) -> None:
        self.norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.w_gate = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.w_up = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.w_down = nnx.Linear(d_ff, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        normed = self.norm(x)
        activated = jax.nn.silu(self.w_gate(normed)) * self.w_up(normed)
        return x + self.w_down(activated)


class TransformerBlock(nnx.Module):
    """Single pre-RMSNorm Transformer block with Causal RoPE Attention + SwiGLU MLP."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 128,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.attn = CausalRoPEAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rngs=rngs,
        )
        self.mlp = SwiGLUMLP(d_model=d_model, d_ff=d_ff, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x, _ = self.attn(x)
        x = self.mlp(x)
        return x


class MiniTransformerLM(nnx.Module):
    """Decoder-only autoregressive Mini-Transformer in Flax NNX."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 192,
        max_seq_len: int = 64,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embed = nnx.Embed(num_embeddings=vocab_size, features=d_model, rngs=rngs)
        self.blocks = nnx.List(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.lm_head = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        x = self.token_embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def make_synthetic_next_token_batch(
    batch_size: int = 8,
    seq_len: int = 32,
    vocab_size: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Creates a deterministic structured sequence task (repeating algorithmic motifs)."""
    base_pattern = jnp.arange(seq_len + 1, dtype=jnp.int32)
    rows = []
    for b in range(batch_size):
        stride = (b % 3) + 1
        offset = b * 7
        seq = (offset + base_pattern * stride) % vocab_size
        rows.append(seq)
    full = jnp.stack(rows, axis=0)
    return full[:, :-1], full[:, 1:]


@nnx.jit
def train_step(
    model: MiniTransformerLM,
    optimizer: nnx.Optimizer,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Compiled training step mutating model & optimizer state cleanly via NNX."""

    def loss_fn(m: MiniTransformerLM) -> jax.Array:
        logits = m(inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.mean(loss)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def run_experiment(steps: int = 50) -> dict[str, Any]:
    vocab_size = 128
    rngs = nnx.Rngs(params=42)
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=192,
        max_seq_len=64,
        rngs=rngs,
    )
    tx = optax.adamw(learning_rate=3e-3, weight_decay=1e-2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    inputs, targets = make_synthetic_next_token_batch(
        batch_size=8, seq_len=32, vocab_size=vocab_size
    )

    param_leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    total_params = sum(int(p.size) for p in param_leaves)

    loss_arrays: list[jax.Array] = []
    t_first_start = time.perf_counter()
    first_loss_arr = train_step(model, optimizer, inputs, targets)
    jax.block_until_ready(first_loss_arr)
    first_step_ms = (time.perf_counter() - t_first_start) * 1000.0
    loss_arrays.append(first_loss_arr)

    t_steady_start = time.perf_counter()
    for _ in range(1, steps):
        loss_val = train_step(model, optimizer, inputs, targets)
        loss_arrays.append(loss_val)
    jax.block_until_ready(loss_arrays[-1])
    steady_total_ms = (time.perf_counter() - t_steady_start) * 1000.0
    steady_step_ms = steady_total_ms / max(1, (steps - 1))
    losses: list[float] = [float(x) for x in loss_arrays]

    # Check monotonic decrease across 10-step checkpoints
    checkpoint_indices = [0, 9, 19, 29, 39, 49]
    checkpoint_losses = {f"step_{i + 1}": round(losses[i], 5) for i in checkpoint_indices}
    strictly_monotonic_every_step = all(
        losses[i + 1] < losses[i] for i in range(len(losses) - 1)
    )

    results = {
        "total_params": total_params,
        "steps": steps,
        "initial_loss": round(losses[0], 5),
        "final_loss": round(losses[-1], 5),
        "loss_reduction_pct": round((1.0 - losses[-1] / losses[0]) * 100.0, 2),
        "strictly_monotonic_every_step": strictly_monotonic_every_step,
        "checkpoint_losses": checkpoint_losses,
        "first_step_compile_and_run_ms": round(first_step_ms, 2),
        "steady_state_step_ms": round(steady_step_ms, 3),
        "speedup_after_jit": round(first_step_ms / steady_step_ms, 1),
    }

    print("=" * 72)
    print("EXPERIMENT 04: Compiled Mini-Transformer Training with @nnx.jit & Optax")
    print("=" * 72)
    print(f"Total trainable params  : {results['total_params']:,}")
    print(f"Step 1 (Compile + Run)  : {results['first_step_compile_and_run_ms']:.2f} ms")
    print(
        f"Steps 2..{steps} Avg Latency : {results['steady_state_step_ms']:.3f} ms "
        f"({results['speedup_after_jit']}x faster)"
    )
    print(
        f"Loss trajectory         : {results['initial_loss']:.4f} -> "
        f"{results['final_loss']:.4f} (-{results['loss_reduction_pct']}%)"
    )
    print(f"Checkpoints             : {checkpoint_losses}")
    print(f"Strictly monotonic      : {strictly_monotonic_every_step}")
    assert strictly_monotonic_every_step, "Expected strictly monotonic loss decrease!"
    return results


if __name__ == "__main__":
    run_experiment()
