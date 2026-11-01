"""Experiment 03: Multi-Head Causal Self-Attention with RoPE and RMSNorm in Flax NNX.

Implements Rotary Position Embeddings (RoPE), RMSNorm pre-normalization, and
causal masking from scratch in `Flax NNX`. Verifies strict causal invariance:
perturbing future tokens at positions t > k produces 0.0 difference in outputs
at positions <= k.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def precompute_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[jax.Array, jax.Array]:
    """Precomputes cosine and sine tables for Rotary Position Embeddings."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE."
    half_dim = head_dim // 2
    freq_seq = jnp.arange(0, half_dim, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / half_dim))
    positions = jnp.arange(0, max_seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, inv_freq)  # (seq_len, half_dim)
    cos = jnp.cos(angles)[None, :, None, :]  # (1, seq_len, 1, half_dim)
    sin = jnp.sin(angles)[None, :, None, :]
    return cos, sin


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Applies rotary position embeddings to query or key tensor (B, T, H, D)."""
    seq_len = x.shape[1]
    cos_t = cos[:, :seq_len, :, :]
    sin_t = sin[:, :seq_len, :, :]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated_first = x1 * cos_t - x2 * sin_t
    rotated_second = x2 * cos_t + x1 * sin_t
    return jnp.concatenate([rotated_first, rotated_second], axis=-1)


class RoPECache(nnx.Variable):
    """Non-trainable static buffer variable so Optax only differentiates nnx.Param."""


class CausalRoPEAttention(nnx.Module):
    """Multi-Head Causal Self-Attention with Rotary Position Embeddings in Flax NNX."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 128,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.q_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

        cos, sin = precompute_rope_cache(max_seq_len, self.head_dim)
        self.rope_cos = RoPECache(cos)
        self.rope_sin = RoPECache(sin)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        bsz, seq_len, _ = x.shape
        normed = self.norm(x)

        q = self.q_proj(normed).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(normed).reshape(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(normed).reshape(bsz, seq_len, self.num_heads, self.head_dim)

        q = apply_rope(q, self.rope_cos[...], self.rope_sin[...])
        k = apply_rope(k, self.rope_cos[...], self.rope_sin[...])

        scale = 1.0 / jnp.sqrt(jnp.float32(self.head_dim))
        scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale

        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[
            None, None, :, :
        ]
        masked_scores = jnp.where(causal_mask, scores, jnp.float32(-1e9))
        attn_weights = jax.nn.softmax(masked_scores, axis=-1)

        context = jnp.einsum("bhts,bshd->bthd", attn_weights, v).reshape(
            bsz, seq_len, self.d_model
        )
        return x + self.out_proj(context), attn_weights


def run_experiment() -> dict[str, Any]:
    rngs = nnx.Rngs(params=2026)
    attn_block = CausalRoPEAttention(d_model=64, num_heads=4, max_seq_len=64, rngs=rngs)

    x_clean = jax.random.normal(jax.random.key(11), (2, 16, 64))
    # Perturb tokens strictly in the future (indices 8..15)
    perturbation = jnp.zeros_like(x_clean).at[:, 8:, :].set(25.0)
    x_perturbed_future = x_clean + perturbation

    out_clean, weights_clean = attn_block(x_clean)
    out_perturbed, _ = attn_block(x_perturbed_future)

    # Prefix positions 0..7 must be causally invariant to tokens 8..15
    prefix_max_diff = float(jnp.max(jnp.abs(out_clean[:, :8, :] - out_perturbed[:, :8, :])))
    future_max_diff = float(jnp.max(jnp.abs(out_clean[:, 8:, :] - out_perturbed[:, 8:, :])))

    # Upper triangle of attention matrix above main diagonal must be strictly 0
    upper_tri_mask = jnp.triu(jnp.ones((16, 16), dtype=jnp.bool_), k=1)
    upper_tri_mass = float(jnp.max(jnp.where(upper_tri_mask, weights_clean, 0.0)))

    # RoPE norm preservation check: rotation is an orthogonal transform
    sample_q = jax.random.normal(jax.random.key(19), (2, 16, 4, 16))
    rotated_q = apply_rope(sample_q, attn_block.rope_cos[...], attn_block.rope_sin[...])
    rope_norm_diff = float(
        jnp.max(jnp.abs(jnp.linalg.norm(sample_q, axis=-1) - jnp.linalg.norm(rotated_q, axis=-1)))
    )

    results = {
        "d_model": 64,
        "num_heads": 4,
        "head_dim": 16,
        "seq_len": 16,
        "causal_prefix_leakage_max_diff": prefix_max_diff,
        "future_perturbation_response_max_diff": round(future_max_diff, 4),
        "upper_triangle_attention_mass": upper_tri_mass,
        "rope_l2_norm_preservation_diff": rope_norm_diff,
        "output_shape": list(np.asarray(out_clean).shape),
    }

    print("=" * 72)
    print("EXPERIMENT 03: Multi-Head Causal Self-Attention with RoPE & RMSNorm")
    print("=" * 72)
    print(f"Output tensor shape     : {results['output_shape']}")
    print(
        f"Causal prefix leakage   : {results['causal_prefix_leakage_max_diff']:.3e} "
        "(must be <= 1e-6)"
    )
    print(f"Future token response   : {results['future_perturbation_response_max_diff']:.4f}")
    print(f"Upper-tri attention max : {results['upper_triangle_attention_mass']:.3e}")
    print(f"RoPE L2 norm drift      : {results['rope_l2_norm_preservation_diff']:.3e}")
    assert prefix_max_diff <= 1e-6
    assert upper_tri_mass == 0.0
    assert rope_norm_diff <= 1e-5
    return results


if __name__ == "__main__":
    run_experiment()
