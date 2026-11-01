"""Experiment 01: Stateful Layers in Flax NNX vs PyTorch nn.Module.

Demonstrates how Flax NNX eliminates the stateless init/apply ceremony of legacy
Flax Linen, offering direct Python attribute inspection, explicit PRNG streams via
nnx.Rngs, and typed state containers (nnx.Param, nnx.BatchStat) alongside a
side-by-side comparison with torch.nn.Module.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from flax import nnx


class TorchSwiGLUMLP(nn.Module):
    """PyTorch reference SwiGLU MLP block."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        gate = nn.functional.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class CallCounter(nnx.Variable):
    """Custom typed state variable in Flax NNX to track forward invocations."""


class NNXSwiGLUMLP(nnx.Module):
    """Flax NNX stateful SwiGLU MLP block with typed Variables and explicit Rngs."""

    def __init__(self, d_model: int, d_ff: int, *, rngs: nnx.Rngs) -> None:
        self.w_gate = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.w_up = nnx.Linear(d_model, d_ff, use_bias=False, rngs=rngs)
        self.w_down = nnx.Linear(d_ff, d_model, use_bias=False, rngs=rngs)
        self.scale = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
        self.forward_calls = CallCounter(jnp.array(0, dtype=jnp.int32))

    def __call__(self, x: jax.Array) -> jax.Array:
        self.forward_calls[...] = self.forward_calls[...] + 1
        gate = jax.nn.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up) * self.scale[...]


def run_experiment(
    batch_size: int = 4,
    seq_len: int = 32,
    d_model: int = 64,
    d_ff: int = 192,
) -> dict[str, Any]:
    torch.manual_seed(42)
    torch_mlp = TorchSwiGLUMLP(d_model=d_model, d_ff=d_ff)

    rngs = nnx.Rngs(params=42)
    nnx_mlp = NNXSwiGLUMLP(d_model=d_model, d_ff=d_ff, rngs=rngs)

    # Copy exact weights from NNX into PyTorch (transposed for nn.Linear layout)
    with torch.no_grad():
        torch_mlp.w_gate.weight.copy_(
            torch.from_numpy(np.asarray(nnx_mlp.w_gate.kernel[...]).T.copy())
        )
        torch_mlp.w_up.weight.copy_(
            torch.from_numpy(np.asarray(nnx_mlp.w_up.kernel[...]).T.copy())
        )
        torch_mlp.w_down.weight.copy_(
            torch.from_numpy(np.asarray(nnx_mlp.w_down.kernel[...]).T.copy())
        )

    x_np = np.random.default_rng(7).standard_normal((batch_size, seq_len, d_model)).astype(
        np.float32
    )
    x_torch = torch.from_numpy(x_np)
    x_jax = jnp.asarray(x_np)

    t0 = time.perf_counter()
    out_torch = torch_mlp(x_torch).detach().numpy()
    torch_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    out_nnx = np.asarray(nnx_mlp(x_jax))
    nnx_eager_ms = (time.perf_counter() - t1) * 1000.0

    max_abs_diff = float(np.max(np.abs(out_torch - out_nnx)))

    # Inspect parameters filtered by type
    params_state = nnx.state(nnx_mlp, nnx.Param)
    counters_state = nnx.state(nnx_mlp, CallCounter)
    param_leaves = jax.tree_util.tree_leaves(params_state)
    total_params = sum(int(p.size) for p in param_leaves)

    results = {
        "d_model": d_model,
        "d_ff": d_ff,
        "total_params": total_params,
        "w_gate_shape": list(nnx_mlp.w_gate.kernel[...].shape),
        "w_up_shape": list(nnx_mlp.w_up.kernel[...].shape),
        "w_down_shape": list(nnx_mlp.w_down.kernel[...].shape),
        "scale_shape": list(nnx_mlp.scale[...].shape),
        "forward_calls_count": int(nnx_mlp.forward_calls[...]),
        "counter_leaves": len(jax.tree_util.tree_leaves(counters_state)),
        "max_abs_diff_vs_torch": max_abs_diff,
        "torch_eager_ms": round(torch_ms, 3),
        "nnx_eager_ms": round(nnx_eager_ms, 3),
    }

    print("=" * 72)
    print("EXPERIMENT 01: Flax NNX Module vs PyTorch nn.Module Ergonomics")
    print("=" * 72)
    print(f"Direct attribute access : nnx_mlp.w_gate.kernel.shape = {results['w_gate_shape']}")
    print(f"Direct attribute access : nnx_mlp.w_down.kernel.shape = {results['w_down_shape']}")
    print(f"Direct Param attribute  : nnx_mlp.scale.shape         = {results['scale_shape']}")
    print(f"Total nnx.Param scalars : {results['total_params']:,}")
    print(f"Typed CallCounter state : {results['forward_calls_count']} call(s) recorded in-place")
    print(f"Max |NNX - PyTorch|     : {results['max_abs_diff_vs_torch']:.3e}")
    assert max_abs_diff < 1e-5, f"Numerical parity failed: {max_abs_diff}"
    return results


if __name__ == "__main__":
    run_experiment()
