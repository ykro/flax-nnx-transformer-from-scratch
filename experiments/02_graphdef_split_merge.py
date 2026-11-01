"""Experiment 02: Demystifying the GraphDef + State Split/Merge Boundary.

Shows how `nnx.split(model)` decomposes a stateful Python object into a static
`GraphDef` and a pure JAX PyTree `State` (which can be partitioned by variable
type), how pure XLA kernels transform `State` functionally, and how
`nnx.merge(graphdef, state)` or `nnx.update(model, state)` restores mutations.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class LoRALinear(nnx.Module):
    """Linear layer augmented with a low-rank adapter using distinct Variable types."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.base = nnx.Linear(in_features, out_features, use_bias=False, rngs=rngs)
        self.lora_a = nnx.LoRAParam(
            jax.random.normal(rngs.params(), (in_features, rank)) * 0.02
        )
        self.lora_b = nnx.LoRAParam(jnp.zeros((rank, out_features), dtype=jnp.float32))
        self.ActivationStat = nnx.BatchStat(jnp.zeros((out_features,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x) + (x @ self.lora_a[...]) @ self.lora_b[...]
        self.ActivationStat[...] = jnp.mean(jnp.abs(y), axis=(0, 1))
        return y


def run_experiment() -> dict[str, Any]:
    rngs = nnx.Rngs(params=101)
    model = LoRALinear(in_features=64, out_features=64, rank=8, rngs=rngs)

    # 1. Split into GraphDef + full State, or partition by Variable filter
    graphdef, full_state = nnx.split(model)
    _, lora_params, base_params, batch_stats = nnx.split(
        model, nnx.LoRAParam, nnx.Param, nnx.BatchStat
    )

    lora_scalar_count = sum(int(x.size) for x in jax.tree_util.tree_leaves(lora_params))
    base_scalar_count = sum(int(x.size) for x in jax.tree_util.tree_leaves(base_params))
    stat_scalar_count = sum(int(x.size) for x in jax.tree_util.tree_leaves(batch_stats))

    # 2. Pure JAX jitted function operating on (GraphDef, State) PyTrees
    @jax.jit
    def pure_xla_step(
        graph_def: nnx.GraphDef[LoRALinear],
        state: nnx.State,
        x: jax.Array,
    ) -> tuple[jax.Array, nnx.State]:
        transient_model = nnx.merge(graph_def, state)
        out = transient_model(x)
        _, updated_state = nnx.split(transient_model)
        return out, updated_state

    x = jax.random.normal(jax.random.key(9), (2, 16, 64))
    out_before, new_state = pure_xla_step(graphdef, full_state, x)

    # Verify round-trip weight identity before mutation
    reconstructed = nnx.merge(graphdef, full_state)
    weight_diff = float(
        jnp.max(jnp.abs(reconstructed.base.kernel[...] - model.base.kernel[...]))
    )

    # Propagate functional state update back to the live Python object
    nnx.update(model, new_state)
    activation_mean_norm = float(jnp.linalg.norm(model.ActivationStat[...]))

    # Verify that mutating LoRAParam functionally leaves base Param untouched
    mutated_lora = jax.tree_util.tree_map(lambda w: w + 0.5, lora_params)
    merged_after_lora = nnx.merge(graphdef, mutated_lora, base_params, batch_stats)
    base_unchanged_diff = float(
        jnp.max(jnp.abs(merged_after_lora.base.kernel[...] - model.base.kernel[...]))
    )
    lora_delta = float(
        jnp.max(jnp.abs(merged_after_lora.lora_b[...] - model.lora_b[...]))
    )

    results = {
        "base_param_scalars": base_scalar_count,
        "lora_param_scalars": lora_scalar_count,
        "batch_stat_scalars": stat_scalar_count,
        "total_leaves": len(jax.tree_util.tree_leaves(full_state)),
        "split_merge_weight_max_diff": weight_diff,
        "base_unchanged_after_lora_edit_diff": base_unchanged_diff,
        "lora_b_applied_delta": lora_delta,
        "updated_activation_stat_l2": round(activation_mean_norm, 6),
        "output_shape": list(np.asarray(out_before).shape),
    }

    print("=" * 72)
    print("EXPERIMENT 02: GraphDef & State Split/Merge Boundary for XLA")
    print("=" * 72)
    print(f"State PyTree leaves     : {results['total_leaves']}")
    print(f"Base nnx.Param scalars  : {results['base_param_scalars']:,}")
    print(f"LoRA nnx.LoRAParam      : {results['lora_param_scalars']:,}")
    print(f"nnx.BatchStat scalars   : {results['batch_stat_scalars']:,}")
    print(f"Round-trip weight diff  : {results['split_merge_weight_max_diff']:.3e}")
    print(
        f"Selective LoRA delta    : {results['lora_b_applied_delta']:.3f} "
        f"(base diff={base_unchanged_diff:.1e})"
    )
    print(f"Updated BatchStat L2    : {results['updated_activation_stat_l2']:.6f}")
    assert weight_diff == 0.0
    assert base_unchanged_diff == 0.0
    return results


if __name__ == "__main__":
    run_experiment()
