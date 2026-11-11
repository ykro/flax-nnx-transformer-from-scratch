"""Experiment 05: Tracking Recompilations and Inspecting StableHLO & XLA Fusion.

Proves that a stateful Flax NNX training step traces and compiles exactly ONCE
across 50 training steps when tensor shapes remain static, and inspects the
lowered StableHLO IR and compiled HLO cost/memory analysis.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def _load_exp04() -> Any:
    spec = importlib.util.spec_from_file_location(
        "exp04_compiled_train_step",
        Path(__file__).resolve().parent / "04_compiled_train_step.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_exp04 = _load_exp04()
MiniTransformerLM = _exp04.MiniTransformerLM
make_synthetic_next_token_batch = _exp04.make_synthetic_next_token_batch


class _ScaledByInt(nnx.Module):
    """A plain Python int attribute is static: it is part of the GraphDef."""

    def __init__(self) -> None:
        self.factor = 3
        self.w = nnx.Param(jnp.ones((4,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.w[...] * self.factor


def python_int_retrace_demo() -> int:
    traces = {"count": 0}

    @nnx.jit
    def forward(mod: _ScaledByInt, x: jax.Array) -> jax.Array:
        traces["count"] += 1
        return mod(x)

    mod = _ScaledByInt()
    x = jnp.ones((4,), dtype=jnp.float32)
    forward(mod, x)
    forward(mod, x)
    mod.factor = 4
    forward(mod, x)
    return traces["count"]


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
    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=3e-3, weight_decay=1e-2), wrt=nnx.Param
    )
    inputs, targets = make_synthetic_next_token_batch(
        batch_size=8, seq_len=32, vocab_size=vocab_size
    )

    trace_counter = {"count": 0}

    # Explicit split/merge inside @jax.jit to inspect StableHLO directly and count Python traces
    @jax.jit
    def pure_train_kernel(
        model_graphdef: nnx.GraphDef[MiniTransformerLM],
        model_state: nnx.State,
        opt_graphdef: nnx.GraphDef[nnx.Optimizer],
        opt_state: nnx.State,
        x: jax.Array,
        y: jax.Array,
    ) -> tuple[jax.Array, nnx.State, nnx.State]:
        trace_counter["count"] += 1
        m = nnx.merge(model_graphdef, model_state)
        opt = nnx.merge(opt_graphdef, opt_state)

        def loss_fn(mod: MiniTransformerLM) -> jax.Array:
            logits = mod(x)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, y)
            )

        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        _, new_m_state = nnx.split(m)
        _, new_opt_state = nnx.split(opt)
        return loss, new_m_state, new_opt_state

    m_graphdef, m_state = nnx.split(model)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # Lower to StableHLO before execution
    lowered = pure_train_kernel.lower(
        m_graphdef, m_state, opt_graphdef, opt_state, inputs, targets
    )
    stablehlo_text = lowered.as_text("stablehlo")
    traces_at_lower = trace_counter["count"]
    compiled = lowered.compile()
    hlo_text = compiled.as_text()

    # Execute 50 steps and track Python trace invocations
    traces_after_step_1 = 0
    for step_idx in range(steps):
        loss, m_state, opt_state = pure_train_kernel(
            m_graphdef, m_state, opt_graphdef, opt_state, inputs, targets
        )
        if step_idx == 0:
            traces_after_step_1 = trace_counter["count"]
    jax.block_until_ready(loss)

    nnx.update(model, m_state)
    nnx.update(optimizer, opt_state)

    total_traces = trace_counter["count"]
    recompiles_after_warmup = total_traces - traces_after_step_1

    stablehlo_lines = stablehlo_text.splitlines()
    dot_general_ops = sum(1 for line in stablehlo_lines if "stablehlo.dot_general" in line)
    reduce_ops = sum(1 for line in stablehlo_lines if "stablehlo.reduce" in line)
    fusion_count = sum(1 for line in hlo_text.splitlines() if "fusion" in line.lower())

    results = {
        "steps_executed": steps,
        "traces_at_lower": traces_at_lower,
        "traces_on_step_1": traces_after_step_1,
        "total_traces_after_50_steps": total_traces,
        "recompilations_after_step_1": recompiles_after_warmup,
        "stablehlo_line_count": len(stablehlo_lines),
        "stablehlo_dot_general_count": dot_general_ops,
        "stablehlo_reduce_count": reduce_ops,
        "compiled_hlo_fusion_mentions": fusion_count,
        "final_loss": round(float(loss), 5),
        "python_int_attr_change_traces": python_int_retrace_demo(),
    }

    print("=" * 72)
    print("EXPERIMENT 05: StableHLO Inspection & Zero-Recompilation Audit")
    print("=" * 72)
    print(f"Traces at .lower()      : {results['traces_at_lower']}")
    print(f"Total traces (50 steps) : {results['total_traces_after_50_steps']}")
    print(f"Retraces after .lower() : {results['recompilations_after_step_1']}")
    print(f"StableHLO IR lines      : {results['stablehlo_line_count']}")
    print(f"stablehlo.dot_general   : {results['stablehlo_dot_general_count']}")
    print(f"stablehlo.reduce        : {results['stablehlo_reduce_count']}")
    print(f"HLO lines with fusion   : {results['compiled_hlo_fusion_mentions']}")
    print(
        f"Python int attr change  : {results['python_int_attr_change_traces']} traces "
        "(int lives in the GraphDef)"
    )
    assert recompiles_after_warmup == 0, "Expected 0 recompilations after warmup!"
    assert traces_at_lower == 1
    assert results["python_int_attr_change_traces"] == 2
    return results


if __name__ == "__main__":
    run_experiment()
