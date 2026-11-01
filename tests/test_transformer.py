"""Automated pytest suite for Flax NNX Transformer from Scratch."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load(name: str, rel_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exp01_nnx_module_vs_torch_parity() -> None:
    res = _load("exp01", "experiments/01_nnx_module_vs_torch.py").run_experiment()
    assert res["max_abs_diff_vs_torch"] < 1e-5
    assert res["forward_calls_count"] == 1
    assert res["total_params"] == 36928


def test_exp02_graphdef_split_merge_roundtrip() -> None:
    res = _load("exp02", "experiments/02_graphdef_split_merge.py").run_experiment()
    assert res["split_merge_weight_max_diff"] == 0.0
    assert res["base_unchanged_after_lora_edit_diff"] == 0.0
    assert res["lora_b_applied_delta"] == 0.5


def test_exp03_causal_rope_attention_invariance() -> None:
    res = _load("exp03", "experiments/03_causal_attention_rope.py").run_experiment()
    assert res["causal_prefix_leakage_max_diff"] <= 1e-6
    assert res["upper_triangle_attention_mass"] == 0.0
    assert res["rope_l2_norm_preservation_diff"] <= 1e-5


def test_exp04_compiled_train_step_monotonic_convergence() -> None:
    res = _load("exp04", "experiments/04_compiled_train_step.py").run_experiment(steps=50)
    assert res["strictly_monotonic_every_step"] is True
    assert res["final_loss"] < res["initial_loss"] * 0.05


def test_exp05_zero_recompilations_and_stablehlo() -> None:
    res = _load("exp05", "experiments/05_inspect_stablehlo_recompiles.py").run_experiment(
        steps=50
    )
    assert res["traces_on_step_1"] == 1
    assert res["total_traces_after_50_steps"] == 1
    assert res["recompilations_after_step_1"] == 0
    assert res["stablehlo_dot_general_count"] > 10
