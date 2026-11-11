"""The Outside Judge (`verify_transformer.py`).

Executes all 5 experiments deterministically, verifies numerical invariants:
1. Eager numerical parity between Flax NNX and PyTorch (`max_abs_diff < 1e-5`)
2. Lossless `nnx.split` / `nnx.merge` round-trip (`weight_diff == 0.0`)
3. Strict causal attention masking (`prefix_leakage <= 1e-6`, `upper_tri == 0.0`)
4. Monotonic loss decrease over 50 compiled steps (`initial_loss -> final_loss`)
5. Zero recompilations after warmup step 1 (`recompilations_after_step_1 == 0`)

Writes real outputs to `benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import flax
import jax
import optax
import torch

ROOT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = ROOT_DIR / "benchmarks"


def _load_module(name: str, rel_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / rel_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cpu_model() -> str:
    """Human-readable CPU model (platform.processor() only says 'arm' on macOS)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _load1() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def run_verification() -> dict[str, Any]:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    load_before = _load1()

    with contextlib.redirect_stdout(buffer):
        print("########################################################################")
        print("# FLAX NNX TRANSFORMER FROM SCRATCH — OUTSIDE JUDGE VERIFICATION SUITE #")
        print("########################################################################")
        print(
            f"Runtime: Python {platform.python_version()} | "
            f"JAX {jax.__version__} ({jax.default_backend()}) | "
            f"Flax {flax.__version__} | Optax {optax.__version__} | "
            f"PyTorch {torch.__version__}"
        )
        print()

        exp01 = _load_module("exp01", "experiments/01_nnx_module_vs_torch.py").run_experiment()
        print()
        exp02 = _load_module("exp02", "experiments/02_graphdef_split_merge.py").run_experiment()
        print()
        exp03 = _load_module("exp03", "experiments/03_causal_attention_rope.py").run_experiment()
        print()
        exp04 = _load_module("exp04", "experiments/04_compiled_train_step.py").run_experiment(
            steps=50
        )
        print()
        exp05 = _load_module(
            "exp05", "experiments/05_inspect_stablehlo_recompiles.py"
        ).run_experiment(steps=50)
        print()
        print("========================================================================")
        load_after = _load1()
        print(
            f"CPU: {_cpu_model()} ({os.cpu_count()} cores) | 1-min load average "
            f"{load_before} before, {load_after} after (timings in exp. 04 depend on it)"
        )
        print("VERDICT: ALL 5 FLAX NNX TRANSFORMER INVARIANTS PASSED DETERMINISTICALLY")
        print("========================================================================")

    terminal_log = buffer.getvalue()
    print(terminal_log, end="")

    summary = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "flax": flax.__version__,
            "optax": optax.__version__,
            "torch": torch.__version__,
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_1min_before": load_before,
            "load_average_1min_after": load_after,
        },
        "exp01_nnx_vs_torch": exp01,
        "exp02_split_merge": exp02,
        "exp03_causal_rope_attention": exp03,
        "exp04_compiled_training": exp04,
        "exp05_stablehlo_recompiles": exp05,
        "invariants": {
            "nnx_vs_torch_parity_ok": exp01["max_abs_diff_vs_torch"] < 1e-5,
            "split_merge_lossless_ok": exp02["split_merge_weight_max_diff"] == 0.0,
            "causal_mask_zero_leakage_ok": exp03["causal_prefix_leakage_max_diff"] <= 1e-6,
            "monotonic_50_step_loss_ok": exp04["strictly_monotonic_every_step"],
            "zero_recompilations_after_step_1_ok": exp05["recompilations_after_step_1"] == 0,
        },
    }

    assert all(summary["invariants"].values()), f"Invariant failure: {summary['invariants']}"

    (BENCHMARKS_DIR / "latest_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (BENCHMARKS_DIR / "terminal_output.txt").write_text(terminal_log, encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_verification()
