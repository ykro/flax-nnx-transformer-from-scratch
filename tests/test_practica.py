"""The snippets in practica/ are the ones typed in the practice video: each must run and print
what the video shows (01_copiar_roto.py must fail with the error the video explains)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PRACTICA = Path(__file__).resolve().parent.parent / "practica"


def run(name: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(PRACTICA / name)]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)


def test_01_modulo_reads_like_pytorch() -> None:
    lines = run("01_modulo.py").stdout.splitlines()
    assert lines == ["kernel de gate: (64, 192)", "llamadas: 1", "parámetros: 36864"]


def test_01_copiar_roto_fails_without_transpose() -> None:
    r = run("01_copiar_roto.py")
    assert r.returncode != 0
    assert r.stdout.splitlines() == ["kernel de nnx:   (64, 192)", "weight de torch: (192, 64)"]
    err = "RuntimeError: The size of tensor a (64) must match the size of tensor b (192)"
    assert err in r.stderr


def test_01_copiar_matches_pytorch() -> None:
    lines = run("01_copiar.py").stdout.splitlines()
    assert lines[:2] == ["kernel de nnx:   (64, 192)", "weight de torch: (192, 64)"]
    assert lines[2].startswith("diferencia máxima:")
    assert float(lines[2].split(":")[1]) < 1e-5


def test_02_split_graphdef_has_no_arrays() -> None:
    lines = run("02_split.py").stdout.splitlines()
    assert lines == [
        "graphdef: 0 arreglos",
        "pesos: lineal.bias (4,)",
        "pesos: lineal.kernel (4, 4)",
        "contador: llamadas ()",
        "copia igual: True",
    ]


def test_02_jit_only_nnx_jit_keeps_the_mutation() -> None:
    lines = run("02_jit.py").stdout.splitlines()
    assert lines == ["con jax.jit: 0", "con nnx.jit: 1"]


def test_03_fuga_without_mask_the_loss_collapses() -> None:
    lines = run("03_fuga.py").stdout.splitlines()
    assert lines[0] == "al azar: loss 2.773"
    con = float(lines[1].removeprefix("máscara=True: loss "))
    sin = float(lines[2].removeprefix("máscara=False: loss "))
    # Random tokens: with the mask nothing beats chance (log 16);
    # without it the model copies the next token from the future
    assert abs(con - 2.773) < 0.05
    assert sin < 0.5


def test_04_trazas_shape_and_python_int_retrace() -> None:
    lines = run("04_trazas.py").stdout.splitlines()
    assert lines == ["misma forma, dos llamadas: 1", "otra forma: 2", "otro entero: 3"]


def test_04_variable_changes_value_without_retrace() -> None:
    lines = run("04_variable.py").stdout.splitlines()
    assert lines == ["[3. 3. 3. 3.]", "[4. 4. 4. 4.]", "trazas: 1"]
