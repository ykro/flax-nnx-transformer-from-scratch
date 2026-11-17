import jax
import jax.numpy as jnp
from flax import nnx


class Contador(nnx.Variable):
    pass


class SwiGLU(nnx.Module):
    def __init__(self, d, d_ff, *, rngs):
        self.gate = nnx.Linear(d, d_ff, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(d, d_ff, use_bias=False, rngs=rngs)
        self.down = nnx.Linear(d_ff, d, use_bias=False, rngs=rngs)
        self.llamadas = Contador(jnp.array(0))

    def __call__(self, x):
        self.llamadas[...] += 1
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


mlp = SwiGLU(64, 192, rngs=nnx.Rngs(0))
mlp(jnp.ones((4, 64)))
print("kernel de gate:", mlp.gate.kernel.shape)
print("llamadas:", mlp.llamadas[...])
pesos = nnx.state(mlp, nnx.Param)
print("parámetros:", sum(p.size for p in jax.tree.leaves(pesos)))
