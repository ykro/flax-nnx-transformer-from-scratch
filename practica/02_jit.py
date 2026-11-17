import jax
import jax.numpy as jnp
from flax import nnx


class Contador(nnx.Variable):
    pass


class Capa(nnx.Module):
    def __init__(self, rngs):
        self.lineal = nnx.Linear(4, 4, rngs=rngs)
        self.llamadas = Contador(jnp.array(0))

    def __call__(self, x):
        self.llamadas[...] += 1
        return self.lineal(x)


x = jnp.ones((2, 4))
a = Capa(nnx.Rngs(0))
jax.jit(lambda m, x: m(x))(a, x)
print("con jax.jit:", a.llamadas[...])

b = Capa(nnx.Rngs(0))
nnx.jit(lambda m, x: m(x))(b, x)
print("con nnx.jit:", b.llamadas[...])
