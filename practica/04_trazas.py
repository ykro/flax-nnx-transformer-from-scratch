import jax.numpy as jnp
from flax import nnx

trazas = 0


class Escala(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.ones(4))
        self.factor = 3

    def __call__(self, x):
        return x * self.w[...] * self.factor


@nnx.jit
def forward(m, x):
    global trazas
    trazas += 1
    return m(x)


m = Escala()
forward(m, jnp.ones((2, 4)))
forward(m, jnp.ones((2, 4)))
print("misma forma, dos llamadas:", trazas)
forward(m, jnp.ones((3, 4)))
print("otra forma:", trazas)
m.factor = 4
forward(m, jnp.ones((2, 4)))
print("otro entero:", trazas)
