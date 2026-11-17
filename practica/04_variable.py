import jax.numpy as jnp
from flax import nnx

trazas = 0


class Escala(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.ones(4))
        self.factor = nnx.Variable(jnp.array(3.0))

    def __call__(self, x):
        return x * self.w[...] * self.factor[...]


@nnx.jit
def forward(m, x):
    global trazas
    trazas += 1
    return m(x)


m = Escala()
print(forward(m, jnp.ones((2, 4)))[0])
m.factor[...] = 4.0
print(forward(m, jnp.ones((2, 4)))[0])
print("trazas:", trazas)
