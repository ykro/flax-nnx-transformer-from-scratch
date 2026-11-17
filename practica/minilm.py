import jax
import jax.numpy as jnp
from flax import nnx

V, T, D = 16, 12, 32


class MiniLM(nnx.Module):
    def __init__(self, mascara, rngs):
        self.tok = nnx.Embed(V, D, rngs=rngs)
        self.pos = nnx.Embed(T, D, rngs=rngs)
        self.q = nnx.Linear(D, D, rngs=rngs)
        self.k = nnx.Linear(D, D, rngs=rngs)
        self.v = nnx.Linear(D, D, rngs=rngs)
        self.salida = nnx.Linear(D, V, rngs=rngs)
        self.mascara = mascara

    def __call__(self, ids):
        h = self.tok(ids) + self.pos(jnp.arange(T))
        s = self.q(h) @ self.k(h).swapaxes(-1, -2) / jnp.sqrt(D)
        if self.mascara:
            s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -1e9)
        return self.salida(jax.nn.softmax(s) @ self.v(h))
