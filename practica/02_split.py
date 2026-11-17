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


capa = Capa(nnx.Rngs(0))
graphdef, pesos, contador = nnx.split(capa, nnx.Param, Contador)
print("graphdef:", len(jax.tree.leaves(graphdef)), "arreglos")
for ruta, v in nnx.to_flat_state(pesos):
    print("pesos:", ".".join(ruta), v.shape)
for ruta, v in nnx.to_flat_state(contador):
    print("contador:", ".".join(ruta), v.shape)

copia = nnx.merge(graphdef, pesos, contador)
iguales = copia.lineal.kernel[...] == capa.lineal.kernel[...]
print("copia igual:", bool(iguales.all()))
