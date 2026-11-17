import jax
import jax.numpy as jnp
import optax
from flax import nnx
from minilm import MiniLM, T, V


@nnx.jit
def paso(modelo, opt, ids):
    def perdida(m):
        logits = m(ids[:, :-1])
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]).mean()

    loss, grads = nnx.value_and_grad(perdida)(modelo)
    opt.update(modelo, grads)
    return loss


print(f"al azar: loss {jnp.log(V):.3f}")
for mascara in [True, False]:
    modelo = MiniLM(mascara, nnx.Rngs(0))
    opt = nnx.Optimizer(modelo, optax.adam(1e-2), wrt=nnx.Param)
    for i in range(300):
        ids = jax.random.randint(jax.random.key(i), (64, T + 1), 0, V)
        loss = paso(modelo, opt, ids)
    print(f"máscara={mascara}: loss {loss:.3f}")
