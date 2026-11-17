import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

capa = nnx.Linear(64, 192, use_bias=False, rngs=nnx.Rngs(0))
ref = torch.nn.Linear(64, 192, bias=False)
print("kernel de nnx:  ", capa.kernel.shape)
print("weight de torch:", tuple(ref.weight.shape))

w = np.array(capa.kernel[...])
with torch.no_grad():
    ref.weight.copy_(torch.from_numpy(w.T))

x = np.random.default_rng(0).standard_normal((4, 64), dtype=np.float32)
a = np.asarray(capa(jnp.asarray(x)))
b = ref(torch.from_numpy(x)).detach().numpy()
print("diferencia máxima:", np.abs(a - b).max())
