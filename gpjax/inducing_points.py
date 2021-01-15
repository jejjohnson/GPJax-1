import jax.numpy as jnp
from objax import Module


class InducingVariables(Module):
    def __init__(self,
                 inputs: jnp.ndarray,
                 n_inducing: int,
                 name: str = "Inducing variables"):
        self.n_inducing = n_inducing
        self.name = name
        self.Z = None
