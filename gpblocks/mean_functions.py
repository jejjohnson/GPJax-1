import jax.numpy as jnp


class MeanFunction:
    def __init__(self, name=None):
        self.name = name 

    def __call__(self, X):
        return self.compute_mu(X)

    def compute_mu(self, X):
        raise NotImplementedError


class Zero(MeanFunction):
    def __init__(self, name="ZeroMean"):
        self.name = name 
    
    def compute_mu(self, X):
        return jnp.zeros_like(X)