from __future__ import annotations
import typing
import jax.numpy as jnp
from .posteriors import Posterior
from ..likelihoods import Gaussian, Likelihood
from ..parameters import Parameter


if typing.TYPE_CHECKING:
    from .priors import DTCPrior


class DTCPosterior(Posterior):
    def __init__(self, prior: DTCPrior, likelihood: Gaussian):
        super().__init__(prior, likelihood)
        self.inducing_variables = prior.inducing_variables

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        pass