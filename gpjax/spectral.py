import jax.numpy as jnp
import jax.random as jr
from .kernel import Stationary
from .parameters import Parameter
from .transforms import Identity
from typing import Optional


class SpectralKernel(Stationary):
    def __init__(self,
                 num_basis: int,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 key=jr.PRNGKey(123),
                 name: str = "Spectral Kernel",
                 ):
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)
        # TODO: This assumes the lengthscale is ARD. This value should be driven by the data's dimension instead.
        self.input_dim = lengthscale.shape[0]
        self.num_basis = num_basis
        self.features = Parameter(self.spectral_density(key,
                                            shape=(self.num_basis,
                                                   self.input_dim)),
                                  transform=Identity())

    @property
    def spectral_density(self):
        """
        Return the kernel's corresponding spectral density.
        """
        raise NotImplementedError

    def scale_frequencies(self) -> jnp.array:
        r"""
        For a set of frequencies, scale by the kernel's lengthscale value(s). In the literature, this quantity is often
        referred to as :math:`\omega`.

        Returns: The original frequencies, now scaled by the kernel's lengthscale.
        """
        return self.features.untransform / self.lengthscale.untransform

    @staticmethod
    def compute_gram(phi):
        """
        For the truncated frequency matrix :
        Args:
            phi:

        Returns:

        """
        gram = jnp.matmul(phi, phi.T)
        return gram


class SpectralRBF(SpectralKernel):
    """
    Random Fourier feature approximation to the RBF kernel.
    """
    def __init__(self,
                 num_basis: int,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 key=jr.PRNGKey(123),
                 name: str = "RBF"):
        super().__init__(num_basis=num_basis, lengthscale=lengthscale, variance=variance, name=name)
        self.input_dim = lengthscale.shape[
            0]  # TODO: This assumes the lengthscale is ARD. This value should be driven by the data's dimension instead.
        self.num_basis = num_basis
        self.features = Parameter(self.spectral_density(
            key, shape=(self.num_basis, self.input_dim)),
                                  transform=Identity())
        self.spectral = True

    @property
    def spectral_density(self):
        return jr.normal

    def _compute_phi(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Takes an NxD matrix and returns an N*2M matrix. This quantity is often called the design matrix and denoted by :math:`\phi`.

        :param X:
        :return:
        """
        omega = self.scale_frequencies()
        cos_freqs = jnp.cos(X.dot(
            omega.T))  # TODO: Can possible do away with the tranpose
        sin_freqs = jnp.sin(X.dot(omega.T))
        phi = jnp.hstack((cos_freqs, sin_freqs))
        assert phi.shape == (X.shape[0], self.num_basis *
                             2), "Phi matrix incorrectly computed."
        return phi

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
        phi = self._compute_phi(X)
        return (self.variance.untransform/self.num_basis)*self.compute_gram(phi)
