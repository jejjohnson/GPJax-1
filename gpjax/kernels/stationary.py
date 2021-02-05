import jax.numpy as jnp
from objax import Module
from typing import Callable, Optional
from jax import vmap
from .parameters import Parameter
from .base import Kernel


class RBF(Kernel):
    """
    The radial basis function kernel.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "RBF"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the RBF specific function :math:`k(x,y)=\sigma^2 \exp\left( \frac{-0.5 \tau}{2\ell^2}\right) ` where  :math:`\tau = \lVert x-y \rVert_{2}^{2}`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=2)
        return sigma * jnp.exp(-0.5 * tau)


class Matern12(Kernel):
    """
    The Matern kernel with a smoothness parameter of 1/2.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Matern 1/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial    value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 \exp\left( \frac{-\tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        return sigma * jnp.exp(-0.5 * tau)


class Matern32(Kernel):
    """
    The Matern kernel with a smoothness parameter of 3/2.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Matern 3/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial    value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 (1 + \frac{\sqrt{3} \tau}{\ell}) \exp\left( \frac{\sqrt{3} \tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        return sigma*(1+(jnp.sqrt(3.)*tau))*jnp.exp(-jnp.sqrt(3.)*tau)


class Matern52(Kernel):
    """
    The Matern kernel with a smoothness parameter of 5/2.
    """
    def __init__(self,
                 lengthscale: jnp.ndarray = jnp.array([1.]),
                 variance: jnp.ndarray = jnp.array([1.]),
                 name: str = "Matern 5/2"):
        """
        Args:
            lengthscale: The initial value of the kernel's lengthscale value. The value of this parameter controls the horizontal magnitude of the kernel's resultant values.
            variance: The initial    value of the kernel's variance. This value controls the kernel's vertical amplitude.
            name: Optional argument to name the kernel.
        """
        super().__init__(lengthscale=lengthscale, variance=variance, name=name)

    def kernel_func(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Compute the Matern 1/2 specific function :math:`k(x,y)=\sigma^2 (1 + \frac{\sqrt{5} \tau}{\ell} + \frac{2.5*\tau**2}{\ell**2}) \exp\left( \frac{\sqrt{5} \tau}{\ell}\right) ` where  :math:`\tau = \lvert x-y \rvert`.
        """
        sigma = self.variance.untransform
        tau = self.scaled_distance(x, y, power=1)
        sqrt5 = jnp.sqrt(5.0)
        return sigma*(1.0+sqrt5*tau + (5.0/3.0) * jnp.square(tau))*jnp.exp(-sqrt5*tau)
