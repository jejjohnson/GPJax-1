import jax.numpy as jnp
from jax.nn import softplus
from typing import Optional


class Transform:
    """
    Base class for parameter transforms
    """
    def __init__(self, name: Optional[str] = "Transformation"):
        self.name = name

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        """
        Map from an unconstrained space to the original parameter space.

        Args:
            x: The transformed parameter value.

        Returns: The untransformed parameter.
        """
        raise NotImplementedError

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        """
        Map from an constrained space to the space of all real numbers.

        Args:
            x: The constrained parameter value.

        Returns: An unconstrained form of the parameter.
        """
        raise NotImplementedError


class Softplus(Transform):
    """
    The softplus transformation.
    """
    def __init__(self):
        super().__init__(name="Softplus")

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Map from an unconstrained space to the original parameter space by the transformation .. math::
            \log(\exp(x) - 1).

        Args:
            x: The transformed parameter value.

        Returns: The untransformed parameter.
        """
        return jnp.log(jnp.exp(x) - 1.)

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        r"""
        Map from an constrained space to the space of all real numbers by the transformation .. math::
            \log(\exp(x) + 1).

        Args:
            x: The constrained parameter value.

        Returns: An unconstrained form of the parameter.
        """
        return softplus(x)


class Identity(Transform):
    """
    The identity transformation. This is equivalent to having no transformation applied.
    """
    def __init__(self):
        super().__init__(name='Identity')

    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        """
        Return the original parameter.

        Args:
            x: The parameter's value.

        Returns: The parameter's value.
        """
        return x

    @staticmethod
    def backward(x: jnp.ndarray) -> jnp.ndarray:
        """
        Return the original parameter.

        Args:
            x: The parameter's value.

        Returns: The parameter's value.
        """
        return x
