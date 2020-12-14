import jax.numpy as jnp
from gpjax.parameters import Parameter
from gpjax.transforms import Softplus, Identity
import pytest


def hardcode_softplus(x: jnp.ndarray):
    return jnp.log(jnp.exp(x)-1.0)


@pytest.mark.parametrize("val", [0.5, 1.0])
def test_softplus(val):
    v = jnp.array([val])
    x = Parameter(v, transform=Softplus())
    assert x.untransform == v
    assert x.value == hardcode_softplus(v)


@pytest.mark.parametrize("val", [1.0, 2.0])
def test_identity(val):
    v = jnp.array([val])
    x = Parameter(v, transform=Identity())
    assert x.untransform == v
    assert x.value == v
