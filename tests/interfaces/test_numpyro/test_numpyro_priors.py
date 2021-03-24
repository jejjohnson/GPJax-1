from gpjax.interfaces.numpyro import add_constraints, numpyro_dict_params, add_priors
import pytest

import jax.numpy as jnp
import jax.random as jr

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import chex

from gpjax.gps import Prior
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.parameters import initialise


# TODO: test conjugate posterior
def get_conjugate_posterior_params() -> dict:
    kernel = RBF()
    prior = Prior(kernel=kernel)
    lik = Gaussian()
    posterior = prior * lik
    params = initialise(posterior)
    return params


@pytest.mark.parametrize(
    "prior",
    [
        dist.Gamma(concentration=1.0, rate=1.0),
        dist.HalfCauchy(scale=1.0),
        dist.LogNormal(loc=0.0, scale=1.0),
    ],
)
def test_numpyro_add_priors_all(prior):

    gpjax_params = get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # add constraint
    new_numpyro_params = add_priors(numpyro_params, prior)
    for iparams in new_numpyro_params.values():

        # check if constraint in new dictionary
        chex.assert_equal(iparams["param_type"], "prior")
        chex.assert_equal(iparams["prior"], prior)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, get_conjugate_posterior_params())


@pytest.mark.parametrize(
    "variable",
    ["lengthscale", "obs_noise", "variance"],
)
@pytest.mark.parametrize(
    "prior",
    [
        dist.Gamma(concentration=1.0, rate=1.0),
        dist.HalfCauchy(scale=1.0),
        dist.LogNormal(loc=0.0, scale=1.0),
    ],
)
def test_numpyro_add_priors_str(variable, prior):

    gpjax_params = get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # add constraint
    new_numpyro_params = add_priors(numpyro_params, variable, prior)

    # check if constraint in new dictionary
    chex.assert_equal(new_numpyro_params[variable]["param_type"], "prior")
    chex.assert_equal(new_numpyro_params[variable]["prior"], prior)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, get_conjugate_posterior_params())


@pytest.mark.parametrize(
    "variable",
    ["lengthscale", "obs_noise", "variance"],
)
@pytest.mark.parametrize(
    "prior",
    [
        dist.Gamma(concentration=1.0, rate=1.0),
        dist.HalfCauchy(scale=1.0),
        dist.LogNormal(loc=0.0, scale=1.0),
    ],
)
def test_numpyro_add_priors_dict(variable, prior):

    gpjax_params = get_conjugate_posterior_params()
    numpyro_params = numpyro_dict_params(gpjax_params)

    # create new dictionary
    new_param_dict = {str(variable): prior}

    # add constraint
    new_numpyro_params = add_priors(numpyro_params, new_param_dict)

    # check if constraint in new dictionary
    chex.assert_equal(new_numpyro_params[variable]["param_type"], "prior")
    chex.assert_equal(new_numpyro_params[variable]["prior"], prior)

    # check we didn't modify original dictionary
    chex.assert_equal(gpjax_params, get_conjugate_posterior_params())