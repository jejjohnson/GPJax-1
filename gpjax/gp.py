from .likelihoods import Gaussian, Likelihood
from .kernel import Kernel
from .spectral import SpectralKernel
from .mean_functions import MeanFunction
from .mean_functions import ZeroMean
from objax import Module
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor
import jax.random as jr
from tensorflow_probability.substrates import jax as tfp
from typing import Optional
from jax.scipy.linalg import solve_triangular

tfd = tfp.distributions


class Prior(Module):
    r"""
    The base class for Gaussian process priors. Considering a set :math:`X` and  function :math:`f`, the GP induces the
    prior :math:`p(f)\sim\mathcal{GP}(m, k)` where :math:`m: X \rightarrow \mathbb{R}` is a mean function and kernel
    :math:`k: X \times X \rightarrow \mathbb{R}`.
    """
    def __init__(self,
                 kernel: Kernel,
                 mean_function: Optional[MeanFunction] = ZeroMean(),
                 jitter: Optional[float] = 1e-6):
        """
        Args:
            kernel: The Gaussian process model's kernel, or covariance, function.
            mean_function: The prior mean function. This is optional and will default to a zero-mean function.
            jitter: A small amount of noise to stabilise the prior covariance matrix.
        """
        self.meanf = mean_function
        self.kernel = kernel
        self.jitter = jitter

    def sample(self,
               X: jnp.ndarray,
               key,
               n_samples: Optional[int] = 1) -> jnp.ndarray:
        """
        Draw a set of n samples from the GP prior at a set of input points.

        Args:
            X: The finite set of points from where we wish to sample from the GP prior.
            key: The Jax key to ensure reproducibility
            n_samples: The number of samples to be drawn.

        Returns: A Jax array of samples.

        """
        mu = self.meanf(X)
        cov = self.kernel(X, X)
        if cov.shape[0] == cov.shape[1]:
            Inn = jnp.eye(cov.shape[0]) * self.jitter
            cov += Inn
        return jr.multivariate_normal(key,
                                      mu.squeeze(),
                                      cov,
                                      shape=(n_samples, ))

    def __mul__(self, other: Likelihood):
        """
        The posterior distribution is proportional to the product of the prior and the data's likelihood. This magic
        method enables this mathematical behaviour to be represented computationally.

        Args:
            other: A likelihood distribution.

        Returns: A Gaussian process posterior.
        """
        if isinstance(self.kernel, SpectralKernel):
            return SpectralPosterior(self, other)
        else:
            return Posterior(self, other)


class Posterior(Module):
    """
    A Gaussian process posterior whereby the likelihood function is a Gaussian distribution.
    """
    def __init__(self, prior: Prior, likelihood: Gaussian):
        self.kernel = prior.kernel
        self.meanf = prior.meanf
        self.likelihood = likelihood
        self.jitter = prior.jitter

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""
        Here we compute :math:`\log p(y | x, \theta)
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A multivariate normal distribution

        """
        Inn = jnp.eye(X.shape[0])
        mu = self.meanf(X)
        cov = self.kernel(X, X) + self.jitter * Inn
        cov += self.likelihood.noise.untransform * Inn
        L = jnp.linalg.cholesky(cov)
        return tfd.MultivariateNormalTriL(loc=mu, scale_tril=L)

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        In gradient based optimisation of the marginal log-likelihood (MLL), we minimise the negative MLL.
        Args:
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A unary float array

        """
        rv = self.marginal_ll(X, y)
        return -rv.log_prob(y.squeeze()).mean()

    def predict(self, Xstar, X, y):
        """
        Conditional upon the GP posterior, compute the predictive posterior given a set of new and unseen test points.
        Args:
            Xstar: The new inputs that correspond to the locations that we'd like to sample the GP predictive posterior at
            X: A set of N X M inputs
            y: A set of N X 1 outputs

        Returns: A predictive mean and predictive variance

        """
        # Compute covariance matrices and jitter matrix
        Kff = self.kernel(X, X)
        Kfx = self.kernel(Xstar, X)
        Kxx = self.kernel(Xstar, Xstar)
        Inn = jnp.eye(X.shape[0])
        # Compute prior mean
        mu_f = self.meanf(X)
        # Realise the current estimate of the observational noise
        sigma = self.likelihood.noise.untransform
        # Compute the lower Cholesky decomposition
        L = cho_factor(Kff + Inn * sigma, lower=True)
        err = y.reshape(-1, 1) - mu_f.reshape(-1, 1)
        weights = cho_solve(L, err)
        # Compute the predictive mean
        mu = jnp.dot(Kfx, weights)
        # Compute the predictive variance
        v = cho_solve(L, Kfx.T)
        cov = Kxx - jnp.dot(Kfx, v)
        return mu, cov


class SpectralPosterior(Posterior):
    def __init__(self, prior: Prior, likelihood: Gaussian):
        super().__init__(prior, likelihood)

    def marginal_ll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        N = X.shape[0]
        m = self.kernel.num_basis
        l_var = self.likelihood.noise.untransform
        k_var = self.kernel.variance.untransform
        phi = self.kernel._compute_phi(X)
        A = (k_var / m) * jnp.matmul(phi.T, phi) + l_var * jnp.eye(m * 2)
        assert A.shape == (2*m, 2*m)
        Rt = jnp.linalg.cholesky(A)
        RtiPhit = solve_triangular(Rt, phi.T)
        # assert RtiPhit.shape == (N, N)
        RtiPhity = jnp.matmul(RtiPhit, y)
        # assert RtiPhity.shape == y.shape
        term1 = (jnp.sum(y**2) -
                 jnp.sum(RtiPhity**2) * k_var / m) * 0.5 / l_var
        term2 = jnp.sum(jnp.log(jnp.diag(
            Rt.T))) + (N * 0.5 - m) * jnp.log(l_var) + (N * 0.5 *
                                                        jnp.log(2 * jnp.pi))
        tot = term1 + term2
        return tot.reshape()

    def neg_mll(self, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return self.marginal_ll(X, y)

    def predict(self, Xstar, X, y):
        N = X.shape[0]
        m = self.kernel.num_basis
        l_var = self.likelihood.noise.untransform
        k_var = self.kernel.variance.untransform
        phi = self.kernel._compute_phi(X)
        A = (k_var / m) * jnp.matmul(phi.T, phi) + l_var * jnp.eye(m * 2)
        Rt = jnp.linalg.cholesky(A)
        RtiPhit = solve_triangular(Rt, phi.T)
        RtiPhity = jnp.matmul(RtiPhit, y)

        alpha = (k_var/m) * solve_triangular(Rt, RtiPhity, lower=False)
        omega = self.kernel.scale_frequencies()
        phistar = jnp.matmul(Xstar, omega.T)

        cos_freqs = jnp.cos(phistar)
        sin_freqs = jnp.sin(phistar)
        phistar = jnp.hstack((cos_freqs, sin_freqs))
        mu = jnp.matmul(phistar, alpha)

        RtiPhistart = solve_triangular(Rt, phistar.T)
        PhiRistar = RtiPhistart.T

        cov = l_var*(k_var/m) * jnp.matmul(PhiRistar, PhiRistar.T) + jnp.eye(Xstar.shape[0])*self.jitter
        return mu, cov