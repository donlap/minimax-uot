"""Plan-based estimators from the discrete unbalanced transport plan."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


@partial(jax.jit, static_argnames=("num_iter",))
def mm_unbalanced(a, b, cost, reg, num_iter=2000, stop_thr=1e-15):
    """Solves KL-penalized unbalanced transport by majorization-minimization.

    Adapted to JAX from ot.unbalanced.mm_unbalanced of the POT library,
    without the entropic term.

    Args:
        a: Source weights of shape (n,).
        b: Target weights of shape (m,).
        cost: Cost matrix of shape (n, m).
        reg: Weight of each marginal KL penalty.
        num_iter: Maximum number of iterations, checked every num_iter // 20.
        stop_thr: Tolerance on the change of the plan.

    Returns:
        The transport plan of shape (n, m).
    """
    eps = jnp.finfo(a.dtype).eps
    interval = max(1, num_iter // 20)
    total = reg + reg
    r1 = reg / total
    r2 = reg / total
    kernel = (a[:, None] ** r1) * (b[None, :] ** r2) * jnp.exp(-cost / total)

    def step(plan, _):
        rows = jnp.sum(plan, axis=1, keepdims=True)
        cols = jnp.sum(plan, axis=0, keepdims=True)
        return kernel * plan / ((rows**r1) * (cols**r2) + eps), None

    def body(state):
        i, _, plan = state
        before, _ = jax.lax.scan(step, plan, None, length=max(0, interval - 1))
        after, _ = step(before, None)
        return i + interval, jnp.linalg.norm(after - before), after

    def cond(state):
        i, err, _ = state
        return jnp.logical_and(i < num_iter, err > stop_thr)

    return jax.lax.while_loop(cond, body, (0, jnp.inf, jnp.outer(a, b)))[2]


def plan_maps(x, y, a, b, rho=0.5, num_iter=2000, normalize=False):
    """Computes the transport and growth values at the source points.

    The computation runs in the floating-point precision of x.

    Args:
        x: Source sample of shape (n, d).
        y: Target sample of shape (m, d).
        a: Source weights of shape (n,).
        b: Target weights of shape (m,).
        rho: Weight of each marginal KL penalty.
        num_iter: Number of solver iterations.
        normalize: Whether to divide the cost by its median.

    Returns:
        A tuple of the barycentric map of shape (n, d) and the growth values of
        shape (n,).
    """
    cost = 0.5 * cdist(x, y, metric="sqeuclidean")
    scale = np.median(cost) if normalize else 1.0
    cost = (cost / scale).astype(x.dtype)
    plan = mm_unbalanced(np.asarray(a), np.asarray(b), cost, rho, num_iter=num_iter)
    rows = jnp.sum(plan, axis=1)
    ratio = rows / (a + 1e-15)
    t = jnp.dot(plan, y) / (rows[:, None] + 1e-15)
    shift = jnp.where(rows > 1e-12, jnp.sum((x - t) ** 2, axis=-1) / x.dtype.type(scale), 0.0)
    lam = jnp.sqrt(ratio) * jnp.exp(0.25 * shift)
    return np.asarray(t), np.asarray(lam)


def extend_1nn(x, coords, t, lam):
    """Extends the sample values to the grid by the nearest neighbor.

    Args:
        x: Source sample of shape (n, d).
        coords: Grid points of shape (G, d).
        t: Transport values at the sample, of shape (n, d).
        lam: Growth values at the sample, of shape (n,).

    Returns:
        A tuple of the transport map of shape (G, d) and the growth map of
        shape (G,).
    """
    nearest = np.argmin(cdist(coords, x), axis=1)
    return t[nearest], lam[nearest]


def extend_kernel(x, coords, t, lam, bandwidth=None):
    """Extends the sample values to the grid by Gaussian kernel smoothing.

    Without a given bandwidth, the bandwidth maximizes the 5-fold
    cross-validated likelihood of a Gaussian kernel density estimate on 30
    log-spaced values around Silverman's rule.

    Args:
        x: Source sample of shape (n, d).
        coords: Grid points of shape (G, d).
        t: Transport values at the sample, of shape (n, d).
        lam: Growth values at the sample, of shape (n,).
        bandwidth: Optional fixed bandwidth.

    Returns:
        A tuple of the transport map of shape (G, d) and the growth map of
        shape (G,).
    """
    h = bandwidth
    if h is None:
        n, d = x.shape
        silverman = 1.06 * np.mean(np.std(x, axis=0)) * (n ** (-1.0 / (d + 4)))
        bandwidths = np.logspace(np.log10(silverman / 5), np.log10(silverman * 5), 30)
        search = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=5, n_jobs=-1)
        h = search.fit(x).best_estimator_.bandwidth
    k = np.exp(-cdist(coords, x, metric="sqeuclidean") / (2 * h**2))
    w = k / np.maximum(np.sum(k, axis=1, keepdims=True), 1e-12)
    return w @ t, np.sqrt(w @ (lam**2))
