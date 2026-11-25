"""Semi-discrete dual solver for unbalanced transport between grid densities."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def c_transform(u, grid_x, grid_y, d):
    """Computes v(y) = max_x (<x, y> - u(x)) on Cartesian grids.

    The maximum is taken one axis at a time, which uses the separability of
    the inner product.

    Args:
        u: Values of u on the source grid, of shape (len(grid_x)**d,).
        grid_x: Source grid points per axis.
        grid_y: Target grid points per axis.
        d: Dimension.

    Returns:
        The values of v on the target grid, of shape (len(grid_y)**d,).
    """
    values = (-u).reshape((len(grid_x),) * d)
    inner = jnp.outer(grid_y, grid_x)
    for k in range(d):
        moved = jnp.expand_dims(jnp.moveaxis(values, k, -1), axis=-2)
        values = jnp.moveaxis(jnp.max(inner + moved, axis=-1), -1, k)
    return values.ravel()


def solve(coords, p, q, mass_p, mass_q, volume, rho=0.5, num_iter=2000, restarts=6):
    """Solves the dual problem and returns the induced transport-growth pair.

    The dual is minimized by L-BFGS-B with warm restarts. The restarts stop
    when the c-transform moves by at most 1e-5 in the sup norm, or when the
    objective stops decreasing.

    Args:
        coords: Grid points of shape (G, d), shared by both measures.
        p: Source density on the grid.
        q: Target density on the grid.
        mass_p: Total mass of the source measure.
        mass_q: Total mass of the target measure.
        volume: Volume of a grid cell.
        rho: Weight of each marginal KL penalty.
        num_iter: Maximum number of L-BFGS-B iterations per restart.
        restarts: Maximum number of restarts.

    Returns:
        A function that maps points of shape (B, d) to the transport map of
        shape (B, d) and the growth map of shape (B,).
    """
    mu = p.ravel() * mass_p * volume
    nu = q.ravel() * mass_q * volume
    d = coords.shape[1]
    grid_x = jnp.unique(coords[:, 0])
    grid_y = jnp.unique(coords[:, 0])
    sq_x = jnp.sum(coords**2, axis=-1)
    sq_y = jnp.sum(coords**2, axis=-1)

    def objective(u):
        v = c_transform(u, grid_x, grid_y, d)
        term_mu = jnp.sum(rho * (jnp.exp(-(0.5 * sq_x - u) / rho) - 1.0) * mu)
        term_nu = jnp.sum(rho * (jnp.exp(-(0.5 * sq_y - v) / rho) - 1.0) * nu)
        return term_mu + term_nu

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def fun(u):
        f, g = value_and_grad(jnp.asarray(u))
        return float(f), np.asarray(g, dtype=np.float64)

    u = np.asarray(0.5 * sq_x, dtype=np.float64)
    f_prev = np.inf
    v_prev = None
    options = {"maxiter": num_iter, "maxfun": 3 * num_iter, "ftol": 1e-17, "gtol": 1e-12}
    for _ in range(restarts):
        res = minimize(fun, u, jac=True, method="L-BFGS-B", options=options)
        u = res.x
        v = np.asarray(c_transform(jnp.asarray(u), grid_x, grid_y, d))
        if v_prev is not None and np.max(np.abs(v - v_prev)) <= 1e-5:
            break
        if f_prev - res.fun <= 1e-9 * max(1.0, abs(res.fun)):
            break
        v_prev = v
        f_prev = res.fun
    v = c_transform(jnp.asarray(u), grid_x, grid_y, d)
    psi = 0.5 * sq_y - v

    def point_map(x):
        scores = jnp.dot(coords, x) - v
        best = jnp.argmax(scores)
        phi = 0.5 * jnp.sum(x**2) - scores[best]
        return jnp.take(coords, best, axis=0), jnp.exp((-phi + jnp.take(psi, best)) / (4.0 * rho))

    batched = jax.jit(jax.vmap(point_map))

    def maps(points, batch_size=1000):
        points = np.asarray(points)
        out = [batched(points[i : i + batch_size]) for i in range(0, len(points), batch_size)]
        return (
            np.concatenate([np.asarray(t) for t, _ in out], axis=0),
            np.concatenate([np.asarray(lam) for _, lam in out], axis=0),
        )

    return maps
