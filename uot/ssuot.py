"""SSUOT baseline of Vacher and Vialard (2023)."""

import cvxpy as cp
import numpy as np

try:
    from mosek import Error as MosekError
except ImportError:
    MosekError = cp.error.SolverError


def _solve(problem):
    """Solves a problem with MOSEK, or with SCS if MOSEK is unavailable.

    Args:
        problem: A cvxpy problem.

    Returns:
        The name of the solver that was used.
    """
    try:
        problem.solve(solver=cp.MOSEK)
        return cp.MOSEK
    except (cp.error.SolverError, MosekError):
        problem.solve(solver=cp.SCS)
        return cp.SCS


def solve(x, y, a, b, rho=0.5, lower=0.0, upper=1.0):
    """Solves the semi-dual cone program with MOSEK or, as a fallback, SCS.

    Args:
        x: Source sample of shape (n, d).
        y: Target sample of shape (m, d).
        a: Source weights of shape (n,).
        b: Target weights of shape (m,).
        rho: Weight of each marginal KL penalty.
        lower: Lower bound of the gradients.
        upper: Upper bound of the gradients.

    Returns:
        A tuple of the potential values at x of shape (n,), the gradients at x
        of shape (n, d), and the target potential at the matched points, of
        shape (n,).

    Raises:
        RuntimeError: If the solver does not return an optimal solution.
    """
    mu = np.asarray(a, dtype=np.float64).ravel()
    nu = np.asarray(b, dtype=np.float64).ravel()
    n, d = x.shape
    m = y.shape[0]
    q_x = 0.5 * np.sum(x**2, axis=1)
    q_y = 0.5 * np.sum(y**2, axis=1)
    f = cp.Variable(n)
    g = cp.Variable(m)
    xi = cp.Variable((n, d))
    eta = cp.Variable((m, d))
    obj_mu = rho * cp.sum(cp.multiply(mu, cp.exp((f - q_x) / rho) - 1))
    obj_nu = rho * cp.sum(cp.multiply(nu, cp.exp((g - q_y) / rho) - 1))
    xy = x @ y.T
    constraints = [
        xi >= lower,
        xi <= upper,
        eta >= lower,
        eta <= upper,
        f[:, np.newaxis] + g[np.newaxis, :] >= xy,
        f[:, np.newaxis] - f[np.newaxis, :] >= (x @ xi.T) - cp.sum(cp.multiply(x, xi), axis=1)[np.newaxis, :],
        g[:, np.newaxis] - g[np.newaxis, :] >= (y @ eta.T) - cp.sum(cp.multiply(y, eta), axis=1)[np.newaxis, :],
    ]
    problem = cp.Problem(cp.Minimize(obj_mu + obj_nu), constraints)
    solver = _solve(problem)
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"{solver} returned status {problem.status}")
    best = np.argmax(xy - g.value[np.newaxis, :], axis=1)
    return f.value, xi.value, (q_y - g.value)[best]


def extend(coords, x, f, xi, psi, rho=0.5):
    """Extends the solution to the grid by the maximum of affine minorants.

    Args:
        coords: Grid points of shape (G, d).
        x: Source sample of shape (n, d).
        f: Potential values at x, of shape (n,).
        xi: Gradients at x, of shape (n, d).
        psi: Target potential at the matched points, of shape (n,).
        rho: Weight of each marginal KL penalty.

    Returns:
        A tuple of the transport map of shape (G, d) and the growth map of
        shape (G,).
    """
    size, d = coords.shape
    u = np.zeros(size)
    t = np.zeros((size, d))
    psi_t = np.zeros(size)
    for i in range(size):
        planes = f + np.sum(xi * (coords[i] - x), axis=1)
        best = np.argmax(planes)
        u[i] = planes[best]
        t[i] = xi[best]
        psi_t[i] = psi[best]
    phi = 0.5 * np.sum(coords**2, axis=1) - u
    return t, np.exp((-phi + psi_t) / (4.0 * rho))
