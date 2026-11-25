"""Grids, densities and samplers of the simulation."""

import numpy as np


def grid(d, size):
    """Builds a uniform Cartesian grid on the unit cube.

    Args:
        d: Dimension.
        size: Number of points per axis.

    Returns:
        A tuple of the 1D grid, its spacing, and the grid points of shape
        (size**d, d).
    """
    grid_1d = np.linspace(0, 1, size)
    mesh = np.meshgrid(*([grid_1d] * d), indexing="ij")
    coords = np.column_stack([m.ravel() for m in mesh])
    return grid_1d, grid_1d[1] - grid_1d[0], coords


def density(coords, location, theta, dx):
    """Evaluates the mixture density on a grid.

    The density is (1 - theta) p + theta, where p is proportional to
    prod_i |sin(pi (x_i - location))|. Both components are normalized by the
    Riemann sum on the grid.

    Args:
        coords: Grid points of shape (G, d).
        location: Shift of the sine factors.
        theta: Weight of the uniform component.
        dx: Grid spacing.

    Returns:
        The density values of shape (G,).
    """
    d = coords.shape[1]
    p = np.prod(np.abs(np.sin(np.pi * (coords - location))), axis=1)
    p /= np.sum(p) * (dx**d)
    unif = np.ones_like(p)
    unif /= np.sum(unif) * (dx**d)
    return (1.0 - theta) * p + theta * unif


def sample(n, d, location, rng, theta):
    """Draws a sample from the mixture density.

    Each point is uniform with probability theta and is otherwise drawn from
    p by rejection.

    Args:
        n: Sample size.
        d: Dimension.
        location: Shift of the sine factors.
        rng: A numpy Generator.
        theta: Weight of the uniform component.

    Returns:
        The sample of shape (n, d).
    """
    n_unif = rng.binomial(n, theta)
    unif = rng.uniform(0.0, 1.0, (n_unif, d))
    n_sin = n - n_unif
    accepted = []
    while len(accepted) < n_sin:
        cand = rng.uniform(0.0, 1.0, (n, d))
        u = rng.uniform(0.0, 1.0, n)
        pdf = np.prod(np.abs(np.sin(np.pi * (cand - location))), axis=1)
        accepted.extend(cand[u < pdf])
    sin = np.array(accepted[:n_sin]).reshape(n_sin, d)
    return rng.permutation(np.concatenate([sin, unif], axis=0), axis=0)
