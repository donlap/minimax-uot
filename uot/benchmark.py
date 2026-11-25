"""Simulation study comparing the estimators of the transport-growth pair."""

import numpy as np
import pandas as pd

from uot import continuous, data, discrete, ssuot
from uot.density import NeumannKernelDensity, WaveletDensity, select_cutoff, wavelet_level

DIMENSIONS = (1, 2, 3, 4)
SAMPLE_SIZES = (100, 200, 500, 1000)
GRID_SIZE = {1: 128, 2: 64, 3: 32, 4: 32}
CUTOFF_RANGE = {1: (6.0, 100.0), 2: (6.0, 60.0), 3: (6.0, 30.0), 4: (6.0, 20.0)}
MASS_SOURCE = 1.0
MASS_TARGET = 2.5
LOCATION_SOURCE = 1 / 3
LOCATION_TARGET = 2 / 3
THETA = 0.1
RHO = 0.5
FOLD_SEED = 42
TRUTH_RESTARTS = 12


def fold_orders(seed=FOLD_SEED):
    """Draws the permutations that assign the cross-validation folds.

    The permutations are drawn from one stream in the order of DIMENSIONS
    and SAMPLE_SIZES, so each dimension gets the same folds whichever
    dimensions run.

    Args:
        seed: Seed of the stream.

    Returns:
        A dict mapping (d, n) to a permutation of range(n).
    """
    stream = np.random.RandomState(seed)
    return {(d, n): stream.permutation(n) for d in DIMENSIONS for n in SAMPLE_SIZES}


def fit_restarts(d, seed):
    """Returns the number of dual restarts for a fitted plug-in estimator.

    Args:
        d: Dimension.
        seed: Replicate index.

    Returns:
        The maximum number of restarts.
    """
    if d <= 2:
        return 6
    return 2 if seed == 0 else 1


def errors(estimate, truth, weight, volume):
    """Computes the weighted squared errors of a transport-growth pair.

    Args:
        estimate: Pair of the transport map of shape (G, d) and the growth
            map of shape (G,).
        truth: Pair of the true maps with the same shapes.
        weight: Source density on the grid, of shape (G,).
        volume: Volume of a grid cell.

    Returns:
        A tuple of the transport map error and the growth map error.
    """
    mse_map = np.sum(weight * np.sum((estimate[0] - truth[0]) ** 2, axis=1)) * volume
    mse_growth = np.sum(weight * (estimate[1] - truth[1]) ** 2) * volume
    return mse_map, mse_growth


def estimates(x, y, coords, grid_1d, volume, cutoff, level, restarts):
    """Fits the five estimators and evaluates them on the grid.

    Args:
        x: Source sample of shape (n, d).
        y: Target sample of shape (n, d).
        coords: Grid points of shape (G, d).
        grid_1d: Grid points per axis.
        volume: Volume of a grid cell.
        cutoff: Frequency cutoff of PI-Kernel.
        level: Number of decomposition levels of PI-Wavelet.
        restarts: Maximum number of dual restarts of the plug-in estimators.

    Returns:
        A dict mapping each estimator name to its transport-growth pair on
        the grid.
    """
    n = len(x)
    a = np.ones(n) * (MASS_SOURCE / n)
    b = np.ones(n) * (MASS_TARGET / n)
    t, lam = discrete.plan_maps(x, y, a, b, rho=RHO)
    f, xi, psi = ssuot.solve(x, y, a, b, rho=RHO)
    out = {
        "PB-1NN": discrete.extend_1nn(x, coords, t, lam),
        "PB-Kernel": discrete.extend_kernel(x, coords, t, lam),
        "SSUOT": ssuot.extend(coords, x, f, xi, psi, rho=RHO),
    }
    for name, model in (("PI-Wavelet", WaveletDensity(level)), ("PI-Kernel", NeumannKernelDensity(x.shape[1], cutoff))):
        p, q = model.estimate(x, grid_1d), model.estimate(y, grid_1d)
        maps = continuous.solve(coords, p, q, MASS_SOURCE, MASS_TARGET, volume, rho=RHO, restarts=restarts)
        out[name] = maps(coords)
    return out


def run(dims=DIMENSIONS, seeds=10, out=None):
    """Runs the simulation study.

    Args:
        dims: Dimensions.
        seeds: Number of replicates per setting.
        out: Optional CSV path, rewritten after each setting.

    Returns:
        A DataFrame with one row per dimension, sample size, replicate and
        estimator.
    """
    orders = fold_orders()
    rows = []
    for d in dims:
        grid_1d, dx, coords = data.grid(d, GRID_SIZE[d])
        volume = dx**d
        p0 = data.density(coords, LOCATION_SOURCE, THETA, dx)
        q0 = data.density(coords, LOCATION_TARGET, THETA, dx)
        maps = continuous.solve(coords, p0, q0, MASS_SOURCE, MASS_TARGET, volume, rho=RHO, restarts=TRUTH_RESTARTS)
        truth = maps(coords)
        for n in SAMPLE_SIZES:
            x_cv = data.sample(n, d, LOCATION_SOURCE, np.random.default_rng(10_000 + n), THETA)
            cutoff = select_cutoff(x_cv, orders[d, n], *CUTOFF_RANGE[d])
            level = wavelet_level(n, d, GRID_SIZE[d])
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                x = data.sample(n, d, LOCATION_SOURCE, rng, THETA)
                y = data.sample(n, d, LOCATION_TARGET, rng, THETA)
                fits = estimates(x, y, coords, grid_1d, volume, cutoff, level, fit_restarts(d, seed))
                for name, pair in fits.items():
                    mse_map, mse_growth = errors(pair, truth, p0, volume)
                    rows.append(
                        {"d": d, "n": n, "seed": seed, "estimator": name, "mse_map": mse_map, "mse_growth": mse_growth}
                    )
            if out is not None:
                pd.DataFrame(rows).to_csv(out, index=False)
            print(f"d={d} n={n} cutoff={cutoff:.2f} level={level}", flush=True)
    return pd.DataFrame(rows)
