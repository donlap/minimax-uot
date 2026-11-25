"""Wavelet and Neumann kernel density estimators on grids of the unit cube."""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pywt
import scipy.ndimage as ndi


def wavelet_level(n, d, grid_size, smoothness=2):
    """Returns the resolution level 2^J of order n^(1 / (d + 2 (s - 1))).

    Args:
        n: Sample size.
        d: Dimension.
        grid_size: Number of grid points per axis.
        smoothness: Smoothness s of the densities.

    Returns:
        The number of decomposition levels J.
    """
    level = int(np.ceil(np.log2(n) / (d + 2.0 * (smoothness - 1))))
    return max(1, min(level, int(np.log2(grid_size)) - 1))


def _sure_threshold(coeffs, size):
    """Selects a soft threshold by Stein's unbiased risk estimate.

    Args:
        coeffs: Standardized detail coefficients.
        size: Number of grid points per axis.

    Returns:
        The threshold, raised to sqrt(2 log size) when it is below half of
        that value.
    """
    w = np.sort(np.abs(coeffs).ravel())
    m = len(w)
    if m == 0:
        return 0.0
    universal = np.sqrt(2.0 * np.log(size))
    k = np.arange(1, m + 1)
    risks = (m - 2.0 * k) + np.cumsum(w**2) + (m - k) * w**2
    threshold = w[np.argmin(risks)]
    if threshold < 0.5 * universal:
        return universal
    return threshold


class WaveletDensity:
    """Thresholded stationary wavelet density estimator.

    The estimator bins the sample on the grid, extends the histogram by even
    reflection, applies the db4 stationary wavelet transform, soft-thresholds
    the detail coefficients with a locally scaled SURE threshold, inverts the
    transform, and renormalizes the positive part.

    Attributes:
        level: Number of decomposition levels.
    """

    def __init__(self, level):
        """Initializes the estimator.

        Args:
            level: Number of decomposition levels.
        """
        self.level = level

    def estimate(self, samples, grid_1d):
        """Estimates the density on the grid.

        Args:
            samples: Sample of shape (n, d).
            grid_1d: Grid points per axis, of power-of-two length.

        Returns:
            The density values of shape (len(grid_1d),) * d.
        """
        samples = np.asarray(samples, dtype=np.float64)
        grid_1d = np.asarray(grid_1d, dtype=np.float64)
        d = samples.shape[1]
        size = len(grid_1d)
        spacing = grid_1d[1] - grid_1d[0]
        edges = [np.linspace(grid_1d[0] - spacing / 2.0, grid_1d[-1] + spacing / 2.0, size + 1) for _ in range(d)]
        hist, _ = np.histogramdd(samples, bins=edges, density=True)
        padded = np.pad(hist, pad_width=[(0, size)] * d, mode="symmetric")
        coeffs = pywt.swtn(padded, wavelet="db4", level=self.level)
        approx = "a" * d
        thresholded = []
        for bands in coeffs:
            kept = {}
            for key, detail in bands.items():
                if key == approx:
                    kept[key] = detail
                    continue
                mean = ndi.uniform_filter(detail, size=3)
                sq_mean = ndi.uniform_filter(detail**2, size=3)
                sigma = np.maximum(np.sqrt(np.maximum(sq_mean - mean**2, 0.0)), 1e-12)
                threshold = _sure_threshold(detail / sigma, size) * sigma
                kept[key] = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
            thresholded.append(kept)
        estimate = pywt.iswtn(thresholded, wavelet="db4")[tuple(slice(0, size) for _ in range(d))]
        estimate = np.maximum(estimate, 0.0)
        integral = np.sum(estimate) * (spacing**d)
        if integral > 0:
            estimate /= integral
        return estimate


@jax.jit
def _taper(t):
    """Smooth taper equal to 1 on [0, 1/2) and to 0 on [1, inf).

    Args:
        t: Squared frequency norms divided by the squared cutoff.

    Returns:
        The taper values.
    """
    is_one = t < 0.5
    is_transition = (t >= 0.5) & (t < 1.0)
    u = jnp.where(is_transition, 2.0 * t - 1.0, 0.5)
    exp_1 = jnp.exp(-1.0 / (1.0 - u + 1e-15))
    exp_2 = jnp.exp(-1.0 / (u + 1e-15))
    values = jnp.where(is_one, 1.0, 0.0)
    return jnp.where(is_transition, exp_1 / (exp_1 + exp_2), values)


@partial(jax.jit, static_argnums=(2, 3))
def _basis_sum(points, freqs, n_freqs, d):
    """Sums the unnormalized cosine basis over the points.

    Args:
        points: Points of shape (B, d).
        freqs: Frequencies of shape (K, d).
        n_freqs: Number of frequencies K.
        d: Dimension.

    Returns:
        The sums of shape (K,).
    """
    phi = jnp.ones((points.shape[0], n_freqs), dtype=jnp.float32)
    for r in range(d):
        phi *= jnp.cos(jnp.pi * points[:, r, None] * freqs[None, :, r])
    return jnp.sum(phi, axis=0)


@partial(jax.jit, static_argnums=(2, 3))
def _series(points, freqs, n_freqs, d, coeffs):
    """Evaluates a cosine series at the points.

    Args:
        points: Points of shape (B, d).
        freqs: Frequencies of shape (K, d).
        n_freqs: Number of frequencies K.
        d: Dimension.
        coeffs: Series coefficients of shape (K,).

    Returns:
        The series values of shape (B,).
    """
    phi = jnp.ones((points.shape[0], n_freqs), dtype=jnp.float32)
    for r in range(d):
        phi *= jnp.cos(jnp.pi * points[:, r, None] * freqs[None, :, r])
    return phi @ coeffs


class NeumannKernelDensity:
    """Kernel density estimator built on the Neumann Laplacian eigenbasis.

    The estimator expands the empirical measure in the cosine basis of the
    unit cube, tapers the frequencies k with |k| <= L, and renormalizes the
    positive part on a grid.

    Attributes:
        d: Dimension.
        freqs: Retained frequencies of shape (K, d).
        norms: Squared norms of the unnormalized basis functions.
        weights: Taper times the squared normalizing constants.
    """

    def __init__(self, d, cutoff):
        """Initializes the estimator.

        Args:
            d: Dimension.
            cutoff: Frequency cutoff L.
        """
        self.d = d
        k = np.array(list(itertools.product(range(int(np.floor(cutoff)) + 1), repeat=d)))
        sq_norms = np.sum(k**2, axis=1)
        valid = sq_norms <= cutoff**2
        self.freqs = jnp.array(k[valid], dtype=jnp.float32)
        ratio = jnp.array(sq_norms[valid] / (cutoff**2), dtype=jnp.float32)
        c = jnp.where(self.freqs > 0, jnp.sqrt(2.0), 1.0)
        self.norms = jnp.prod(c**2, axis=1)
        self.weights = _taper(ratio) * self.norms

    def coefficients(self, samples, batch_size=50000):
        """Computes the tapered series coefficients.

        Args:
            samples: Sample of shape (n, d).
            batch_size: Number of points per batch.

        Returns:
            The coefficients of shape (K,).
        """
        samples = jnp.array(samples, dtype=jnp.float32)
        n_freqs = len(self.freqs)
        coeffs = jnp.zeros(n_freqs, dtype=jnp.float32)
        for i in range(0, len(samples), batch_size):
            coeffs += _basis_sum(samples[i : i + batch_size], self.freqs, n_freqs, self.d)
        coeffs /= len(samples)
        return coeffs * self.weights

    def estimate(self, samples, grid_1d):
        """Estimates the density on the grid.

        Args:
            samples: Sample of shape (n, d).
            grid_1d: Grid points per axis.

        Returns:
            The density values of shape (len(grid_1d),) * d.
        """
        grid_1d = np.asarray(grid_1d)
        dx = grid_1d[1] - grid_1d[0]
        mesh = np.meshgrid(*([grid_1d] * self.d), indexing="ij")
        coords = jnp.array(np.column_stack([m.ravel() for m in mesh]), dtype=jnp.float32)
        n_freqs = len(self.freqs)
        safe = max(1, int(1.5 * (1024**3) / (16 * n_freqs)))
        batch_size = min(5000, safe)
        coeffs = self.coefficients(samples, batch_size=min(50000, safe))
        density = jnp.concatenate(
            [
                _series(coords[i : i + batch_size], self.freqs, n_freqs, self.d, coeffs)
                for i in range(0, coords.shape[0], batch_size)
            ]
        )
        density = jnp.maximum(density, 0.0)
        integral = jnp.sum(density) * (dx**self.d)
        density = jnp.where(integral > 0, density / integral, density)
        return np.array(density).reshape((len(grid_1d),) * self.d)

    def lscv(self, train, test):
        """Computes the least-squares cross-validation risk of one split.

        Args:
            train: Training sample of shape (n_train, d).
            test: Validation sample of shape (n_test, d).

        Returns:
            The squared L2 norm of the series minus twice its mean at the
            validation points.
        """
        coeffs = self.coefficients(train)
        sq_l2 = float(jnp.sum(coeffs**2 / self.norms))
        values = _series(jnp.array(test, dtype=jnp.float32), self.freqs, len(self.freqs), self.d, coeffs)
        return sq_l2 - 2.0 * float(jnp.mean(values))


def select_cutoff(samples, order, low, high, folds=10, tol=0.1):
    """Selects the frequency cutoff by ternary search on the LSCV risk.

    Args:
        samples: Sample of shape (n, d).
        order: Permutation of range(n) that assigns the folds.
        low: Lower end of the search interval.
        high: Upper end of the search interval.
        folds: Number of folds.
        tol: Length of the final interval.

    Returns:
        The midpoint of the final interval.
    """
    d = samples.shape[1]
    parts = np.array_split(order, folds)

    def risk(cutoff):
        model = NeumannKernelDensity(d, cutoff)
        total = 0.0
        for i in range(folds):
            train = np.concatenate([parts[j] for j in range(folds) if j != i])
            total += model.lscv(samples[train], samples[parts[i]])
        return total

    while high - low > tol:
        m1 = low + (high - low) / 3.0
        m2 = high - (high - low) / 3.0
        if risk(m1) < risk(m2):
            high = m2
        else:
            low = m1
    return (low + high) / 2.0
