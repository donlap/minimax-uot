"""Figures of the simulation study and the shape completion experiment."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress

ESTIMATORS = ["PI-Kernel", "PI-Wavelet", "PB-Kernel", "PB-1NN", "SSUOT"]
LABELS = {"SSUOT": "SSUOT [VV23]"}
COLORS = {
    "PI-Wavelet": "#1f77b4",
    "PI-Kernel": "#d62734",
    "PB-1NN": "#2ca02c",
    "PB-Kernel": "#ff7f0e",
    "SSUOT": "#9467bd",
}
MARKERS = {"PI-Wavelet": "o", "PI-Kernel": "s", "PB-1NN": "^", "PB-Kernel": "D", "SSUOT": "v"}
FONT = 34
FACE = "#FA967D"
EDGE = "#701700"


def rates(df, metric, dims):
    """Fits the slope of log mean error against log n in each dimension.

    Args:
        df: Results with columns d, n, estimator and the metric.
        metric: Name of the error column.
        dims: Dimensions.

    Returns:
        A DataFrame indexed by estimator with columns slope_d and se_d.
    """
    table = {}
    for est in ESTIMATORS:
        row = {}
        for d in dims:
            g = df[(df["d"] == d) & (df["estimator"] == est)].groupby("n")[metric].mean().sort_index()
            fit = linregress(np.log(g.index.values.astype(float)), np.log(g.values))
            row[f"slope_{d}"], row[f"se_{d}"] = fit.slope, fit.stderr
        table[est] = row
    return pd.DataFrame(table).T


def _style(ax, xlabel, title):
    """Applies the shared axis style.

    Args:
        ax: Matplotlib axes.
        xlabel: Label of the x axis.
        title: Title of the axes.
    """
    ax.set_xlabel(xlabel, fontsize=FONT)
    ax.set_title(title, fontsize=FONT)
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.tick_params(axis="both", which="major", labelsize=13)


def _error_panel(ax, df, metric, d, title, ylabel):
    """Plots the mean error against n with one standard error.

    Args:
        ax: Matplotlib axes.
        df: Results.
        metric: Name of the error column.
        d: Dimension.
        title: Title of the axes.
        ylabel: Label of the y axis.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    for est in ESTIMATORS:
        g = df[(df["estimator"] == est) & (df["d"] == d)].groupby("n")[metric].agg(["mean", "sem"]).sort_index()
        x, y, sem = g.index.values.astype(float), g["mean"].values, g["sem"].values
        ax.plot(x, y, color=COLORS[est], marker=MARKERS[est], markersize=9, linewidth=2.2, label=LABELS.get(est, est))
        ax.fill_between(x, np.maximum(y - sem, 1e-15), y + sem, color=COLORS[est], alpha=0.15, edgecolor="none")
    _style(ax, r"Sample size $n$", title)
    ax.set_xlim(90, 1100)
    ax.set_ylabel(ylabel, fontsize=FONT)
    ax.set_xticks([100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())


def _rate_panel(ax, table, dims, title):
    """Plots the fitted slopes against d with one standard error.

    Args:
        ax: Matplotlib axes.
        table: Output of rates.
        dims: Dimensions.
        title: Title of the axes.
    """
    x = np.asarray(dims, dtype=float)
    for est in ESTIMATORS:
        y = np.array([table.loc[est, f"slope_{d}"] for d in dims], dtype=float)
        se = np.array([table.loc[est, f"se_{d}"] for d in dims], dtype=float)
        ax.plot(x, y, color=COLORS[est], marker=MARKERS[est], markersize=9, linewidth=2.2, label=LABELS.get(est, est))
        ax.fill_between(x, y - se, y + se, color=COLORS[est], alpha=0.15, edgecolor="none")
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        grid,
        np.maximum(-1.0, -4.0 / (2.0 + grid)),
        color="black",
        linestyle="--",
        linewidth=4.0,
        label=r"Upper Bound (Thm. 4)",
    )
    _style(ax, r"Dimension $d$", title)
    ax.set_xlim(0.85, 4.2)
    ax.set_ylabel(r"Slope of $\log$ MSE vs $\log n$", fontsize=FONT)
    ax.set_xticks(list(dims))


def figure(df, path, dims=(1, 2, 3, 4), d_panel=4):
    """Draws the errors at one dimension and the fitted rates in all dimensions.

    Args:
        df: Results with columns d, n, seed, estimator, mse_map and mse_growth.
        path: Output path.
        dims: Dimensions of the rate panels.
        d_panel: Dimension of the error panels.

    Returns:
        A dict mapping each metric to its output of rates.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _error_panel(axes[0, 0], df, "mse_map", d_panel, rf"MSE for $T_0$  ($d={d_panel}$)", r"MSE of $\widehat T$")
    _error_panel(
        axes[0, 1], df, "mse_growth", d_panel, rf"MSE for $\lambda_0$  ($d={d_panel}$)", r"MSE of $\widehat\lambda$"
    )
    tables = {metric: rates(df, metric, dims) for metric in ("mse_map", "mse_growth")}
    _rate_panel(axes[1, 0], tables["mse_map"], dims, r"Learning Rates for $T_0$")
    _rate_panel(axes[1, 1], tables["mse_growth"], dims, r"Learning Rates for $\lambda_0$")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=32,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return tables


def shapes(inputs, fits, path, grid_size=16):
    """Draws the test shapes and their completions.

    Args:
        inputs: Occupancy grids of the test shapes, of shape (k, grid_size**3),
            chairs in the first half and cars in the second half.
        fits: Output of uot.shapes.complete.
        path: Output path.
        grid_size: Number of voxels per axis.
    """
    k = inputs.shape[0]
    fig = plt.figure(figsize=(3 * k, 9))
    top = []
    for i in range(k):
        ax = fig.add_subplot(3, k, i + 1, projection="3d")
        top.append(ax)
        ax.voxels(inputs[i].reshape((grid_size,) * 3) > 0.1, facecolors=FACE, edgecolor=EDGE, linewidth=0.0)
        ax.axis("off")
        for row, name in enumerate(("PB-1NN", "PB-Kernel"), start=1):
            values, lam = fits[name]
            ax = fig.add_subplot(3, k, row * k + i + 1, projection="3d")
            ax.voxels(values[i].reshape((grid_size,) * 3) >= 0.5, facecolors=FACE, edgecolor=EDGE, linewidth=0.0)
            ax.axis("off")
            ax.text2D(
                0.5,
                -0.05,
                rf"$\widehat\lambda \approx {lam[i]:.2f}$",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=40,
            )
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    for height, label in ((0.83, "Input"), (0.50, "PB-1NN"), (0.17, "PB-Kernel")):
        fig.text(0.01, height, label, va="center", ha="left", fontsize=42, rotation=90)
    y = top[0].get_position().y1 + 0.02
    for first, last, label in ((0, k // 2 - 1, "Chair"), (k // 2, k - 1, "Car")):
        x0, x1 = top[first].get_position().x0, top[last].get_position().x1
        for xs, ys in (([x0, x1], [y, y]), ([x0, x0], [y, y - 0.02]), ([x1, x1], [y, y - 0.02])):
            fig.add_artist(Line2D(xs, ys, color="black", lw=2.0, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, y + 0.02, label, ha="center", va="bottom", fontsize=60)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
