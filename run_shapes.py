"""Runs the shape completion experiment and draws its figure."""

import argparse

import numpy as np

from uot import plotting, shapes


def main():
    """Parses the arguments, runs the experiment and prints the growth values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="completion3d")
    parser.add_argument("--out", default="shapes.pdf")
    args = parser.parse_args()
    x, y, test, pca, inputs = shapes.dataset(shapes.download(args.data))
    fits = shapes.complete(x, y, test, pca)
    plotting.shapes(inputs, fits, args.out)
    for name, (_, lam) in fits.items():
        print(name, np.round(lam, 3))


if __name__ == "__main__":
    main()
