"""Runs the simulation study and writes the errors to a CSV file."""

import argparse

from uot.benchmark import DIMENSIONS, run


def main():
    """Parses the arguments and runs the study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=list(DIMENSIONS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--out", default="simulation.csv")
    args = parser.parse_args()
    run(args.dims, args.seeds, args.out)


if __name__ == "__main__":
    main()
