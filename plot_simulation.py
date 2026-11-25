"""Draws the simulation figure from the CSV file of run_simulation.py."""

import argparse

import pandas as pd

from uot.plotting import figure


def main():
    """Parses the arguments, draws the figure and prints the fitted rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--out", default="simulation.pdf")
    args = parser.parse_args()
    for metric, table in figure(pd.read_csv(args.csv), args.out).items():
        print(metric)
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
