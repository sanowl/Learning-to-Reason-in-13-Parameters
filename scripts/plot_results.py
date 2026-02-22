#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregated benchmark/sweep results")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--x", required=True, help="Column for x axis")
    parser.add_argument("--y", required=True, help="Column for y axis")
    parser.add_argument("--hue", default=None, help="Optional grouping column")
    parser.add_argument("--title", default="Result Plot", help="Figure title")
    parser.add_argument("--output", required=True, help="Output image path (.png)")
    parser.add_argument("--x-log", action="store_true", help="Use log scale for x")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.csv).resolve())
    if not rows:
        raise ValueError("No rows in input CSV")

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get(args.x) in {None, ""} or row.get(args.y) in {None, ""}:
            continue
        key = row.get(args.hue, "all") if args.hue is not None else "all"
        grouped[str(key)].append((float(row[args.x]), float(row[args.y])))

    plt.figure(figsize=(9, 5))
    for key, points in sorted(grouped.items(), key=lambda item: item[0]):
        points.sort(key=lambda item: item[0])
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, marker="o", label=key)

    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(args.title)
    if args.x_log:
        plt.xscale("log")
    if args.hue is not None:
        plt.legend(title=args.hue)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    print("[Done]", output)


if __name__ == "__main__":
    main()
