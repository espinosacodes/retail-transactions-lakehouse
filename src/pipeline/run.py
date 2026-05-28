"""CLI: orquesta bronze → silver → gold."""
from __future__ import annotations

import argparse
import time

from . import bronze, gold, silver


STEPS = {
    "bronze": bronze.run,
    "silver": silver.run,
    "gold": gold.run,
}


def main():
    parser = argparse.ArgumentParser(description="Supermarket pipeline orchestrator")
    parser.add_argument("--step", choices=list(STEPS) + ["all"], default="all")
    args = parser.parse_args()

    order = [args.step] if args.step != "all" else list(STEPS)
    for name in order:
        t0 = time.perf_counter()
        print(f"\n=== running step: {name} ===")
        STEPS[name]()
        print(f"=== {name} done in {time.perf_counter() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
