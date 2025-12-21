"""
View association rules for a chosen store.

Reads the per-store CSV produced by build_association_rules.py and prints the
strongest rules after optional filtering.

Example:
    python view_store_rules.py --store 25 --top 15 --min-lift 1.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DEFAULT_RULES_DIR = BASE_DIR / "data" / "assoc_rules"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View association rules for a store")
    p.add_argument("--store", type=int, required=True, help="store_nbr to inspect")
    p.add_argument(
        "--rules-dir",
        type=Path,
        default=DEFAULT_RULES_DIR,
        help="Directory with assoc_rules_store_<store>.csv files",
    )
    p.add_argument("--top", type=int, default=20, help="How many rules to show")
    p.add_argument("--min-support", type=float, default=0.0, help="Filter by support")
    p.add_argument("--min-confidence", type=float, default=0.0, help="Filter by confidence")
    p.add_argument("--min-lift", type=float, default=0.0, help="Filter by lift")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = args.rules_dir / f"assoc_rules_store_{args.store}.csv"
    if not path.exists():
        raise SystemExit(
            f"Rules not found for store {args.store}: {path}\n"
            "Run build_association_rules.py first to generate per-store CSVs."
        )

    df = pd.read_csv(path)
    if args.min_support:
        df = df[df["support"] >= args.min_support]
    if args.min_confidence:
        df = df[df["confidence"] >= args.min_confidence]
    if args.min_lift:
        df = df[df["lift"] >= args.min_lift]

    if df.empty:
        print("No rules after filtering. Try lowering thresholds.")
        return

    cols = [
        "lhs_item",
        "rhs_item",
        "lhs_family" if "lhs_family" in df.columns else None,
        "rhs_family" if "rhs_family" in df.columns else None,
        "support",
        "confidence",
        "lift",
    ]
    cols = [c for c in cols if c]
    df = df.sort_values(["lift", "confidence", "support"], ascending=False)
    print(f"Store {args.store} — top {args.top} rules (sorted by lift):")
    print(df[cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
