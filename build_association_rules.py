"""
Build per-store association rules from data/train.csv without touching source CSVs.

- Reads train.csv in chunks, optionally filtering to a store list (--stores).
- Builds baskets per (store, date) and computes pairwise rules A -> B.
- Writes one CSV per store into data/assoc_rules/assoc_rules_store_<store>.csv.
- Adds family names if data/products.csv is present.

Defaults (tweak with flags):
    --min-support     0.05   # 5% of baskets (per store)
    --min-confidence  0.30   # 30%
    --chunksize       500000

Examples:
    python build_association_rules.py --stores 25 --min-support 0.02 --min-confidence 0.2
    python build_association_rules.py --stores 5,7,25 --min-support 0.05
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
PRODUCTS_PATH = BASE_DIR / "data" / "products.csv"
OUTPUT_DIR = BASE_DIR / "data" / "assoc_rules"


def read_products() -> Dict[int, str]:
    """Return item_nbr -> family mapping if products.csv exists."""
    if not PRODUCTS_PATH.exists():
        return {}
    df = pd.read_csv(PRODUCTS_PATH, usecols=["item_nbr", "family"])
    return dict(zip(df["item_nbr"].astype(int), df["family"].astype(str)))


def load_baskets(
    chunksize: int = 500_000, stores: Set[int] | None = None
) -> Dict[int, Dict[str, Set[int]]]:
    """
    Stream train.csv and build baskets per store/date.

    Returns:
        {store_nbr: {date: set(item_nbr, ...)}}
    """
    baskets: Dict[int, Dict[str, Set[int]]] = {}
    usecols = ["date", "store_nbr", "item_nbr", "unit_sales"]
    for chunk in pd.read_csv(TRAIN_PATH, usecols=usecols, chunksize=chunksize):
        if stores:
            chunk = chunk[chunk["store_nbr"].isin(stores)]
            if chunk.empty:
                continue
        # keep only positive sales
        chunk = chunk[chunk["unit_sales"] > 0]
        grouped = (
            chunk.groupby(["store_nbr", "date"])["item_nbr"]
            .apply(lambda s: set(s.astype(int)))
            .items()
        )
        for (store, date), items in grouped:
            store = int(store)
            if store not in baskets:
                baskets[store] = {}
            if date not in baskets[store]:
                baskets[store][date] = set()
            baskets[store][date].update(items)
    return baskets


def filter_frequent_items(
    baskets: Dict[str, Set[int]], min_support: float
) -> Tuple[Counter, Set[int], int]:
    """Count items and return (counts, frequent_items, basket_total)."""
    counts: Counter[int] = Counter()
    for items in baskets.values():
        counts.update(items)
    basket_total = len(baskets)
    threshold = max(1, int(min_support * basket_total))
    frequent = {item for item, cnt in counts.items() if cnt >= threshold}
    return counts, frequent, basket_total


def generate_rules_for_store(
    store: int,
    baskets: Dict[str, Set[int]],
    min_support: float,
    min_confidence: float,
) -> pd.DataFrame:
    """Compute pairwise rules A -> B for a single store."""
    item_counts, frequent_items, basket_total = filter_frequent_items(
        baskets, min_support
    )
    if not frequent_items:
        return pd.DataFrame()

    pair_counts: Counter[Tuple[int, int]] = Counter()
    for items in baskets.values():
        filtered = sorted(i for i in items if i in frequent_items)
        for a, b in combinations(filtered, 2):
            pair_counts[(a, b)] += 1

    rows = []
    for (a, b), cnt in pair_counts.items():
        support = cnt / basket_total
        for lhs, rhs in [(a, b), (b, a)]:
            lhs_cnt = item_counts[lhs]
            rhs_cnt = item_counts[rhs]
            confidence = support / lhs_cnt if lhs_cnt else 0.0
            lift = confidence / (rhs_cnt / basket_total) if rhs_cnt else 0.0
            if support >= min_support and confidence >= min_confidence:
                rows.append(
                    {
                        "store_nbr": store,
                        "lhs_item": lhs,
                        "rhs_item": rhs,
                        "support": round(support, 6),
                        "confidence": round(confidence, 6),
                        "lift": round(lift, 6),
                        "support_count": cnt,
                        "lhs_count": lhs_cnt,
                        "rhs_count": rhs_cnt,
                        "basket_total": basket_total,
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.sort_values(["lift", "confidence", "support"], ascending=False, inplace=True)
    return df


def attach_names(df: pd.DataFrame, product_map: Dict[int, str]) -> pd.DataFrame:
    """Add family names if available."""
    if not product_map or df.empty:
        return df
    df["lhs_family"] = df["lhs_item"].map(product_map)
    df["rhs_family"] = df["rhs_item"].map(product_map)
    return df


def write_rules_per_store(
    baskets_per_store: Dict[int, Dict[str, Set[int]]],
    min_support: float,
    min_confidence: float,
) -> None:
    product_map = read_products()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for store, baskets in baskets_per_store.items():
        df = generate_rules_for_store(store, baskets, min_support, min_confidence)
        if df.empty:
            print(f"[store {store}] no rules found (support/conf threshold too high).")
            continue
        df = attach_names(df, product_map)
        out_path = OUTPUT_DIR / f"assoc_rules_store_{store}.csv"
        df.to_csv(out_path, index=False)
        print(f"[store {store}] wrote {len(df)} rules -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-store association rules from data/train.csv"
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.05,
        help="Minimum support per store (fraction of baskets) [default: 0.05]",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
        help="Minimum confidence [default: 0.30]",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Chunk size when reading train.csv [default: 500000]",
    )
    parser.add_argument(
        "--stores",
        type=str,
        default="",
        help="Comma-separated store_nbr list to process (default: all stores)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Reading baskets from {TRAIN_PATH} with chunksize={args.chunksize}, "
        f"min_support={args.min_support}, min_confidence={args.min_confidence}"
    )
    store_filter: Set[int] | None = None
    if args.stores:
        store_filter = {int(s.strip()) for s in args.stores.split(',') if s.strip()}
        print(f"Processing only stores: {sorted(store_filter)}")

    baskets_per_store = load_baskets(chunksize=args.chunksize, stores=store_filter)
    print(
        f"Loaded baskets for {len(baskets_per_store)} stores. "
        f"Now generating rules..."
    )
    write_rules_per_store(
        baskets_per_store,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
    )
    print(f"Done. Files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
