# Assosation Rules Toolkit

Scripts to compute association rules from the provided retail dataset without modifying the original CSVs in `data/`.

- `build_association_rules.py` — stream `data/train.csv` (optionally a store list) and write per‑store rule files to `data/assoc_rules/assoc_rules_store_<store>.csv`.
- `view_store_rules.py` — inspect a single store’s rules with filters (support/confidence/lift).
- `bdm-week2.ipynb` — original notebook used for market-basket exploration (source: [Kaggle notebook by plenoi](https://www.kaggle.com/code/plenoi/bdm-week2/notebook#Association-Rules)).

## Usage

Create per-store rules (example for store 25):
```bash
python build_association_rules.py --stores 25 --min-support 0.02 --min-confidence 0.20
# outputs: data/assoc_rules/assoc_rules_store_25.csv
```

View rules for a store:
```bash
python view_store_rules.py --store 25 --top 15 --min-lift 1.1
```

Notes:
- Original `data/*.csv` files are never modified; results are written to `data/assoc_rules/`.
- Lower `--min-support`/`--min-confidence` if a small store yields no rules.
