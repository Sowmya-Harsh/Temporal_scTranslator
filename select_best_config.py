"""
select_best_config.py
────────────────────────
Reads full_sweep_pooled.csv (from combine_full_sweep.py) and selects
the best config using VALIDATION R only -- never test. Test R for the
selected config is reported afterward, purely for the final headline
number, exactly once, on the config already chosen by val.

This is the step that was missing: the earlier console summary in
combine_full_sweep.py ranked by test R for a quick look, which is
fine to eyeball but must NOT be used to pick which config goes in the
thesis. This script is the one whose output is trustworthy for that.

Usage:
    python select_best_config.py --csv new_results/full_sweep_combined/full_sweep_pooled.csv
"""
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    val_ranked = val.sort_values("mean_pearson_r", ascending=False).reset_index(drop=True)
    print(f"{'='*70}\nTOP {args.top_n} CONFIGS BY VALIDATION R (the correct selection criterion)\n{'='*70}")
    print(val_ranked[["tag", "mean_pearson_r", "mean_ccc"]].head(args.top_n).to_string(index=False))

    best_tag = val_ranked.iloc[0]["tag"]
    best_val_r = val_ranked.iloc[0]["mean_pearson_r"]
    best_val_ccc = val_ranked.iloc[0]["mean_ccc"]

    test_row = test[test["tag"] == best_tag]
    if len(test_row) == 0:
        print(f"\nWARNING: no test row found for selected config '{best_tag}'")
        return
    test_r = test_row.iloc[0]["mean_pearson_r"]
    test_ccc = test_row.iloc[0]["mean_ccc"]

    print(f"\n{'='*70}\nSELECTED CONFIG (chosen on val, reported on test)\n{'='*70}")
    print(f"Config: {best_tag}")
    print(f"  Validation: R={best_val_r:.4f} CCC={best_val_ccc:.4f}  <- used to SELECT")
    print(f"  Test:       R={test_r:.4f} CCC={test_ccc:.4f}  <- used to REPORT")

    # Sanity check: is this the same config the naive test-ranking would have picked?
    test_ranked = test.sort_values("mean_pearson_r", ascending=False).reset_index(drop=True)
    naive_best_tag = test_ranked.iloc[0]["tag"]
    if naive_best_tag != best_tag:
        print(f"\nNOTE: naive test-ranking would have picked a DIFFERENT config "
              f"('{naive_best_tag}', test R={test_ranked.iloc[0]['mean_pearson_r']:.4f}). "
              f"This is exactly the config-selection-leakage risk val-based selection avoids.")
    else:
        print(f"\nNOTE: val-based and naive test-based selection agree on this config "
              f"('{best_tag}') -- no discrepancy here, but val selection is still the "
              f"methodologically correct choice to report as the process used.")


if __name__ == "__main__":
    main()
