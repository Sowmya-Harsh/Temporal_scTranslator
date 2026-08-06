"""
09_prepare_myeloid_geo.py

Purpose: Download the myeloid RNA + protein h5ad pair from GSE210065
(Keenan et al. 2022, the dataset CrossmodalNet benchmarks against) and
verify/adapt its obs columns to match what 06_generalized_finetune.py
expects (a numeric time column).

These files are ALREADY .h5ad (no reshaping from raw matrices needed,
unlike the Kaggle HSPC set) — this script just handles download,
decompression, and a column-compatibility check/fix.

GEO FTP URL pattern for supplementary files: samples are grouped into
directories by GSM accession with the last 3 digits replaced by 'nnn',
e.g. GSM6416060 -> .../samples/GSM6416nnn/GSM6416060/suppl/<filename>

Usage:
    python 09_prepare_myeloid_geo.py --out_dir myeloid_geo_data --inspect_only
    # then, once obs columns are confirmed/fixed:
    python 09_prepare_myeloid_geo.py --out_dir myeloid_geo_data
"""
import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import pandas as pd

FILES = {
    "rna": {
        "gsm": "GSM6416060",
        "filename": "GSM6416060_RNA_myeloid.h5ad.gz",
    },
    "protein": {
        "gsm": "GSM6416061",
        "filename": "GSM6416061_protein_myeloid.h5ad.gz",
    },
}

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples"


def gsm_dir(gsm: str) -> str:
    # e.g. GSM6416060 -> GSM6416nnn
    return gsm[:-3] + "nnn"


def download_file(gsm: str, filename: str, out_dir: Path) -> Path:
    url = f"{GEO_BASE}/{gsm_dir(gsm)}/{gsm}/suppl/{filename}"
    out_path = out_dir / filename
    if out_path.exists():
        print(f"  [already downloaded] {out_path}")
        return out_path
    print(f"  Downloading {url} ...")
    result = subprocess.run(["wget", "-q", "--show-progress", url, "-O", str(out_path)])
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"Download failed for {url}. If wget isn't available, try: "
            f"curl -L -o {out_path} {url}  — or verify the exact download link "
            f"by right-clicking the file on the GEO page and copying the link address, "
            f"since GEO occasionally changes its FTP path structure."
        )
    return out_path


def decompress(gz_path: Path) -> Path:
    out_path = gz_path.with_suffix("")  # strips .gz
    if out_path.exists():
        print(f"  [already decompressed] {out_path}")
        return out_path
    print(f"  Decompressing {gz_path} -> {out_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def inspect_and_fix(rna_path: Path, prot_path: Path, time_col_candidates, out_dir: Path, inspect_only: bool):
    print(f"\nLoading {rna_path} (backed mode, metadata only)...")
    rna = ad.read_h5ad(rna_path, backed="r")
    print(f"  RNA shape: {rna.shape}")
    print(f"  obs columns: {list(rna.obs.columns)}")
    print(f"  var columns: {list(rna.var.columns)}")

    print(f"\nLoading {prot_path} (backed mode, metadata only)...")
    prot = ad.read_h5ad(prot_path, backed="r")
    print(f"  Protein shape: {prot.shape}")
    print(f"  obs columns: {list(prot.obs.columns)}")

    detected_time_col = None
    for cand in time_col_candidates:
        if cand in rna.obs.columns:
            detected_time_col = cand
            break

    if detected_time_col:
        print(f"\nDetected likely time column: '{detected_time_col}'")
        print(f"  Unique values: {sorted(rna.obs[detected_time_col].unique().tolist())}")
    else:
        print(f"\nNo obvious time column found among candidates {time_col_candidates}.")
        print("  Full obs columns again for manual inspection:", list(rna.obs.columns))
        print("  Check for columns like 'week', 'timepoint', 'visit', 'day', 'treatment_time', etc.")

    if inspect_only:
        print("\n[--inspect_only set, stopping here. Re-run without it once time column is confirmed.]")
        return

    if not detected_time_col:
        raise ValueError(
            "Cannot proceed without a confirmed time column. Re-run with --inspect_only, "
            "identify the correct obs column name, then pass it via --time_col explicitly."
        )

    # Standardize to 'time_min' as expected by 06_generalized_finetune.py's default.
    # NOTE: 'HD' (healthy donor) is excluded here deliberately -- it's a different
    # patient population (healthy vs. biliary cancer), not an earlier timepoint on
    # the same disease/treatment trajectory, so including it in an ordinal time
    # sequence alongside T01/T02/T03 would corrupt the OT cross-timepoint matching
    # (see thesis discussion, Section 6.8, on why this external dataset's cross-cell
    # OT-based forecasting task is NOT what CrossmodalNet's own paper validates --
    # CrossmodalNet predicts each cell's own protein from its own RNA at the SAME
    # timepoint, using time only as a covariate; it never does true future-state
    # forecasting across destructively-sampled populations the way this thesis does).
    print(f"\nWriting standardized copies with 'time_min' column (from '{detected_time_col}')...")
    rna_full = ad.read_h5ad(rna_path)
    prot_full = ad.read_h5ad(prot_path)

    raw_vals = rna_full.obs[detected_time_col]
    if detected_time_col == "timepoint" and set(raw_vals.unique()) >= {"T01", "T02", "T03"}:
        print("  Excluding 'HD' (healthy donor, different population) from the temporal sequence.")
        keep_mask = raw_vals.isin(["T01", "T02", "T03"])
        n_before = rna_full.n_obs
        rna_full = rna_full[keep_mask].copy()
        prot_full = prot_full[keep_mask].copy()
        print(f"  Kept {rna_full.n_obs}/{n_before} cells (T01/T02/T03 only)")
        # T01/T02/T03 = before, 1 week after, and 3 WEEKS after anti-PD-1 treatment
        # (per Keenan et al. 2022, Cell Reports: "PBMCs obtained from BTC patients
        # before, 1 week after, and 3 weeks after anti-PD-1 treatment") -- NOT
        # evenly spaced: a 1-week gap T01->T02, then a 2-week gap T02->T03.
        # Converting to actual minutes (not ordinal 0/1/2) so delta_t correctly
        # reflects this asymmetry, since it directly feeds the FiLM time-conditioning
        # and the OT matching's biological interpretation of "how much time passed."
        WEEK_TO_MIN = 7 * 24 * 60  # 10080
        mapping = {"T01": 0.0, "T02": 1 * WEEK_TO_MIN, "T03": 3 * WEEK_TO_MIN}
        print(f"  Mapping timepoints to minutes (pre-treatment / 1wk / 3wk): {mapping}")
        rna_full.obs["time_min"] = rna_full.obs[detected_time_col].map(mapping).astype(float)
    elif pd.api.types.is_numeric_dtype(raw_vals):
        rna_full.obs["time_min"] = raw_vals.astype(float)
    else:
        unique_sorted = sorted(raw_vals.unique().tolist())
        mapping = {v: i for i, v in enumerate(unique_sorted)}
        print(f"  Non-numeric time column — mapping categories to ordinal values: {mapping}")
        rna_full.obs["time_min"] = raw_vals.map(mapping).astype(float)

    prot_full.obs["time_min"] = rna_full.obs["time_min"].reindex(prot_full.obs.index)
    if prot_full.obs["time_min"].isna().any():
        print("  WARNING: some protein cells didn't get a time_min value via index alignment — "
              "check whether RNA/protein cells share the same obs index (barcode) before proceeding.")

    out_rna = out_dir / "myeloid_RNA_standardized.h5ad"
    out_prot = out_dir / "myeloid_Protein_standardized.h5ad"
    rna_full.write_h5ad(out_rna)
    prot_full.write_h5ad(out_prot)
    print(f"\nSaved {out_rna}")
    print(f"Saved {out_prot}")
    print(f"\nNext step:")
    print(f"  python 06_generalized_finetune.py --rna_path {out_rna} --prot_path {out_prot} "
          f"--time_col time_min --checkpoint <your_checkpoint.pt> --out_dir myeloid_finetune_results")


def main(out_dir, time_col, inspect_only):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading myeloid RNA + protein h5ad files from GEO...")
    rna_gz = download_file(FILES["rna"]["gsm"], FILES["rna"]["filename"], out_dir)
    prot_gz = download_file(FILES["protein"]["gsm"], FILES["protein"]["filename"], out_dir)

    rna_path = decompress(rna_gz)
    prot_path = decompress(prot_gz)

    time_col_candidates = [time_col] if time_col else [
        "week", "Week", "timepoint", "Timepoint", "time", "Time",
        "visit", "Visit", "day", "Day", "treatment_time",
    ]
    inspect_and_fix(rna_path, prot_path, time_col_candidates, out_dir, inspect_only)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="myeloid_geo_data")
    parser.add_argument("--time_col", type=str, default=None,
                         help="Force a specific obs column name for time, skipping auto-detection")
    parser.add_argument("--inspect_only", action="store_true",
                         help="Download + decompress + print obs columns, then stop (no writing standardized files)")
    args = parser.parse_args()
    main(args.out_dir, args.time_col, args.inspect_only)