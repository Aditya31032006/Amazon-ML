"""
Phase 1 Execution Pipeline: Data Cleaning & Normalization.
Reads raw TSV files, applies TextCleaner & DataFrameNormalizer,
and saves the cleaned, optimized tables to Parquet format.
"""
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config.settings import paths
from src.data.loader import DataLoader
from src.preprocessing.normalizer import DataFrameNormalizer

def process_file(input_path: Path, output_path: Path, sample_size: int = None):
    """Loads, cleans, and saves a single dataset."""
    print(f"\n[Processing] {input_path.name}...")
    start_time = time.time()

    if not input_path.exists():
        print(f"  [ERROR] File does not exist: {input_path}")
        return

    # 1. Load TSV
    df_raw = DataLoader.read_tsv(input_path, nrows=sample_size)
    load_time = time.time() - start_time
    print(f"  Loaded {len(df_raw):,} records in {load_time:.2f}s")

    # 2. Normalize and Clean
    clean_start = time.time()
    df_cleaned = DataFrameNormalizer.normalize_dataframe(df_raw)
    clean_time = time.time() - clean_start
    print(f"  Normalized text & extracted features in {clean_time:.2f}s")

    # 3. Report Missingness
    name_miss_pct = (df_cleaned['name_missing'].sum() / len(df_cleaned)) * 100
    addr_miss_pct = (df_cleaned['address_missing'].sum() / len(df_cleaned)) * 100
    print(f"  Stats: Missing Names: {name_miss_pct:.2f}% | Missing Addresses: {addr_miss_pct:.2f}%")

    # 4. Save to TSV
    save_start = time.time()
    DataLoader.save_tsv(df_cleaned, output_path)
    save_time = time.time() - save_start
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved TSV to {output_path.name} ({size_mb:.2f} MB) in {save_time:.2f}s")


def run_phase1(sample_size: int = None):
    """Executes Phase 1 across all train and test sources."""
    print("=" * 60)
    print("STARTING PHASE 1: TEXT CLEANING & NORMALIZATION")
    if sample_size:
        print(f"RUNNING IN SAMPLE MODE: {sample_size} rows per file")
    print("=" * 60)

    total_start = time.time()

    # Targets to process
    tasks = [
        (paths.train_source1, paths.cleaned_train_source1),
        (paths.train_source2, paths.cleaned_train_source2),
        (paths.train_source3, paths.cleaned_train_source3),
        (paths.test_source1, paths.cleaned_test_source1),
        (paths.test_source2, paths.cleaned_test_source2),
        (paths.test_source3, paths.cleaned_test_source3),
    ]

    for raw_p, out_p in tasks:
        process_file(raw_p, out_p, sample_size=sample_size)

    total_duration = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"PHASE 1 COMPLETE! Total time: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print(f"Cleaned datasets saved in: {paths.cleaned_train_source1.parent}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 Text Cleaning")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for testing (e.g. 5000)")
    args = parser.parse_args()

    run_phase1(sample_size=args.sample)
