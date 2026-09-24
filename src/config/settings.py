"""
Settings and Path Configuration for Entity Resolution Pipeline.
"""
from dataclasses import dataclass
from pathlib import Path

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Raw data directory
DATA_RAW_DIR = ROOT_DIR / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"

# Processed / Cleaned data directory
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Output submission directory
OUTPUT_DIR = ROOT_DIR / "output"

@dataclass(frozen=True)
class PathConfig:
    # Raw train files
    train_source1: Path = DATA_RAW_DIR / "train" / "train_source1.tsv"
    train_source2: Path = DATA_RAW_DIR / "train" / "train_source2.tsv"
    train_source3: Path = DATA_RAW_DIR / "train" / "train_source3.tsv"
    train_ground_truth: Path = DATA_RAW_DIR / "train" / "train_ground_truth.tsv"

    # Raw test files
    test_source1: Path = DATA_RAW_DIR / "test" / "test_source1.tsv"
    test_source2: Path = DATA_RAW_DIR / "test" / "test_source2.tsv"
    test_source3: Path = DATA_RAW_DIR / "test" / "test_source3.tsv"

    # Cleaned TSV destinations
    cleaned_train_source1: Path = DATA_PROCESSED_DIR / "train_source1_cleaned.tsv"
    cleaned_train_source2: Path = DATA_PROCESSED_DIR / "train_source2_cleaned.tsv"
    cleaned_train_source3: Path = DATA_PROCESSED_DIR / "train_source3_cleaned.tsv"

    cleaned_test_source1: Path = DATA_PROCESSED_DIR / "test_source1_cleaned.tsv"
    cleaned_test_source2: Path = DATA_PROCESSED_DIR / "test_source2_cleaned.tsv"
    cleaned_test_source3: Path = DATA_PROCESSED_DIR / "test_source3_cleaned.tsv"


    # Final competition output files
    candidate_pairs_tsv: Path = OUTPUT_DIR / "candidate_pairs.tsv"
    matching_results_tsv: Path = OUTPUT_DIR / "matching_results.tsv"

paths = PathConfig()
