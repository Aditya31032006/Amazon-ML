"""
Phase 2 Execution Pipeline: Multi-Key Candidate Generation & Blocking.
Builds the multi-key index, retrieves candidates, saves candidate_pairs.tsv,
and measures Ground-Truth Recall on training data.
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
from src.blocking.candidate_generator import CandidateGenerator
from src.evaluation.blocking_evaluator import BlockingEvaluator

def run_phase2(mode: str = "train", sample_size: int = None):
    print("=" * 60)
    print(f"STARTING PHASE 2: CANDIDATE GENERATION ({mode.upper()} MODE)")
    if sample_size:
        print(f"SAMPLE MODE: {sample_size} records per source")
    print("=" * 60)

    total_start = time.time()

    # 1. Select inputs
    if mode == "train":
        s1_path = paths.cleaned_train_source1
        s2_path = paths.cleaned_train_source2
        s3_path = paths.cleaned_train_source3
    else:
        s1_path = paths.cleaned_test_source1
        s2_path = paths.cleaned_test_source2
        s3_path = paths.cleaned_test_source3

    print(f"\n[1/3] Loading cleaned sources...")
    df_s1 = DataLoader.read_tsv(s1_path, nrows=sample_size)
    df_s2 = DataLoader.read_tsv(s2_path, nrows=sample_size)
    df_s3 = DataLoader.read_tsv(s3_path, nrows=sample_size)
    print(f"  Loaded S1: {len(df_s1):,} | S2: {len(df_s2):,} | S3: {len(df_s3):,}")

    # 2. Candidate Generation
    print(f"\n[2/3] Generating multi-key candidate pools...")
    generator = CandidateGenerator(max_block_size=1500, top_k_per_s1=25)
    cand_start = time.time()
    df_candidates = generator.generate_candidates(df_s1, df_s2, df_s3)
    cand_time = time.time() - cand_start
    print(f"  Generated candidate sets in {cand_time:.2f}s")

    # 3. Save Candidate Pairs
    output_path = paths.candidate_pairs_tsv
    CandidateGenerator.save_candidate_pairs(df_candidates, output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved candidate pairs to {output_path} ({size_mb:.2f} MB)")

    # 4. Evaluate against Ground Truth (in train mode)
    if mode == "train" and paths.train_ground_truth.exists():
        print(f"\n[3/3] Evaluating Blocking Recall against Ground Truth...")
        df_gt = DataLoader.read_tsv(paths.train_ground_truth, nrows=sample_size)
        metrics = BlockingEvaluator.evaluate(df_candidates, df_gt)
        print("-" * 50)
        print(f"  Total True Matches:            {metrics['total_true_pairs']:,}")
        print(f"  Captured True Matches:         {metrics['captured_true_pairs']:,}")
        print(f"  BLOCKING RECALL:               {metrics['blocking_recall_pct']:.2f}% (Target >= 95%)")
        print(f"  Avg Candidates per S1 Entity:  {metrics['avg_candidates_per_entity']:.1f}")
        print(f"  Fully Captured Entities:       {metrics['entities_fully_captured_pct']:.2f}%")
        print("-" * 50)

    total_duration = time.time() - total_start
    print(f"\nPHASE 2 FINISHED in {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 Candidate Blocking")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Run on train or test data")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for testing")
    args = parser.parse_args()

    run_phase2(mode=args.mode, sample_size=args.sample)
