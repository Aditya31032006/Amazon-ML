"""
Candidate Generation Coordinator.
Generates candidate sets for Source 1 records and outputs compliant candidate_pairs.tsv.
"""
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from src.blocking.indexer import BlockingIndexer

class CandidateGenerator:
    """Orchestrates candidate generation across Source 1, Source 2, and Source 3."""

    def __init__(self, max_block_size: int = 1500, top_k_per_s1: int = 25):
        self.indexer = BlockingIndexer(max_block_size=max_block_size, top_k_per_s1=top_k_per_s1)

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Builds the multi-key index on (S2 + S3) and retrieves candidate sets for each S1 entity.
        Returns a DataFrame compliant with candidate_pairs.tsv schema:
        ['source1_entity_id', 'candidate_entity_ids']
        """
        # 1. Combine Source 2 and Source 3 into one candidate search pool
        cols = ["entity_id", "country", "core_name", "cleaned_address"]
        df_cand_pool = pd.concat([df_s2[cols], df_s3[cols]], ignore_index=True)

        print(f"  Building inverted index on {len(df_cand_pool):,} candidate records (S2 + S3)...")
        self.indexer.build_index(df_cand_pool)
        pruned_count = self.indexer.prune_oversized_blocks()
        print(f"  Indexed into {len(self.indexer.index):,} unique blocks (pruned {pruned_count} oversized blocks).")

        # 2. Query candidates for each Source 1 entity
        print(f"  Querying candidates for {len(df_s1):,} Source 1 entities...")
        s1_ids = []
        candidate_lists = []

        records = df_s1[["entity_id", "country", "core_name", "cleaned_address"]].itertuples(index=False)
        for s1_id, country, core_name, cleaned_addr in records:
            cands = self.indexer.query_candidates_for_s1(
                country=str(country or ""),
                core_name=str(core_name or ""),
                cleaned_address=str(cleaned_addr or "")
            )
            # Guarantee uniqueness within the list
            cands_unique = list(dict.fromkeys(cands))

            s1_ids.append(s1_id)
            candidate_lists.append(",".join(cands_unique) if cands_unique else "")

        result_df = pd.DataFrame({
            "source1_entity_id": s1_ids,
            "candidate_entity_ids": candidate_lists
        })

        return result_df

    @staticmethod
    def save_candidate_pairs(df_candidates: pd.DataFrame, output_path: Path) -> None:
        """Saves candidate pairs in tab-separated format adhering to competition rules."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_candidates.to_csv(output_path, sep="\t", index=False)
