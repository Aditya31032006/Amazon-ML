"""
Multi-Key Inverted Index Engine for Candidate Generation.
Indexes Source 2 and Source 3 records into hash buckets by blocking keys.
Retrieves candidate records for Source 1 queries with block size limits.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from src.blocking.keys import BlockingKeyExtractor

class BlockingIndexer:
    """
    Inverted Index that maps blocking keys to sets of candidate record IDs.
    """
    def __init__(self, max_block_size: int = 1500, top_k_per_s1: int = 25):
        """
        Args:
            max_block_size: Maximum records allowed in a single block.
                            Blocks exceeding this limit are skipped to prevent
                            oversized cross-product explosions on generic terms.
            top_k_per_s1: Maximum candidate pairs to retain per Source 1 entity.
        """
        self.max_block_size = max_block_size
        self.top_k_per_s1 = top_k_per_s1
        self.index: Dict[str, List[str]] = defaultdict(list)

    def build_index(self, df_candidates: pd.DataFrame) -> None:
        """
        Indexes all candidate records (Source 2 and Source 3 combined).
        Expects columns: ['entity_id', 'country', 'core_name', 'cleaned_address']
        """
        records = df_candidates[["entity_id", "country", "core_name", "cleaned_address"]].itertuples(index=False)
        for entity_id, country, core_name, cleaned_addr in records:
            keys = BlockingKeyExtractor.extract_keys(
                country=str(country or ""),
                core_name=str(core_name or ""),
                cleaned_address=str(cleaned_addr or "")
            )
            for k in keys:
                self.index[k].append(entity_id)

    def prune_oversized_blocks(self) -> int:
        """Removes blocks that exceed max_block_size to protect memory and speed."""
        oversized_keys = [k for k, v in self.index.items() if len(v) > self.max_block_size]
        for k in oversized_keys:
            del self.index[k]
        return len(oversized_keys)

    def query_candidates_for_s1(self, country: str, core_name: str, cleaned_address: str) -> List[str]:
        """
        Retrieves candidate entity IDs for a single Source 1 record.
        Ranks candidates by the number of shared blocking keys.
        """
        keys = BlockingKeyExtractor.extract_keys(
            country=country,
            core_name=core_name,
            cleaned_address=cleaned_address
        )

        # Count candidate frequency across all matched keys
        cand_counts: Dict[str, int] = defaultdict(int)
        for k in keys:
            if k in self.index:
                cand_list = self.index[k]
                for cand_id in cand_list:
                    cand_counts[cand_id] += 1

        if not cand_counts:
            return []

        # Sort candidates descending by key overlap count
        sorted_candidates = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)
        return [cand_id for cand_id, _ in sorted_candidates[:self.top_k_per_s1]]
