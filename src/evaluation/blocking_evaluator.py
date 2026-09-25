"""
Blocking Evaluation Service.
Calculates Ground Truth Recall and Candidate Reduction Ratio.
Target: Recall >= 95%
"""
from typing import Dict, Set
import pandas as pd

class BlockingEvaluator:
    """Evaluates candidate generation quality against ground truth matches."""

    @staticmethod
    def evaluate(df_candidates: pd.DataFrame, df_ground_truth: pd.DataFrame) -> Dict[str, float]:
        """
        Computes recall: percentage of true ground truth pairs captured in candidate lists.
        
        Args:
            df_candidates: ['source1_entity_id', 'candidate_entity_ids']
            df_ground_truth: ['source1_entity_id', 'matched_entity_ids']
        """
        # Build candidate lookup: s1_id -> set of candidate IDs
        candidate_map: Dict[str, Set[str]] = {}
        for row in df_candidates.itertuples(index=False):
            s1_id = str(row.source1_entity_id)
            cands = str(row.candidate_entity_ids or "").strip()
            candidate_map[s1_id] = set(cands.split(",")) if cands else set()

        total_true_pairs = 0
        captured_true_pairs = 0
        entities_with_matches = 0
        entities_fully_captured = 0

        total_candidates_generated = 0

        for row in df_ground_truth.itertuples(index=False):
            s1_id = str(row.source1_entity_id)
            gt_matches = str(row.matched_entity_ids or "").strip()
            if not gt_matches:
                # Singleton in ground truth (no true matches)
                continue

            true_ids = set(gt_matches.split(","))
            total_true_pairs += len(true_ids)
            entities_with_matches += 1

            found_cands = candidate_map.get(s1_id, set())
            total_candidates_generated += len(found_cands)

            captured = true_ids.intersection(found_cands)
            captured_true_pairs += len(captured)

            if len(captured) == len(true_ids):
                entities_fully_captured += 1

        recall = (captured_true_pairs / total_true_pairs) * 100 if total_true_pairs > 0 else 0.0
        avg_cands = total_candidates_generated / len(df_candidates) if len(df_candidates) > 0 else 0.0

        return {
            "total_true_pairs": total_true_pairs,
            "captured_true_pairs": captured_true_pairs,
            "blocking_recall_pct": round(recall, 2),
            "avg_candidates_per_entity": round(avg_cands, 2),
            "entities_fully_captured_pct": round((entities_fully_captured / entities_with_matches) * 100, 2) if entities_with_matches > 0 else 0.0
        }
