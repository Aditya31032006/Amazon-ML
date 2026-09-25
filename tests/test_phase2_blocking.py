"""
Unit tests for Phase 2: Multi-Key Candidate Generation & Blocking.
Verifies Soundex phonetic encoding, address anchor extraction,
and multi-key inverted index matching.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.blocking.keys import compute_soundex, extract_address_anchor, BlockingKeyExtractor
from src.blocking.indexer import BlockingIndexer
import pandas as pd

def test_soundex():
    print("=== Testing Soundex Phonetic Code ===")
    s1 = compute_soundex("Robotics")
    s2 = compute_soundex("Robotix")
    print(f"Soundex('Robotics'): {s1}")
    print(f"Soundex('Robotix'):  {s2}")
    assert s1 == s2 == "R132", f"Expected both to be R132, got {s1} and {s2}"
    print("[PASS] Soundex successfully matched spelling variation!")

def test_address_anchor():
    print("\n=== Testing Address Anchor Extractor ===")
    cases = [
        ("500 market street san jose", "500_market"),
        ("1795 westchester drive high point nc", "1795_westchester"),
        ("109 2 floor 16 cross road jp nagar bangalore", "109_cross"),
    ]
    for addr, exp in cases:
        anchor = extract_address_anchor(addr)
        print(f"Address: '{addr}' -> Anchor: '{anchor}' (Expected: '{exp}')")
        assert anchor == exp, f"Mismatch for '{addr}': got '{anchor}', expected '{exp}'"
    print("[PASS] Address anchors extracted accurately!")

def test_indexer_retrieval():
    print("\n=== Testing Inverted Index Retrieval ===")
    df_cands = pd.DataFrame([
        {
            "entity_id": "S2-118820",
            "country": "US",
            "core_name": "acme robotics",
            "cleaned_address": "500 market street san jose"
        },
        {
            "entity_id": "S2-540221",
            "country": "US",
            "core_name": "acme robotix",
            "cleaned_address": "12 elm road san jose"
        },
        {
            "entity_id": "S3-999999",
            "country": "INDIA",
            "core_name": "acme robotics",
            "cleaned_address": "500 market street mumbai"
        }
    ])

    indexer = BlockingIndexer(max_block_size=500, top_k_per_s1=10)
    indexer.build_index(df_cands)

    # Query for S1 entity in US
    cands = indexer.query_candidates_for_s1(
        country="US",
        core_name="acme robotics",
        cleaned_address="500 market street san jose"
    )

    print(f"Queried candidates for S1 (US, 'acme robotics', '500 market street'): {cands}")
    assert "S2-118820" in cands, "S2-118820 should be matched on name and address anchor"
    assert "S2-540221" in cands, "S2-540221 should be matched on phonetic and name prefix"
    assert "S3-999999" not in cands, "S3-999999 (INDIA) should be hard-partitioned out for US entity"

    print("[PASS] Inverted index retrieval & hard country partitioning verified!")

if __name__ == "__main__":
    test_soundex()
    test_address_anchor()
    test_indexer_retrieval()
    print("\n[ALL PHASE 2 TESTS PASSED SUCCESSFULLY!]")
