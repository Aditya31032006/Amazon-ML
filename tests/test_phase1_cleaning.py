"""
Unit test to verify Phase 1 TextCleaner behavior on edge cases:
- Mojibake encoding repair
- Transliteration of Indic scripts and Indic legal suffix removal
- Legal suffix canonicalization and core name extraction
- Duplicate suffix collapse (e.g. "pvt ltd ltd" -> "pvt ltd", "llc llc" -> "llc")
- Apostrophe-s handling (e.g. "candy's" -> "candys")
- Country-aware address standardization (e.g. TN -> Tennessee in US vs Tamil Nadu in India)
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.preprocessing.text_cleaner import TextCleaner

def test_cleaning_examples():
    test_cases = [
        # (raw_name, expected_cleaned, expected_core)
        ("Acme Robotics Inc.", "acme robotics inc", "acme robotics"),
        ("Acme Robotics Incorporated", "acme robotics inc", "acme robotics"),
        ("Delta Foods Co - 2067865001", "delta foods co", "delta foods"),
        ("Consulting Nyasa Nursing Private Limited", "consulting nyasa nursing pvt ltd", "consulting nyasa nursing"),
        ("Shree Infracon Private Ltd", "shree infracon pvt ltd", "shree infracon"),
        ("Bright Cafe LLC #8", "bright cafe llc", "bright cafe"),
        ("Olszewski Holding Company LLC LLC", "olszewski holdings co llc", "olszewski"),
        ("CANDY'S-SERVICES", "candys services", "candys"),
        ("www.wilfordhancock.com", "wilfordhancock", "wilfordhancock"),
        ("B+ Retail Inc", "b plus retail inc", "b plus retail"),
        ("SHIVSHAKTI CORP | www.shivshakti.com", "shivshakti corp", "shivshakti"),
        ("Café Green", "cafe green", "cafe green"),
        # Indic legal suffix tests
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "raam maarketting pvt ltd", "raam maarketting"),
        ("आदित्य प्रॉपर्टीज एलएलपी", "aadity pronprttiij llp", "aadity pronprttiij"),
        ("குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "kulloopl picinnns pvt ltd", "kulloopl picinnns"),
    ]

    print("=== Testing Business Name Cleaner ===")
    for raw, exp_clean, exp_core in test_cases:
        cleaned, core = TextCleaner.clean_name(raw)
        print(f"RAW:     {raw}")
        print(f"CLEANED: {cleaned} (Expected: {exp_clean})")
        print(f"CORE:    {core} (Expected: {exp_core})")
        print("-" * 50)
        assert cleaned == exp_clean, f"Cleaned mismatch for '{raw}': got '{cleaned}', expected '{exp_clean}'"
        assert core == exp_core, f"Core mismatch for '{raw}': got '{core}', expected '{exp_core}'"

    print("\n=== Testing Country-Aware Address Cleaner ===")
    addr_cases = [
        # (raw_address, country, expected_cleaned)
        ("1418 MEADOWBROOK DR, JOHNSON CITY, TN", "US", "1418 meadowbrook drive johnson city tennessee"),
        ("HIGHWAY 49, ASHLAND CITY, TN", "US", "highway 49 ashland city tennessee"),
        ("C-21 - S, AMBATTUR, CHENNAI, TN", "INDIA", "c 21 s ambattur chennai tamil nadu"),
        ("Mulund Goreagon Link Rd, Near Fortis, Mumbai, MH", "INDIA", "mulund goreagon link road near fortis mumbai maharashtra"),
        ("294 Meadowcreek Drive, Unit Unit 2, Pewaukee, WI", "US", "294 meadowcreek drive unit 2 pewaukee wisconsin"),
    ]
    for raw_addr, country, exp_addr in addr_cases:
        cleaned_addr = TextCleaner.clean_address(raw_addr, country=country)
        print(f"RAW [{country}]: {raw_addr}")
        print(f"CLEANED:        {cleaned_addr} (Expected: {exp_addr})")
        print("-" * 50)
        assert cleaned_addr == exp_addr, f"Address mismatch for '{raw_addr}': got '{cleaned_addr}', expected '{exp_addr}'"

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    test_cleaning_examples()
