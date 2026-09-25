"""
Blocking Key Generators for Multi-Key Candidate Generation.
Extracts complementary blocking keys from normalized records:
1. Name Prefix Key (e.g., US::np_acme)
2. Phonetic Soundex Key (e.g., US::ph_A250)
3. Address Anchor Key (e.g., US::addr_500_market)
4. Distinctive Name Tokens (e.g., US::tok_nyasa)
5. Name Bigram / First-Two-Words Key (e.g., US::n2_acme_robotics)

All keys are prefixed with country to enforce hard country partitioning.
"""
import re
from typing import List, Set

# Common English and business stop words that should NOT form single-token blocks
COMMON_STOP_WORDS = {
    "and", "the", "for", "with", "all", "new", "one", "top", "pro",
    "services", "service", "solutions", "solution", "enterprises",
    "enterprise", "traders", "trading", "group", "holdings", "holding",
    "center", "centre", "international", "global", "national", "associates",
    "agency", "management", "consulting", "industries", "industry",
    "products", "commercial", "retail", "general", "marketing"
}

def compute_soundex(word: str) -> str:
    """
    Computes standard American Soundex phonetic code for a word.
    Maps similar-sounding consonants to identical digits:
    Robotics -> R132, Robotix -> R132
    """
    if not word or not word.strip():
        return ""

    clean = re.sub(r"[^a-zA-Z]", "", word).upper()
    if not clean:
        return ""

    first_char = clean[0]

    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6"
    }

    encoded = [first_char]
    prev_code = mapping.get(first_char, "")

    for char in clean[1:]:
        code = mapping.get(char, "")
        if code:
            if code != prev_code:
                encoded.append(code)
            prev_code = code
        else:
            prev_code = ""

    # Pad or truncate to 4 characters
    soundex_code = "".join(encoded)
    soundex_code = (soundex_code + "000")[:4]
    return soundex_code

def extract_address_anchor(cleaned_address: str) -> str:
    """
    Extracts street number + primary street word (e.g., '500_market').
    Provides a powerful anchor for businesses at the exact same physical address.
    """
    if not cleaned_address or not cleaned_address.strip():
        return ""

    tokens = cleaned_address.split()
    number_token = ""
    street_word = ""

    for tok in tokens:
        if not number_token and any(ch.isdigit() for ch in tok):
            # First token containing digits (e.g., "500", "1795", "109")
            number_token = tok
            continue
        if number_token and tok.isalpha() and len(tok) >= 3:
            # Skip road qualifiers like "road", "street", "avenue", "floor", "near"
            if tok not in {"street", "road", "avenue", "lane", "drive", "court", "boulevard", "floor", "near", "opposite", "unit"}:
                street_word = tok
                break

    if number_token and street_word:
        return f"{number_token}_{street_word}"
    return ""

class BlockingKeyExtractor:
    """Extracts a set of complementary blocking keys for a record."""

    @staticmethod
    def extract_keys(country: str, core_name: str, cleaned_address: str) -> Set[str]:
        """
        Returns a set of blocking keys for a single record.
        Each key is prefixed with country (e.g., 'US::...').
        """
        country = (country or "UNKNOWN").strip().upper()
        keys = set()

        if not core_name and not cleaned_address:
            return keys

        tokens = core_name.split() if core_name else []

        # 1. Name Prefix Key (first 3-4 chars of the first word)
        if tokens:
            first_word = tokens[0]
            if len(first_word) >= 3:
                prefix_3 = first_word[:3]
                keys.add(f"{country}::np3_{prefix_3}")
            if len(first_word) >= 4:
                prefix_4 = first_word[:4]
                keys.add(f"{country}::np4_{prefix_4}")

            # 2. Phonetic Key (Soundex of first word)
            s_code = compute_soundex(first_word)
            if s_code:
                keys.add(f"{country}::ph_{s_code}")

            # 3. First Two Words (Bigram)
            if len(tokens) >= 2:
                n2 = f"{tokens[0]}_{tokens[1]}"
                keys.add(f"{country}::n2_{n2}")

            # 4. Distinctive Tokens (rare words >= 4 chars, not common stop words)
            for tok in tokens:
                if len(tok) >= 4 and tok not in COMMON_STOP_WORDS and not tok.isdigit():
                    keys.add(f"{country}::tok_{tok}")

        # 5. Address Anchor Key (e.g., street_number + street_word)
        if cleaned_address:
            addr_anchor = extract_address_anchor(cleaned_address)
            if addr_anchor:
                keys.add(f"{country}::addr_{addr_anchor}")

        return keys
