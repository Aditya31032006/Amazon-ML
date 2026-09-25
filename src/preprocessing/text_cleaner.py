"""
Text Cleaning, Normalization, Transliteration, and Canonicalization.
Handles Phase 1 requirements:
- Encoding & mojibake repair with ftfy
- Native Indic & transliterated corporate suffix canonicalization
- Script transliteration with unidecode
- Unicode NFKD normalization, accent removal, lowercasing
- Noise, URLs, and trailing ID stripping
- Legal suffix canonicalization (cleaned_name vs core_name)
- Country-aware address token and state abbreviation canonicalization
"""
import re
import unicodedata
from typing import Tuple
import ftfy
import unidecode

# ---------------------------------------------------------
# Regex Patterns for Cleaning & Normalization
# ---------------------------------------------------------

# Trailing ID and branch patterns (e.g., "- 2067865001", "#8", "No. 4", "Store 12")
TRAILING_ID_PATTERN = re.compile(
    r"(?:[-–—\s#]+(?:id|no|num|branch|store)?\s*[:#-]?\s*\d+\b|#\s*\d+\b)",
    re.IGNORECASE
)

# DBA (Doing Business As) pattern
DBA_PATTERN = re.compile(r"\b(?:dba|d/b/a|doing business as|t/a|trading as)\b", re.IGNORECASE)

# URL / Website patterns
URL_PREFIX_PATTERN = re.compile(r"\b(?:https?://(?:www\.)?|www\.)", re.IGNORECASE)
DOMAIN_SUFFIX_PATTERN = re.compile(r"\.(?:com|in|org|net|co\.in|co|gov|edu|io|ai)\b", re.IGNORECASE)

# Native Indic Legal Suffixes (mapped before transliteration)
NATIVE_INDIC_LEGAL_MAP = {
    # Hindi / Devanagari
    r"(?:प्राइवेट\s*लिमिटेड|प्रा(?:इवेट)?[\s\.]*लि(?:मिटेड)?[\s\.]*)": " pvt ltd ",
    r"(?:लिमिटेड|लि[\s\.]+)": " ltd ",
    r"(?:एल[\s\.]*एल[\s\.]*पी[\s\.]*|एलएलपी)": " llp ",
    r"(?:कम्पनी|कंपनी)": " co ",
    # Tamil
    r"(?:பிரைவேட்\s*லிமிடெட்|பிரைவேட்)": " pvt ltd ",
    r"லிமிடெட்": " ltd ",
    r"எல்எல்பி": " llp ",
    # Telugu / Kannada / Bengali
    r"ప్రైవేట్\s*లిమిటెడ్": " pvt ltd ",
    r"ಪ್ರೈವೇಟ್\s*ಲಿಮಿಟೆಡ್": " pvt ltd ",
    r"প্রাইভেট\s*লিমিটেড": " pvt ltd ",
}

# English & Transliterated Legal Suffix Mapping (ordered: longest/most specific first)
LEGAL_SUFFIX_MAP = [
    # Private Limited and transliterated variants
    (re.compile(r"\b(?:private\s+limited|pvt\.?\s*ltd\.?|private\s+ltd|pvt\b|praaivett\s+limittedd|praaivett|piraiveett\s+limittett|piraiveett)\b", re.I), "pvt ltd"),
    # Limited Liability Partnership
    (re.compile(r"\b(?:limited\s+liability\s+partnership|l\.?l\.?p\.?|llp|elelpii)\b", re.I), "llp"),
    # Limited Liability Company & Professional LLC
    (re.compile(r"\b(?:professional\s+limited\s+liability\s+company|p\.?l\.?l\.?c\.?|pllc)\b", re.I), "pllc"),
    (re.compile(r"\b(?:limited\s+liability\s+company|l\.?l\.?c\.?|llc)\b", re.I), "llc"),
    # Limited Partnership
    (re.compile(r"\b(?:limited\s+partnership|l\.?p\.?|lp)\b", re.I), "lp"),
    # Incorporated
    (re.compile(r"\b(?:incorporated|inc\.?)\b", re.I), "inc"),
    # Corporation & Professional Corp
    (re.compile(r"\b(?:professional\s+corporation|p\.?c\.\b|\bpc$)\b", re.I), "pc"),
    (re.compile(r"\b(?:corporation|corp\.?)\b", re.I), "corp"),
    # General Limited
    (re.compile(r"\b(?:limited|ltd\.?|limittedd|limittett)\b", re.I), "ltd"),
    # Company
    (re.compile(r"\b(?:company|co\.?)\b", re.I), "co"),
    # Enterprises / Holdings / Services
    (re.compile(r"\b(?:enterprises|enterprise)\b", re.I), "enterprises"),
    (re.compile(r"\b(?:holdings|holding)\b", re.I), "holdings"),
    (re.compile(r"\b(?:services|service)\b", re.I), "services"),
    # French corporate forms (sarl, sasu, sas, eurl, sci, sa)
    (re.compile(r"\b(?:s\.?a\.?r\.?l\.?|sarl)\b", re.I), "sarl"),
    (re.compile(r"\b(?:s\.?a\.?s\.?u\.?|sasu)\b", re.I), "sasu"),
    (re.compile(r"\b(?:s\.?a\.?s\.?|sas)\b", re.I), "sas"),
    (re.compile(r"\b(?:e\.?u\.?r\.?l\.?|eurl)\b", re.I), "eurl"),
    (re.compile(r"\b(?:s\.?c\.?i\.?|sci)\b", re.I), "sci"),
    (re.compile(r"\b(?:s\.?a\.?|sa)\b", re.I), "sa"),
]

# Set of known canonical suffixes to strip from core_name
CANONICAL_SUFFIX_TOKENS = {
    "pvt", "ltd", "inc", "corp", "llc", "pllc", "llp", "lp", "co", "pc",
    "enterprises", "holdings", "services",
    "sarl", "sasu", "sas", "eurl", "sci", "sa",
    "private", "limited", "corporation", "incorporated", "company",
    "praaivett", "limittedd", "elelpii", "piraiveett", "limittett"
}


# Address Token Standardization Mapping
ADDRESS_TOKEN_MAP = {
    r"\b(st|st\.)\b": "street",
    r"\b(rd|rd\.)\b": "road",
    r"\b(ave|ave\.|av\.)\b": "avenue",
    r"\b(blvd|blvd\.)\b": "boulevard",
    r"\b(dr|dr\.)\b": "drive",
    r"\b(ln|ln\.)\b": "lane",
    r"\b(ct|ct\.)\b": "court",
    r"\b(hwy|hwy\.)\b": "highway",
    r"\b(pkwy|pkwy\.)\b": "parkway",
    r"\b(apt|apt\.|apartment)\b": "apt",
    r"\b(ste|ste\.|suite)\b": "suite",
    r"\b(fl|flr|fl\.|floor)\b": "floor",
    r"\b(nr|nr\.)\b": "near",
    r"\b(opp|opp\.)\b": "opposite",
    r"\b(bldg|bldg\.|building)\b": "building",
    r"\b(dept|department)\b": "dept",
    r"\b(w/o|s/o|d/o)\b": " ",
}

# State Mapping for India
INDIA_STATE_MAP = {
    r"\b(mh)\b": "maharashtra",
    r"\b(tn)\b": "tamil nadu",
    r"\b(ka)\b": "karnataka",
    r"\b(dl)\b": "delhi",
    r"\b(wb)\b": "west bengal",
    r"\b(up)\b": "uttar pradesh",
    r"\b(gj)\b": "gujarat",
    r"\b(ts|tg)\b": "telangana",
    r"\b(ap)\b": "andhra pradesh",
    r"\b(rj)\b": "rajasthan",
    r"\b(mp)\b": "madhya pradesh",
    r"\b(kl)\b": "kerala",
    r"\b(hr)\b": "haryana",
    r"\b(pb)\b": "punjab",
}

# State Mapping for US (Tennessee, Washington, Ohio, etc.)
US_STATE_MAP = {
    r"\b(tn)\b": "tennessee",
    r"\b(wa)\b": "washington",
    r"\b(oh)\b": "ohio",
    r"\b(nc)\b": "north carolina",
    r"\b(sc)\b": "south carolina",
    r"\b(va)\b": "virginia",
    r"\b(fl)\b": "florida",
    r"\b(tx)\b": "texas",
    r"\b(ca)\b": "california",
    r"\b(ny)\b": "new york",
    r"\b(il)\b": "illinois",
    r"\b(pa)\b": "pennsylvania",
    r"\b(ga)\b": "georgia",
    r"\b(mi)\b": "michigan",
    r"\b(az)\b": "arizona",
    r"\b(co)\b": "colorado",
    r"\b(wi)\b": "wisconsin",
    r"\b(ky)\b": "kentucky",
    r"\b(ok)\b": "oklahoma",
    r"\b(al)\b": "alabama",
    r"\b(or)\b": "oregon",
    r"\b(nv)\b": "nevada",
    r"\b(mo)\b": "missouri",
    r"\b(mn)\b": "minnesota",
    r"\b(md)\b": "maryland",
    r"\b(ct)\b": "connecticut",
    r"\b(nj)\b": "new jersey",
    r"\b(ma)\b": "massachusetts",
}


class TextCleaner:
    """Production-grade text cleaning for multilingual entity resolution."""

    @staticmethod
    def fix_encoding(text: str) -> str:
        """Repairs mojibake (e.g., UTF-8 decoded as Latin-1) and broken entities."""
        if not text:
            return ""
        return ftfy.fix_text(text)

    @staticmethod
    def transliterate_indic(text: str) -> str:
        """Transliterates Indic and other non-Latin scripts to Latin ASCII."""
        if not text:
            return ""
        return unidecode.unidecode(text)

    @staticmethod
    def basic_normalize(text: str) -> str:
        """
        Fixes encoding, transliterates non-Latin scripts,
        decomposes Unicode (NFKD), lowercases, and expands symbols.
        """
        if not text:
            return ""

        # Step 1: Fix encoding mojibake
        text = TextCleaner.fix_encoding(text)

        # Step 2: Handle apostrophe-s (e.g. "candy's" -> "candys")
        text = re.sub(r"['’]s\b", "s", text, flags=re.IGNORECASE)

        # Step 3: Replace native Indic legal suffixes before transliteration
        for pattern, replacement in NATIVE_INDIC_LEGAL_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Step 4: Transliterate non-Latin scripts (Tamil, Hindi, Kannada, etc.)
        text = TextCleaner.transliterate_indic(text)

        # Step 5: Unicode NFKD normalization & accent stripping
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = text.lower()

        # Step 6: Expand common symbols
        text = text.replace("&", " and ")
        text = text.replace("@", " at ")
        text = text.replace("+", " plus ")
        text = text.replace("%", " percent ")

        # Step 7: Replace punctuation with space, keep alphanumeric
        text = re.sub(r"[^\w\s]", " ", text)

        # Step 8: Collapse whitespace
        return " ".join(text.split())

    @staticmethod
    def clean_name(raw_name: str) -> Tuple[str, str]:
        """
        Cleans a business name and returns:
        - cleaned_name: standardized full name with canonical suffixes
        - core_name: name with all corporate/legal suffixes completely stripped
        """
        if not raw_name or not str(raw_name).strip():
            return "", ""

        text = str(raw_name).strip()

        # 1. Handle Pipe separators (e.g. "SHIVSHAKTI CORP | www.shivshakti.com")
        if "|" in text:
            parts = text.split("|")
            text = parts[0].strip()

        # 2. Clean URLs (e.g. "www.wilfordhancock.com" -> "wilfordhancock")
        if "." in text and any(ext in text.lower() for ext in [".com", ".org", ".in", ".co", ".net"]):
            text = URL_PREFIX_PATTERN.sub(" ", text)
            text = DOMAIN_SUFFIX_PATTERN.sub(" ", text)

        # 3. Strip DBA markers
        text = DBA_PATTERN.sub(" ", text)

        # 4. Strip trailing IDs (e.g. "- 2067865001", "#8")
        text = TRAILING_ID_PATTERN.sub(" ", text)

        # 5. Standard normalize (mojibake, transliteration, unicode, lowercase)
        norm = TextCleaner.basic_normalize(text)

        if not norm:
            return "", ""

        # 6. Canonicalize legal suffixes (most specific first)
        cleaned_name = norm
        for pattern, replacement in LEGAL_SUFFIX_MAP:
            cleaned_name = pattern.sub(f" {replacement} ", cleaned_name)
        cleaned_name = " ".join(cleaned_name.split())

        # Collapse duplicate repeated suffixes (e.g. "pvt ltd ltd" -> "pvt ltd", "llc llc" -> "llc")
        cleaned_name = re.sub(r"\b(pvt ltd)(?:\s+ltd)+\b", r"\1", cleaned_name)
        cleaned_name = re.sub(r"\b(llc)(?:\s+llc)+\b", r"\1", cleaned_name)
        cleaned_name = re.sub(r"\b(co)(?:\s+co)+\b", r"\1", cleaned_name)
        cleaned_name = re.sub(r"\b(inc)(?:\s+inc)+\b", r"\1", cleaned_name)
        cleaned_name = " ".join(cleaned_name.split())

        # 7. Extract core_name (strip all legal suffixes from token list)
        tokens = cleaned_name.split()
        core_tokens = [tok for tok in tokens if tok not in CANONICAL_SUFFIX_TOKENS]

        core_name = " ".join(core_tokens).strip()
        # Also remove any remaining trailing store/branch digits (e.g. "cafe 8" -> "cafe")
        core_name = re.sub(r"\s+\d+$", "", core_name).strip()

        # Fallback if stripping removed everything
        if not core_name:
            core_name = cleaned_name

        return cleaned_name, core_name

    @staticmethod
    def clean_address(raw_address: str, country: str = "US") -> str:
        """
        Cleans, transliterates, and canonicalizes an address string.
        Applies country-aware state mapping (e.g. TN -> Tennessee in US, TN -> Tamil Nadu in India).
        """
        if not raw_address or not str(raw_address).strip():
            return ""

        norm = TextCleaner.basic_normalize(str(raw_address))
        if not norm:
            return ""

        # Canonicalize address tokens (st -> street, rd -> road, etc.)
        for pattern, replacement in ADDRESS_TOKEN_MAP.items():
            norm = re.sub(pattern, replacement, norm, flags=re.IGNORECASE)

        # Country-aware state canonicalization
        country_norm = (country or "").strip().upper()
        if country_norm == "INDIA":
            for pattern, replacement in INDIA_STATE_MAP.items():
                norm = re.sub(pattern, replacement, norm, flags=re.IGNORECASE)
        elif country_norm == "US":
            for pattern, replacement in US_STATE_MAP.items():
                norm = re.sub(pattern, replacement, norm, flags=re.IGNORECASE)
        elif country_norm == "FRANCE":
            # Expand road abbreviations in French addresses safely without affecting non-road words
            norm = re.sub(r"\b(r|r\.)\s+(de|du|des|d)\b", r"rue \2", norm, flags=re.IGNORECASE)
            norm = re.sub(r"\b(\d+[\w-]*)\s+(r|r\.)\b", r"\1 rue", norm, flags=re.IGNORECASE)
            norm = re.sub(r"\b(bd|bd\.|bld)\b", "boulevard", norm, flags=re.IGNORECASE)
            norm = re.sub(r"\b(av|av\.)\b", "avenue", norm, flags=re.IGNORECASE)

        # Collapse duplicate adjacent tokens (e.g. "unit unit 2" -> "unit 2")

        tokens = norm.split()
        dedup_tokens = []
        for tok in tokens:
            if not dedup_tokens or tok != dedup_tokens[-1]:
                dedup_tokens.append(tok)

        return " ".join(dedup_tokens)
