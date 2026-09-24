"""
DataFrame Normalizer Service.
Applies TextCleaner across dataset columns with vectorized/batch processing.
"""
from typing import Dict, Any
import pandas as pd
from src.preprocessing.text_cleaner import TextCleaner

class DataFrameNormalizer:
    """Batch normalization for entity resolution DataFrames."""

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes a raw dataframe containing:
        ['entity_id', 'business_name', 'business_address', 'country']
        
        Adds:
        - cleaned_name: canonicalized full name
        - core_name: name with corporate/legal suffixes stripped
        - cleaned_address: normalized address
        - name_missing: 1 if business_name is empty/missing, else 0
        - address_missing: 1 if business_address is empty/missing, else 0
        """
        result = df.copy()

        # Ensure string types and handle nulls
        raw_names = result["business_name"].fillna("").astype(str)
        raw_addresses = result["business_address"].fillna("").astype(str)

        # Missingness indicator flags
        result["name_missing"] = (raw_names.str.strip() == "").astype("int8")
        result["address_missing"] = (raw_addresses.str.strip() == "").astype("int8")

        # Country normalization (always clean uppercase stripped)
        result["country"] = result["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

        # Clean names: produces (cleaned_name, core_name)
        cleaned_names = []
        core_names = []
        for name in raw_names:
            c_name, cr_name = TextCleaner.clean_name(name)
            cleaned_names.append(c_name)
            core_names.append(cr_name)

        result["cleaned_name"] = cleaned_names
        result["core_name"] = core_names

        # Clean addresses with country-aware rules
        countries = result["country"].tolist()
        cleaned_addresses = [
            TextCleaner.clean_address(addr, country=c)
            for addr, c in zip(raw_addresses, countries)
        ]
        result["cleaned_address"] = cleaned_addresses

        return result

