"""
camelCase → snake_case key mapping.

When integrating third-party APIs, their key naming conventions rarely match
yours.  Centralising the conversion here means the rest of the system never
has to know what keys the external API uses.

Concept: recursive dict/list traversal — a common pattern when normalising
         arbitrarily-nested JSON payloads.
"""

import re


def _camel_to_snake(name: str) -> str:
    """
    'riskScore'      → 'risk_score'
    'HTTPSEnabled'   → 'https_enabled'
    'companyId'      → 'company_id'
    """
    # Insert underscore before a capital that follows a lowercase or digit
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Insert underscore before a capital that is followed by a lowercase
    # (handles 'HTTPSEnabled' → 'HTTPS_Enabled')
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    return s.lower()


def normalise_keys(data: dict | list) -> dict | list:
    """
    Recursively convert every dict key from camelCase to snake_case.

    Works on arbitrarily nested structures (dicts inside lists, etc.).
    """
    if isinstance(data, dict):
        return {_camel_to_snake(k): normalise_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalise_keys(item) for item in data]
    return data  # scalar — return as-is
