# wellscope_mvp/pipeline/join_inputs.py

from __future__ import annotations
import re
from typing import Dict, Tuple, Optional

import pandas as pd


def _digits_only(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    d = re.sub(r"\D", "", str(s))
    return d or None


def _normalize_api14(s: Optional[str]) -> Optional[str]:
    """
    Normalize any API-like value to a 14-digit numeric string.

    Rules:
      - strip non-digits
      - if exactly 10 digits: append '0000'
      - left-pad to 14 digits (handles leading-zero loss)
      - trim to 14 digits if longer
    """
    d = _digits_only(s)
    if not d:
        return None
    if len(d) == 10:
        d = d + "0000"
    if len(d) < 14:
        d = d.zfill(14)
    if len(d) > 14:
        d = d[:14]
    return d


def _pick_existing(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def join_headers_and_monthly(
    headers_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    *,
    headers_api_candidates: Tuple[str, ...] = ("API14",),
    monthly_api_candidates: Tuple[str, ...] = ("API14_norm", "API_UWI_norm", "API/UWI"),
    how: str = "inner",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Join headers and monthly data on a shared API-14 key.

    Parameters
    ----------
    headers_df : pd.DataFrame
        DataFrame from headers loader (should include API14 or similar).
    monthly_df : pd.DataFrame
        DataFrame from monthly loader (may include API14_norm or API/UWI).
    headers_api_candidates : tuple[str, ...]
        Candidate column names in headers for the API14 key.
    monthly_api_candidates : tuple[str, ...]
        Candidate column names in monthly for the API key (API14_norm preferred).
    how : str
        Pandas merge how ('inner' by default). Use 'left' to keep all monthly rows.

    Returns
    -------
    joined : pd.DataFrame
        Joined dataset with columns from both inputs.
    stats : dict
        {
          'headers_rows': int,
          'headers_unique_api14': int,
          'monthly_rows': int,
          'monthly_unique_api14': int,
          'joined_rows': int,
          'matched_api14': int,          # unique API14s present in join
          'unmatched_headers_api14': int # headers unique - matched_api14
        }
    """
    # Pick input columns
    h_api_col = _pick_existing(headers_df, headers_api_candidates)
    m_api_col = _pick_existing(monthly_df, monthly_api_candidates)

    if h_api_col is None:
        raise ValueError(
            f"No API column from {headers_api_candidates} found in headers_df."
        )
    if m_api_col is None:
        raise ValueError(
            f"No API column from {monthly_api_candidates} found in monthly_df."
        )

    # Build normalized 14-digit keys
    headers_key = f"{h_api_col}__norm14"
    monthly_key = f"{m_api_col}__norm14"

    headers_norm = headers_df.copy()
    monthly_norm = monthly_df.copy()

    headers_norm[headers_key] = headers_norm[h_api_col].apply(_normalize_api14)
    monthly_norm[monthly_key] = monthly_norm[m_api_col].apply(_normalize_api14)

    # Compute stats before join
    headers_unique_api14 = (
        headers_norm[headers_key].dropna().astype(str).str.zfill(14).nunique()
    )
    monthly_unique_api14 = (
        monthly_norm[monthly_key].dropna().astype(str).str.zfill(14).nunique()
    )

    # Perform join
    joined = monthly_norm.merge(
        headers_norm,
        left_on=monthly_key,
        right_on=headers_key,
        how=how,
        suffixes=("_monthly", "_headers"),
    )

    # Post-join coverage stats
    matched_api14 = (
        joined[monthly_key]
        .dropna()
        .astype(str)
        .str.zfill(14)
        .nunique()
    )
    unmatched_headers_api14 = max(headers_unique_api14 - matched_api14, 0)

    stats = {
        "headers_rows": int(len(headers_df)),
        "headers_unique_api14": int(headers_unique_api14),
        "monthly_rows": int(len(monthly_df)),
        "monthly_unique_api14": int(monthly_unique_api14),
        "joined_rows": int(len(joined)),
        "matched_api14": int(matched_api14),
        "unmatched_headers_api14": int(unmatched_headers_api14),
    }

    return joined, stats