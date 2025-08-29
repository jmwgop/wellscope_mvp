# wellscope_mvp/pipeline/filter_inputs.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


# ---- Config dataclass (lightweight container) ---------------------------------

@dataclass(frozen=True)
class FilterConfig:
    formations: Optional[Iterable[str]] = None                 # e.g. {"EAGLEFORD", "AUSTIN CHALK"}
    subplays: Optional[Iterable[str]] = None                   # optional
    operators: Optional[Iterable[str]] = None                  # optional (Operator (Reported))
    completion_year_range: Optional[Tuple[int, int]] = None    # e.g. (2016, 2024)
    lateral_ft_range: Optional[Tuple[Optional[float], Optional[float]]] = None  # (min,max)
    well_status_in: Optional[Iterable[str]] = None             # e.g. {"ACTIVE","COMPLETED","INACTIVE","SHUT-IN","P & A"}
    min_months_produced: int = 0                               # 0 = no filter
    api_col_headers: str = "API14"                             # normalized, 14-digit in headers
    api_col_monthly: str = None                                # optional override if needed
    formation_col: str = "Target Formation"
    subplay_col: str = "DI Subplay"
    operator_col: str = "Operator (Reported)"
    completion_date_col: str = "Completion Date"
    status_col: str = "Well Status"                            # may not exist in all datasets
    lateral_cols: Tuple[str, ...] = ("DI Lateral Length", "Horizontal Length")
    monthly_date_col: str = "Monthly Production Date"
    monthly_days_col: str = "Days"


# ---- Helper functions ---------------------------------------------------------

def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_months_produced(
    joined_df: pd.DataFrame,
    api_col: str,
    monthly_date_col: str = "Monthly Production Date",
) -> pd.Series:
    """
    Count produced months per well. A month is produced if it has a valid date AND
    any of Monthly Oil/Gas/Water > 0. Ignores 'Days' entirely.
    """

    df = joined_df.copy()

    # Prefer suffixed post-merge columns, else use raw
    date_col = next((c for c in (f"{monthly_date_col}_monthly", monthly_date_col) if c in df.columns), None)
    oil_col = next((c for c in ("Monthly Oil_monthly", "Monthly Oil") if c in df.columns), None)
    gas_col = next((c for c in ("Monthly Gas_monthly", "Monthly Gas") if c in df.columns), None)
    water_col = next((c for c in ("Monthly Water_monthly", "Monthly Water") if c in df.columns), None)

    if date_col is None:
        raise ValueError("No production date column found in joined_df.")

    dt = pd.to_datetime(df[date_col], errors="coerce")
    produced_mask = dt.notna()

    # Volume > 0 across any stream
    vol_positive = pd.Series(False, index=df.index)
    for c in (oil_col, gas_col, water_col):
        if c is not None:
            vol_positive |= (pd.to_numeric(df[c], errors="coerce") > 0)

    produced_mask &= vol_positive

    # Group by API and count distinct months
    per_api_months = (
        df.loc[produced_mask, [api_col, date_col]]
          .assign(_m=lambda x: pd.to_datetime(x[date_col], errors="coerce").dt.to_period("M"))
          .dropna(subset=["_m", api_col])
          .groupby(api_col)["_m"].nunique()
          .rename("months_produced")
    )

    out = df[[api_col]].merge(per_api_months.reset_index(), on=api_col, how="left")["months_produced"]
    out = out.fillna(0).astype(int)
    out.index = df.index
    return out


def apply_filters(
    joined_df: pd.DataFrame,
    cfg: FilterConfig,
) -> Dict[str, Any]:
    """
    Apply pre-clustering filters to the joined dataset.

    Returns a dict:
      {
        'filtered': DataFrame (subset of joined_df),
        'mask': Boolean mask applied to joined_df,
        'stats': {
            'input_rows', 'output_rows',
            'kept_fraction',
            'reasons': { 'formation': kept_count, 'year': kept_count, ... }  # optional simple counts
        }
      }
    """
    df = joined_df.copy()

    # Decide which API column to use for months-produced rollup
    api_col = cfg.api_col_headers if cfg.api_col_monthly is None else cfg.api_col_monthly
    if api_col not in df.columns:
        # try to find a normalized join key in the merged table
        # prefer any column ending with "__norm14"
        norm14_cols = [c for c in df.columns if c.endswith("__norm14")]
        if norm14_cols:
            api_col = norm14_cols[0]
        else:
            raise ValueError("No API column available for filtering/rollups.")

    # Base mask = all True
    mask = pd.Series(True, index=df.index)

    # Formation filter
    if cfg.formations is not None and len(cfg.formations) > 0 and cfg.formation_col in df.columns:
        mask &= df[cfg.formation_col].isin(set(cfg.formations))

    # Subplay filter
    if cfg.subplays is not None and len(cfg.subplays) > 0 and cfg.subplay_col in df.columns:
        mask &= df[cfg.subplay_col].isin(set(cfg.subplays))

    # Operator filter
    if cfg.operators is not None and len(cfg.operators) > 0 and cfg.operator_col in df.columns:
        mask &= df[cfg.operator_col].isin(set(cfg.operators))

    # Completion year filter
    if cfg.completion_year_range is not None and cfg.completion_date_col in df.columns:
        years = pd.to_datetime(df[cfg.completion_date_col], errors="coerce").dt.year
        y0, y1 = cfg.completion_year_range
        if y0 is not None:
            mask &= (years >= int(y0))
        if y1 is not None:
            mask &= (years <= int(y1))

    # Lateral length range filter (use first available lateral column)
    lat_col = _first_existing(df, cfg.lateral_cols)
    if cfg.lateral_ft_range is not None and lat_col is not None:
        lat_min, lat_max = cfg.lateral_ft_range
        lat_vals = pd.to_numeric(df[lat_col], errors="coerce")
        if lat_min is not None:
            mask &= (lat_vals >= float(lat_min))
        if lat_max is not None:
            mask &= (lat_vals <= float(lat_max))

    # Well status filter (optional)
    if cfg.well_status_in is not None and cfg.status_col in df.columns:
        mask &= df[cfg.status_col].isin(set(cfg.well_status_in))

    # Months produced filter
    if cfg.min_months_produced and cfg.min_months_produced > 0:
        months = compute_months_produced(
            df,
            api_col=api_col,
            monthly_date_col=cfg.monthly_date_col,
        )
        mask &= (months >= int(cfg.min_months_produced))

    filtered = df.loc[mask].copy()

    stats = {
        "input_rows": int(len(df)),
        "output_rows": int(len(filtered)),
        "kept_fraction": float(0 if len(df) == 0 else len(filtered) / len(df)),
    }

    return {"filtered": filtered, "mask": mask, "stats": stats}