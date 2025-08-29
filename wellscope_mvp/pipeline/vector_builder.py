# wellscope_mvp/pipeline/vector_builder.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd


NormalizeMode = Literal["q_over_qmax", "pct_decline"]
Stream = Literal["oil", "gas", "water", "boe"]


@dataclass(frozen=True)
class VectorConfig:
    months: int = 24                                # length of shape vector
    normalize: NormalizeMode = "q_over_qmax"        # "q_over_qmax" or "pct_decline"
    stream: Stream = "oil"                          # which stream to use for shape
    boe_gas_factor: float = 6.0                     # gas-to-oil conversion for BOE
    api_col: Optional[str] = None                   # let builder auto-detect if None
    date_col: str = "Monthly Production Date"       # monthly date (pre/post merge)
    oil_col: str = "Monthly Oil"
    gas_col: str = "Monthly Gas"
    water_col: str = "Monthly Water"


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_monthly_col(df: pd.DataFrame, base: str) -> Optional[str]:
    # Prefer post-merge suffix first
    return _first_present(df, [f"{base}_monthly", base])


def _detect_api_col(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns:
        return explicit
    # Prefer any normalized 14-digit join key from the merge
    cands = [c for c in df.columns if c.endswith("__norm14")]
    if cands:
        return cands[0]
    # Fallbacks that show up in loaders/join
    for c in ("API14", "API14_norm", "API_UWI_norm", "API/UWI"):
        if c in df.columns:
            return c
    raise ValueError("No API column found for vector building.")


def _assemble_stream_series(
    df: pd.DataFrame,
    oil_col: Optional[str],
    gas_col: Optional[str],
    water_col: Optional[str],
    stream: Stream,
    boe_gas_factor: float,
) -> pd.Series:
    # Use volumes directly as rate proxy (days are unreliable/zero in your exports)
    oil = pd.to_numeric(df[oil_col], errors="coerce") if oil_col else None
    gas = pd.to_numeric(df[gas_col], errors="coerce") if gas_col else None
    water = pd.to_numeric(df[water_col], errors="coerce") if water_col else None

    if stream == "oil":
        return (oil if oil is not None else pd.Series(0.0, index=df.index)).fillna(0.0)
    if stream == "gas":
        return (gas if gas is not None else pd.Series(0.0, index=df.index)).fillna(0.0)
    if stream == "water":
        return (water if water is not None else pd.Series(0.0, index=df.index)).fillna(0.0)
    # BOE: oil + gas/boe_gas_factor
    oil_part = oil if oil is not None else pd.Series(0.0, index=df.index)
    gas_part = (gas / boe_gas_factor) if gas is not None else pd.Series(0.0, index=df.index)
    return (oil_part.fillna(0.0) + gas_part.fillna(0.0))


def _align_first_n_months(df_api, date_col, value_col, n):
    s = df_api[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col]).sort_values(by=date_col)

    # from first positive month
    pos = pd.to_numeric(s[value_col], errors="coerce").fillna(0.0) > 0
    if pos.any():
        s = s.loc[pos.idxmax():]

    if s.empty:
        return np.zeros(n, dtype=float)

    # group by calendar month to remove duplicates
    s["_month"] = s[date_col].dt.to_period("M")
    g = s.groupby("_month", as_index=False)[value_col].sum()

    # build a continuous monthly index from the first month
    start = g["_month"].min()
    idx = pd.period_range(start, periods=n, freq="M")
    g = g.set_index("_month").reindex(idx, fill_value=0)

    vec = g[value_col].to_numpy(dtype=float)
    return vec[:n] if len(vec) >= n else np.pad(vec, (0, n - len(vec)), constant_values=0.0)

def _normalize_vector(vec: np.ndarray, mode: NormalizeMode) -> np.ndarray:
    if mode == "q_over_qmax":
        vmax = np.nanmax(vec) if vec.size else 0.0
        if not np.isfinite(vmax) or vmax <= 0:
            return np.zeros_like(vec)
        return (vec / vmax).astype(float, copy=False)

    # pct_decline: Δ% month-over-month; length = n (first element = 0.0)
    out = np.zeros_like(vec, dtype=float)
    if vec.size <= 1:
        return out
    prev = vec[:-1]
    curr = vec[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(prev > 0, (curr / prev) - 1.0, 0.0)
    out[0] = 0.0
    out[1:] = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_shape_vectors(
    joined_df: pd.DataFrame,
    cfg: VectorConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build per-well shape vectors of fixed length from monthly production.

    Returns:
      vectors_df: columns = [api_col, *v0..v{n-1}]
      meta: {'n_wells', 'n_features', 'normalize', 'stream'}
    """
    df = joined_df.copy()

    api_col = _detect_api_col(df, cfg.api_col)

    date_col = _pick_monthly_col(df, cfg.date_col)
    oil_col = _pick_monthly_col(df, cfg.oil_col)
    gas_col = _pick_monthly_col(df, cfg.gas_col)
    water_col = _pick_monthly_col(df, cfg.water_col)

    if date_col is None:
        raise ValueError("Monthly production date column not found in joined_df.")

    # Construct the chosen stream series as a new column
    value_series = _assemble_stream_series(df, oil_col, gas_col, water_col, cfg.stream, cfg.boe_gas_factor)
    value_col = "__vec_value__"
    df[value_col] = value_series

    # Build vectors per well
    n = int(cfg.months)
    apis = df[api_col].astype(str).fillna("").str.strip()
    unique_apis = apis[apis != ""].unique().tolist()

    vectors = np.zeros((len(unique_apis), n), dtype=float)
    for i, a in enumerate(unique_apis):
        df_a = df.loc[apis == a, [date_col, value_col]]
        vec = _align_first_n_months(df_a, date_col=date_col, value_col=value_col, n=n)
        vec = _normalize_vector(vec, cfg.normalize)
        vectors[i, :] = vec

    # Produce an output DataFrame: [api, v0..v{n-1}]
    col_names = [f"v{j:02d}" for j in range(n)]
    out = pd.DataFrame(vectors, columns=col_names)
    out.insert(0, api_col, unique_apis)

    meta = {
        "n_wells": int(len(unique_apis)),
        "n_features": n,
        "normalize": cfg.normalize,
        "stream": cfg.stream,
    }
    return out, meta