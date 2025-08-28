# wellscope_mvp/schema/headers_schema.py

"""
Strict schema for Well Headers CSV.

Rules:
- Require EXACT header names listed in REQUIRED_FIELDS (extras are allowed).
- Types are not enforced here (loaders will coerce), but field groupings are provided.
"""

# ⚠️ Exact column titles as they appear in your CSV
REQUIRED_FIELDS = [
    "API14",                                 # primary key (14-digit)
    "Surface Hole Latitude (WGS84)",
    "Surface Hole Longitude (WGS84)",
    "Bottom Hole Latitude (WGS84)",
    "Bottom Hole Longitude (WGS84)",
    "DI Play",
    "DI Subplay",
    "Target Formation",
    "Formation at Total Depth",
    "DI Lateral Length",
    "Horizontal Length",
    "True Vertical Depth",
    "Measured Depth (TD)",
    "Gross Perforated Interval",
    "Upper Perforation",
    "Lower Perforation",
    "Completion Date",
    "Spud Date",
    "First Prod Date",
    "Operator (Reported)",
    "Drill Type",
    "County/Parish",
    "Producing Reservoir",
    "Operator Ticker",

]

# Helpful groupings for loaders (optional to enforce here)
KEY_FIELDS = ["API14"]

DATE_FIELDS = [
    "Completion Date",
    "Spud Date",
    "First Prod Date",
]

FLOAT_FIELDS = [
    "Surface Hole Latitude (WGS84)",
    "Surface Hole Longitude (WGS84)",
    "Bottom Hole Latitude (WGS84)",
    "Bottom Hole Longitude (WGS84)",
    "DI Lateral Length",
    "Horizontal Length",
    "True Vertical Depth",
    "Measured Depth (TD)",
    "Gross Perforated Interval",
    "Upper Perforation",
    "Lower Perforation",
]

STRING_FIELDS = list(
    set(REQUIRED_FIELDS)
    - set(KEY_FIELDS)
    - set(DATE_FIELDS)
    - set(FLOAT_FIELDS)
)

def validate_columns(actual_columns: list[str]) -> tuple[bool, list[str]]:
    """
    Check that all REQUIRED_FIELDS are present (extras are fine).
    Returns (is_valid, missing_fields).
    """
    actual = set(actual_columns)
    missing = [c for c in REQUIRED_FIELDS if c not in actual]
    return (len(missing) == 0, missing)
