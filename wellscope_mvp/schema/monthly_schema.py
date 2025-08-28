# wellscope_mvp/schema/monthly_schema.py

"""
Strict schema for Producing Entity Monthly Production CSV.

Rules:
- Require EXACT header names listed in REQUIRED_FIELDS (extras are allowed).
- Types are not enforced here (loaders will coerce), but field groupings are provided.
"""

# ⚠️ Exact column titles as they appear in your CSV
REQUIRED_FIELDS = [
    "API/UWI",
    "Monthly Production Date",
    "Monthly Oil",
    "Monthly Gas",
    "Monthly Water",
    "Well Count",
    "Producing Month Number",
]

# Helpful groupings for loaders (optional to enforce here)
KEY_FIELDS = ["API/UWI"]

DATE_FIELDS = ["Monthly Production Date"]

FLOAT_FIELDS = [
    "Monthly Oil",
    "Monthly Gas",
    "Monthly Water",
]

INT_FIELDS = [
    "Well Count",
    "Producing Month Number",
]

STRING_FIELDS = list(
    set(REQUIRED_FIELDS)
    - set(KEY_FIELDS)
    - set(DATE_FIELDS)
    - set(FLOAT_FIELDS)
    - set(INT_FIELDS)
)

def validate_columns(actual_columns: list[str]) -> tuple[bool, list[str]]:
    """
    Check that all REQUIRED_FIELDS are present (extras are fine).
    Returns (is_valid, missing_fields).
    """
    actual = set(actual_columns)
    missing = [c for c in REQUIRED_FIELDS if c not in actual]
    return (len(missing) == 0, missing)
