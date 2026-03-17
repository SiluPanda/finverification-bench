"""Error taxonomy for financial statement verification benchmarks.

Defines a structured hierarchy of error types that can appear in (or be
injected into) financial statements.  Every error type carries metadata
used downstream for injection, scoring, and analysis.

Taxonomy
--------
AE  — Arithmetic errors        (sums / subtotals)
CL  — Cross-statement linkage  (values that must agree across statements)
YOY — Year-over-year           (prior-period consistency)
MR  — Magnitude / rounding     (values close but not exact)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict, List


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class ErrorCategory(str, Enum):
    """Top-level error category."""

    ARITHMETIC_ERROR = "AE"
    CROSS_STATEMENT_LINKAGE = "CL"
    YEAR_OVER_YEAR = "YOY"
    MAGNITUDE_ROUNDING = "MR"


@unique
class ErrorSubtype(str, Enum):
    """Fine-grained error subtype, namespaced by category prefix."""

    # Arithmetic errors
    AE_ROW_SUM = "AE_ROW_SUM"
    AE_COLUMN_SUM = "AE_COLUMN_SUM"

    # Cross-statement linkage
    CL_NET_INCOME_TO_RE = "CL_NET_INCOME_TO_RE"
    CL_NET_INCOME_TO_CFS = "CL_NET_INCOME_TO_CFS"
    CL_ENDING_CASH = "CL_ENDING_CASH"
    CL_DEPRECIATION = "CL_DEPRECIATION"

    # Year-over-year
    YOY_OPENING_BALANCE = "YOY_OPENING_BALANCE"
    YOY_COMPUTED_CHANGE = "YOY_COMPUTED_CHANGE"

    # Magnitude / rounding
    MR_MINOR = "MR_MINOR"
    MR_MODERATE = "MR_MODERATE"
    MR_SIGNIFICANT = "MR_SIGNIFICANT"
    MR_EXTREME = "MR_EXTREME"


@unique
class ErrorSeverity(str, Enum):
    """How materially significant the error would be if real."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@unique
class DetectionDifficulty(str, Enum):
    """Relative difficulty of detecting this error type.

    Reflects how hard it is for an LLM or human auditor to notice the
    discrepancy without explicit re-computation.
    """

    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    VERY_HARD = "very_hard"


# ---------------------------------------------------------------------------
# Error-type dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorType:
    """Complete descriptor for a single error type.

    Attributes
    ----------
    category : ErrorCategory
        Top-level classification (AE, CL, YOY, MR).
    subtype : ErrorSubtype
        Fine-grained classification within the category.
    description : str
        Human-readable explanation of the error.
    severity : ErrorSeverity
        Materiality level.
    detection_difficulty : DetectionDifficulty
        How hard the error is to spot.
    affected_statements : list[str]
        Which financial statements are involved (e.g. ``["IS", "BS"]``).
    """

    category: ErrorCategory
    subtype: ErrorSubtype
    description: str
    severity: ErrorSeverity
    detection_difficulty: DetectionDifficulty
    affected_statements: List[str] = field(default_factory=list)

    @property
    def code(self) -> str:
        """Short identifier, e.g. ``'AE_ROW_SUM'``."""
        return self.subtype.value


# ---------------------------------------------------------------------------
# Canonical registry — single source of truth for all error types
# ---------------------------------------------------------------------------

ERROR_REGISTRY: Dict[ErrorSubtype, ErrorType] = {
    # -- Arithmetic errors --------------------------------------------------
    ErrorSubtype.AE_ROW_SUM: ErrorType(
        category=ErrorCategory.ARITHMETIC_ERROR,
        subtype=ErrorSubtype.AE_ROW_SUM,
        description=(
            "Row items do not add to the stated subtotal within a single "
            "financial statement."
        ),
        severity=ErrorSeverity.HIGH,
        detection_difficulty=DetectionDifficulty.EASY,
        affected_statements=["IS", "BS", "CFS"],
    ),
    ErrorSubtype.AE_COLUMN_SUM: ErrorType(
        category=ErrorCategory.ARITHMETIC_ERROR,
        subtype=ErrorSubtype.AE_COLUMN_SUM,
        description=(
            "Subtotals across sections do not add to the stated grand total."
        ),
        severity=ErrorSeverity.HIGH,
        detection_difficulty=DetectionDifficulty.MODERATE,
        affected_statements=["IS", "BS", "CFS"],
    ),

    # -- Cross-statement linkage --------------------------------------------
    ErrorSubtype.CL_NET_INCOME_TO_RE: ErrorType(
        category=ErrorCategory.CROSS_STATEMENT_LINKAGE,
        subtype=ErrorSubtype.CL_NET_INCOME_TO_RE,
        description=(
            "Net income on the income statement does not equal the change "
            "in retained earnings on the balance sheet (after dividends)."
        ),
        severity=ErrorSeverity.CRITICAL,
        detection_difficulty=DetectionDifficulty.HARD,
        affected_statements=["IS", "BS"],
    ),
    ErrorSubtype.CL_NET_INCOME_TO_CFS: ErrorType(
        category=ErrorCategory.CROSS_STATEMENT_LINKAGE,
        subtype=ErrorSubtype.CL_NET_INCOME_TO_CFS,
        description=(
            "Net income on the income statement does not match the starting "
            "figure of the cash flow statement (indirect method)."
        ),
        severity=ErrorSeverity.CRITICAL,
        detection_difficulty=DetectionDifficulty.MODERATE,
        affected_statements=["IS", "CFS"],
    ),
    ErrorSubtype.CL_ENDING_CASH: ErrorType(
        category=ErrorCategory.CROSS_STATEMENT_LINKAGE,
        subtype=ErrorSubtype.CL_ENDING_CASH,
        description=(
            "Ending cash balance on the cash flow statement does not match "
            "the cash line on the balance sheet."
        ),
        severity=ErrorSeverity.CRITICAL,
        detection_difficulty=DetectionDifficulty.MODERATE,
        affected_statements=["CFS", "BS"],
    ),
    ErrorSubtype.CL_DEPRECIATION: ErrorType(
        category=ErrorCategory.CROSS_STATEMENT_LINKAGE,
        subtype=ErrorSubtype.CL_DEPRECIATION,
        description=(
            "Depreciation / amortisation expense is inconsistent between "
            "the income statement and the cash flow statement adjustments."
        ),
        severity=ErrorSeverity.MEDIUM,
        detection_difficulty=DetectionDifficulty.HARD,
        affected_statements=["IS", "CFS"],
    ),

    # -- Year-over-year -----------------------------------------------------
    ErrorSubtype.YOY_OPENING_BALANCE: ErrorType(
        category=ErrorCategory.YEAR_OVER_YEAR,
        subtype=ErrorSubtype.YOY_OPENING_BALANCE,
        description=(
            "Prior-year ending balance does not equal current-year opening "
            "balance for the same line item."
        ),
        severity=ErrorSeverity.HIGH,
        detection_difficulty=DetectionDifficulty.MODERATE,
        affected_statements=["BS"],
    ),
    ErrorSubtype.YOY_COMPUTED_CHANGE: ErrorType(
        category=ErrorCategory.YEAR_OVER_YEAR,
        subtype=ErrorSubtype.YOY_COMPUTED_CHANGE,
        description=(
            "The stated year-over-year change for a line item does not "
            "match the computed difference between prior and current values."
        ),
        severity=ErrorSeverity.MEDIUM,
        detection_difficulty=DetectionDifficulty.MODERATE,
        affected_statements=["IS", "BS", "CFS"],
    ),

    # -- Magnitude / rounding -----------------------------------------------
    ErrorSubtype.MR_MINOR: ErrorType(
        category=ErrorCategory.MAGNITUDE_ROUNDING,
        subtype=ErrorSubtype.MR_MINOR,
        description="Discrepancy of less than 1%.",
        severity=ErrorSeverity.LOW,
        detection_difficulty=DetectionDifficulty.VERY_HARD,
        affected_statements=["IS", "BS", "CFS"],
    ),
    ErrorSubtype.MR_MODERATE: ErrorType(
        category=ErrorCategory.MAGNITUDE_ROUNDING,
        subtype=ErrorSubtype.MR_MODERATE,
        description="Discrepancy between 1% and 5%.",
        severity=ErrorSeverity.MEDIUM,
        detection_difficulty=DetectionDifficulty.HARD,
        affected_statements=["IS", "BS", "CFS"],
    ),
    ErrorSubtype.MR_SIGNIFICANT: ErrorType(
        category=ErrorCategory.MAGNITUDE_ROUNDING,
        subtype=ErrorSubtype.MR_SIGNIFICANT,
        description="Discrepancy between 5% and 20%.",
        severity=ErrorSeverity.HIGH,
        detection_difficulty=DetectionDifficulty.MODERATE,
        affected_statements=["IS", "BS", "CFS"],
    ),
    ErrorSubtype.MR_EXTREME: ErrorType(
        category=ErrorCategory.MAGNITUDE_ROUNDING,
        subtype=ErrorSubtype.MR_EXTREME,
        description="Discrepancy exceeding 20%.",
        severity=ErrorSeverity.CRITICAL,
        detection_difficulty=DetectionDifficulty.EASY,
        affected_statements=["IS", "BS", "CFS"],
    ),
}


def get_error_type(subtype: ErrorSubtype) -> ErrorType:
    """Look up an ``ErrorType`` by its subtype enum value.

    Raises ``KeyError`` if the subtype is not in the registry.
    """
    return ERROR_REGISTRY[subtype]


def list_subtypes_for_category(category: ErrorCategory) -> List[ErrorType]:
    """Return every registered ``ErrorType`` belonging to *category*."""
    return [et for et in ERROR_REGISTRY.values() if et.category == category]
