"""Controlled error injection into parsed financial statements.

This module takes a *clean* financial-statement document (a nested dict
parsed from JSON) and introduces exactly the error specified by an
``ErrorSubtype``, returning both the mutated document and machine-readable
ground-truth metadata that describes what was changed.

Expected input schema (``FinancialStatements`` dict)
----------------------------------------------------
::

    {
        "company": "ACME Corp",
        "period": "FY2024",
        "currency": "USD",
        "unit": "millions",
        "income_statement": {
            "revenue": 1000,
            "cost_of_goods_sold": -600,
            "gross_profit": 400,
            "operating_expenses": -200,
            "depreciation_amortization": -50,
            "operating_income": 150,
            "interest_expense": -10,
            "income_before_tax": 140,
            "income_tax_expense": -35,
            "net_income": 105
        },
        "balance_sheet": {
            "current_year": {
                "cash_and_equivalents": 250,
                "accounts_receivable": 120,
                "inventory": 80,
                "total_current_assets": 450,
                "property_plant_equipment": 500,
                "total_assets": 950,
                "accounts_payable": 90,
                "short_term_debt": 60,
                "total_current_liabilities": 150,
                "long_term_debt": 200,
                "total_liabilities": 350,
                "retained_earnings": 500,
                "total_equity": 600,
                "total_liabilities_and_equity": 950
            },
            "prior_year": { ... }
        },
        "cash_flow_statement": {
            "net_income": 105,
            "depreciation_amortization": 50,
            "changes_in_working_capital": -20,
            "cash_from_operations": 135,
            "capital_expenditures": -60,
            "cash_from_investing": -60,
            "debt_repayment": -30,
            "dividends_paid": -15,
            "cash_from_financing": -45,
            "net_change_in_cash": 30,
            "beginning_cash": 220,
            "ending_cash": 250
        }
    }

Any fields present beyond this skeleton are preserved; only the fields
listed are candidates for injection.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.error_taxonomy import (
    ErrorCategory,
    ErrorSubtype,
    ErrorType,
    ERROR_REGISTRY,
)

# Type aliases for readability.
FinancialStatements = Dict[str, Any]
_Num = float  # financial values are treated as floats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    """Outcome of injecting a single error into a financial statement.

    Attributes
    ----------
    modified_statements : FinancialStatements
        Deep copy of the original statements with the error applied.
    error_injected : bool
        ``True`` if an error was actually injected (``False`` for clean
        instances or when injection was impossible).
    error_type : str | None
        The ``ErrorSubtype`` value string, e.g. ``"AE_ROW_SUM"``.
    error_category : str | None
        The ``ErrorCategory`` value string, e.g. ``"AE"``.
    error_location : str | None
        Dot-path to the field that was modified, e.g.
        ``"income_statement.revenue"``.
    error_magnitude_pct : float | None
        Percentage deviation applied (signed).
    original_value : float | None
        Value before modification.
    modified_value : float | None
        Value after modification.
    description : str | None
        Human-readable description of the injected error.
    """

    modified_statements: FinancialStatements
    error_injected: bool
    error_type: Optional[str] = None
    error_category: Optional[str] = None
    error_location: Optional[str] = None
    error_magnitude_pct: Optional[float] = None
    original_value: Optional[float] = None
    modified_value: Optional[float] = None
    description: Optional[str] = None

    def to_ground_truth(self) -> Dict[str, Any]:
        """Return a dict suitable for serialisation as benchmark ground truth."""
        return {
            "has_error": self.error_injected,
            "error_type": self.error_type,
            "error_category": self.error_category,
            "error_location": self.error_location,
            "error_magnitude_pct": self.error_magnitude_pct,
            "original_value": self.original_value,
            "modified_value": self.modified_value,
            "description": self.description,
        }


@dataclass
class MultiInjectionResult:
    """Outcome of injecting multiple errors into a financial statement."""

    modified_statements: FinancialStatements
    injections: List[InjectionResult] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for inj in self.injections if inj.error_injected)

    def to_ground_truth(self) -> Dict[str, Any]:
        return {
            "has_error": self.error_count > 0,
            "error_count": self.error_count,
            "errors": [inj.to_ground_truth() for inj in self.injections
                       if inj.error_injected],
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _perturb(value: _Num, pct: float, *, rng: random.Random) -> _Num:
    """Shift *value* by *pct* percent.  The sign of the perturbation is
    chosen randomly (but consistently via *rng*) unless *pct* is already
    signed.
    """
    direction = rng.choice([1, -1])
    delta = abs(value) * (pct / 100.0) * direction
    return round(value + delta, 2)


def _get_nested(data: dict, dotpath: str) -> Any:
    """Retrieve a value from *data* using a dot-separated path."""
    keys = dotpath.split(".")
    node = data
    for k in keys:
        node = node[k]
    return node


def _set_nested(data: dict, dotpath: str, value: Any) -> None:
    """Set a value inside *data* using a dot-separated path."""
    keys = dotpath.split(".")
    node = data
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def _magnitude_subtype_for_pct(pct: float) -> ErrorSubtype:
    """Map an absolute percentage deviation to the correct MR subtype."""
    abs_pct = abs(pct)
    if abs_pct < 1.0:
        return ErrorSubtype.MR_MINOR
    if abs_pct < 5.0:
        return ErrorSubtype.MR_MODERATE
    if abs_pct < 20.0:
        return ErrorSubtype.MR_SIGNIFICANT
    return ErrorSubtype.MR_EXTREME


# ---------------------------------------------------------------------------
# Income-statement row-sum targets
# ---------------------------------------------------------------------------
# Each tuple: (subtotal field, component fields that should add to it, sign
# convention — True means components are stored as signed values already).

_IS_ROW_SUMS: List[Tuple[str, List[str]]] = [
    (
        "gross_profit",
        ["revenue", "cost_of_goods_sold"],
    ),
    (
        "operating_income",
        ["gross_profit", "operating_expenses", "depreciation_amortization"],
    ),
    (
        "income_before_tax",
        ["operating_income", "interest_expense"],
    ),
    (
        "net_income",
        ["income_before_tax", "income_tax_expense"],
    ),
]

# Balance-sheet row sums (current_year and prior_year share structure).
_BS_ROW_SUMS: List[Tuple[str, List[str]]] = [
    (
        "total_current_assets",
        ["cash_and_equivalents", "accounts_receivable", "inventory"],
    ),
    (
        "total_assets",
        ["total_current_assets", "property_plant_equipment"],
    ),
    (
        "total_current_liabilities",
        ["accounts_payable", "short_term_debt"],
    ),
    (
        "total_liabilities",
        ["total_current_liabilities", "long_term_debt"],
    ),
    (
        "total_liabilities_and_equity",
        ["total_liabilities", "total_equity"],
    ),
]

# Cash-flow statement row sums.
_CFS_ROW_SUMS: List[Tuple[str, List[str]]] = [
    (
        "cash_from_operations",
        ["net_income", "depreciation_amortization",
         "changes_in_working_capital"],
    ),
    (
        "cash_from_investing",
        ["capital_expenditures"],
    ),
    (
        "cash_from_financing",
        ["debt_repayment", "dividends_paid"],
    ),
    (
        "net_change_in_cash",
        ["cash_from_operations", "cash_from_investing",
         "cash_from_financing"],
    ),
    (
        "ending_cash",
        ["beginning_cash", "net_change_in_cash"],
    ),
]


# ---------------------------------------------------------------------------
# Per-subtype injection functions
# ---------------------------------------------------------------------------

def _inject_ae_row_sum(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Perturb one component of a row-sum so the subtotal no longer adds up."""
    # Pick a random statement and a random row-sum group.
    candidates: List[Tuple[str, str, List[str]]] = []
    for total_field, components in _IS_ROW_SUMS:
        if "income_statement" in stmts:
            is_data = stmts["income_statement"]
            if total_field in is_data and all(c in is_data for c in components):
                candidates.append(("income_statement", total_field, components))

    if "balance_sheet" in stmts and "current_year" in stmts["balance_sheet"]:
        bs = stmts["balance_sheet"]["current_year"]
        for total_field, components in _BS_ROW_SUMS:
            if total_field in bs and all(c in bs for c in components):
                candidates.append(
                    ("balance_sheet.current_year", total_field, components))

    if "cash_flow_statement" in stmts:
        cfs = stmts["cash_flow_statement"]
        for total_field, components in _CFS_ROW_SUMS:
            if total_field in cfs and all(c in cfs for c in components):
                candidates.append(
                    ("cash_flow_statement", total_field, components))

    if not candidates:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    stmt_path, _total_field, components = rng.choice(candidates)
    target_field = rng.choice(components)
    dotpath = f"{stmt_path}.{target_field}"

    original = float(_get_nested(stmts, dotpath))
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.AE_ROW_SUM.value,
        error_category=ErrorCategory.ARITHMETIC_ERROR.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            f"Modified '{target_field}' in {stmt_path} so row items no "
            f"longer sum to the stated subtotal."
        ),
    )


def _inject_ae_column_sum(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Perturb a subtotal so it disagrees with the column grand total."""
    # Target the BS total_assets or total_liabilities_and_equity.
    if "balance_sheet" not in stmts:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    bs_path = "balance_sheet.current_year"
    bs = stmts.get("balance_sheet", {}).get("current_year", {})

    targets = [
        ("total_current_assets", "total_assets"),
        ("total_current_liabilities", "total_liabilities"),
        ("total_liabilities", "total_liabilities_and_equity"),
    ]
    valid = [(sub, grand) for sub, grand in targets
             if sub in bs and grand in bs]
    if not valid:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    subtotal_field, _grand_field = rng.choice(valid)
    dotpath = f"{bs_path}.{subtotal_field}"
    original = float(_get_nested(stmts, dotpath))
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.AE_COLUMN_SUM.value,
        error_category=ErrorCategory.ARITHMETIC_ERROR.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            f"Modified subtotal '{subtotal_field}' on BS so it no longer "
            f"rolls up to the column grand total."
        ),
    )


def _inject_cl_net_income_to_re(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Make net income on IS disagree with retained-earnings change on BS."""
    is_data = stmts.get("income_statement", {})
    bs_cy = stmts.get("balance_sheet", {}).get("current_year", {})
    if "net_income" not in is_data or "retained_earnings" not in bs_cy:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    dotpath = "income_statement.net_income"
    original = float(is_data["net_income"])
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.CL_NET_INCOME_TO_RE.value,
        error_category=ErrorCategory.CROSS_STATEMENT_LINKAGE.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            "Modified net income on IS so it no longer agrees with the "
            "change in retained earnings on the BS."
        ),
    )


def _inject_cl_net_income_to_cfs(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Make net income on CFS disagree with IS net income."""
    cfs = stmts.get("cash_flow_statement", {})
    is_data = stmts.get("income_statement", {})
    if "net_income" not in cfs or "net_income" not in is_data:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    dotpath = "cash_flow_statement.net_income"
    original = float(cfs["net_income"])
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.CL_NET_INCOME_TO_CFS.value,
        error_category=ErrorCategory.CROSS_STATEMENT_LINKAGE.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            "Modified net income on CFS so it no longer matches the IS."
        ),
    )


def _inject_cl_ending_cash(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Make ending cash on CFS disagree with cash on BS."""
    cfs = stmts.get("cash_flow_statement", {})
    bs_cy = stmts.get("balance_sheet", {}).get("current_year", {})
    if "ending_cash" not in cfs or "cash_and_equivalents" not in bs_cy:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    dotpath = "cash_flow_statement.ending_cash"
    original = float(cfs["ending_cash"])
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.CL_ENDING_CASH.value,
        error_category=ErrorCategory.CROSS_STATEMENT_LINKAGE.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            "Modified ending cash on CFS so it no longer matches BS cash."
        ),
    )


def _inject_cl_depreciation(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Make depreciation inconsistent between IS and CFS."""
    is_data = stmts.get("income_statement", {})
    cfs = stmts.get("cash_flow_statement", {})
    if ("depreciation_amortization" not in is_data
            or "depreciation_amortization" not in cfs):
        return InjectionResult(modified_statements=stmts, error_injected=False)

    dotpath = "cash_flow_statement.depreciation_amortization"
    original = float(cfs["depreciation_amortization"])
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.CL_DEPRECIATION.value,
        error_category=ErrorCategory.CROSS_STATEMENT_LINKAGE.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            "Modified depreciation on CFS so it is inconsistent with IS."
        ),
    )


def _inject_yoy_opening_balance(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Modify a prior-year ending balance so it differs from current-year
    opening (i.e., tweak the prior-year figure on the BS)."""
    bs = stmts.get("balance_sheet", {})
    prior = bs.get("prior_year", {})
    if not prior:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    # Pick a field present in both periods.
    current = bs.get("current_year", {})
    shared = [k for k in prior if k in current and isinstance(prior[k], (int, float))]
    if not shared:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    target_field = rng.choice(shared)
    dotpath = f"balance_sheet.prior_year.{target_field}"
    original = float(prior[target_field])
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.YOY_OPENING_BALANCE.value,
        error_category=ErrorCategory.YEAR_OVER_YEAR.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            f"Modified prior-year '{target_field}' on BS so the opening "
            f"balance no longer equals the prior-year ending balance."
        ),
    )


def _inject_yoy_computed_change(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Modify a current-year value so the implied YoY change is wrong."""
    bs = stmts.get("balance_sheet", {})
    current = bs.get("current_year", {})
    prior = bs.get("prior_year", {})
    if not current or not prior:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    shared = [k for k in current if k in prior
              and isinstance(current[k], (int, float))
              and isinstance(prior[k], (int, float))]
    if not shared:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    target_field = rng.choice(shared)
    dotpath = f"balance_sheet.current_year.{target_field}"
    original = float(current[target_field])
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=ErrorSubtype.YOY_COMPUTED_CHANGE.value,
        error_category=ErrorCategory.YEAR_OVER_YEAR.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            f"Modified current-year '{target_field}' so the computed YoY "
            f"change no longer matches."
        ),
    )


def _inject_magnitude_rounding(
    stmts: FinancialStatements,
    magnitude_pct: float,
    rng: random.Random,
) -> InjectionResult:
    """Perturb an arbitrary numeric field by a controlled percentage."""
    # Collect every numeric leaf across all statements.
    leaves: List[str] = []
    for stmt_key in ("income_statement", "cash_flow_statement"):
        section = stmts.get(stmt_key, {})
        for field_name, val in section.items():
            if isinstance(val, (int, float)):
                leaves.append(f"{stmt_key}.{field_name}")

    bs = stmts.get("balance_sheet", {}).get("current_year", {})
    for field_name, val in bs.items():
        if isinstance(val, (int, float)):
            leaves.append(f"balance_sheet.current_year.{field_name}")

    if not leaves:
        return InjectionResult(modified_statements=stmts, error_injected=False)

    dotpath = rng.choice(leaves)
    original = float(_get_nested(stmts, dotpath))
    modified = _perturb(original, magnitude_pct, rng=rng)
    _set_nested(stmts, dotpath, modified)

    subtype = _magnitude_subtype_for_pct(magnitude_pct)

    return InjectionResult(
        modified_statements=stmts,
        error_injected=True,
        error_type=subtype.value,
        error_category=ErrorCategory.MAGNITUDE_ROUNDING.value,
        error_location=dotpath,
        error_magnitude_pct=round((modified - original) / abs(original) * 100, 4)
        if original != 0 else magnitude_pct,
        original_value=original,
        modified_value=modified,
        description=(
            f"Perturbed '{dotpath.split('.')[-1]}' by ~{magnitude_pct}% "
            f"(magnitude/rounding error)."
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_INJECTORS = {
    ErrorSubtype.AE_ROW_SUM: _inject_ae_row_sum,
    ErrorSubtype.AE_COLUMN_SUM: _inject_ae_column_sum,
    ErrorSubtype.CL_NET_INCOME_TO_RE: _inject_cl_net_income_to_re,
    ErrorSubtype.CL_NET_INCOME_TO_CFS: _inject_cl_net_income_to_cfs,
    ErrorSubtype.CL_ENDING_CASH: _inject_cl_ending_cash,
    ErrorSubtype.CL_DEPRECIATION: _inject_cl_depreciation,
    ErrorSubtype.YOY_OPENING_BALANCE: _inject_yoy_opening_balance,
    ErrorSubtype.YOY_COMPUTED_CHANGE: _inject_yoy_computed_change,
    ErrorSubtype.MR_MINOR: _inject_magnitude_rounding,
    ErrorSubtype.MR_MODERATE: _inject_magnitude_rounding,
    ErrorSubtype.MR_SIGNIFICANT: _inject_magnitude_rounding,
    ErrorSubtype.MR_EXTREME: _inject_magnitude_rounding,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_error(
    statements: FinancialStatements,
    subtype: ErrorSubtype,
    magnitude_pct: float = 5.0,
    seed: Optional[int] = None,
) -> InjectionResult:
    """Inject a single error of *subtype* into *statements*.

    Parameters
    ----------
    statements:
        A clean financial-statement dict (see module docstring for schema).
        A **deep copy** is made internally; the caller's data is never mutated.
    subtype:
        Which error to inject — must be a member of ``ErrorSubtype``.
    magnitude_pct:
        Percentage deviation to apply.  For ``MR_*`` subtypes this also
        determines which magnitude bucket the error falls into; for other
        subtypes it controls how large the discrepancy is.
        Common values: ``0.5``, ``1.0``, ``2.0``, ``5.0``, ``10.0``, ``20.0``.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    InjectionResult
        Contains the mutated statements and full ground-truth metadata.

    Raises
    ------
    ValueError
        If *subtype* is not a recognised ``ErrorSubtype``.
    """
    if subtype not in _INJECTORS:
        raise ValueError(f"No injector registered for subtype {subtype!r}")

    rng = random.Random(seed)
    mutated = copy.deepcopy(statements)
    injector_fn = _INJECTORS[subtype]
    result = injector_fn(mutated, magnitude_pct, rng)
    # Ensure the result always carries the deep-copied dict.
    result.modified_statements = mutated
    return result


def inject_multiple_errors(
    statements: FinancialStatements,
    subtypes_and_magnitudes: Sequence[Tuple[ErrorSubtype, float]],
    seed: Optional[int] = None,
) -> MultiInjectionResult:
    """Inject several errors into the same statement set (harder variant).

    Errors are applied sequentially on the same deep copy, so later
    injections see the mutations of earlier ones.  This mirrors realistic
    scenarios where statements contain more than one mistake.

    Parameters
    ----------
    statements:
        Clean financial statements.
    subtypes_and_magnitudes:
        Sequence of ``(ErrorSubtype, magnitude_pct)`` pairs to inject in
        order.
    seed:
        Optional RNG seed.

    Returns
    -------
    MultiInjectionResult
        Aggregated result with the final mutated statements and per-error
        ground truth.
    """
    rng = random.Random(seed)
    mutated = copy.deepcopy(statements)
    injections: List[InjectionResult] = []

    for subtype, mag in subtypes_and_magnitudes:
        if subtype not in _INJECTORS:
            raise ValueError(
                f"No injector registered for subtype {subtype!r}")
        # Each injector gets its own sub-seed for determinism.
        sub_seed = rng.randint(0, 2**31)
        sub_rng = random.Random(sub_seed)
        result = _INJECTORS[subtype](mutated, mag, sub_rng)
        result.modified_statements = mutated
        injections.append(result)

    return MultiInjectionResult(
        modified_statements=mutated,
        injections=injections,
    )
