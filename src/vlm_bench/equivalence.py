"""Conservative, dependency-free mathematical expression comparison.

The checker proves equivalence for identical valid expressions and for exact
identities in a small bounded rational-polynomial subset. It may prove
difference when a safe numeric witness in an explicitly declared test domain
gives different values. All other cases remain undetermined; finite sampling
is never used as proof of equivalence.
"""

from __future__ import annotations

import ast
import itertools
import math
import operator
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from .research import normalize_equation

EQUIVALENCE_VERSION = 2
_MAX_ASSIGNMENTS = 512
_MAX_POWER = 12
_MAX_SOURCE_LENGTH = 1_000
_MAX_AST_NODES = 128
_MAX_AST_DEPTH = 32
_MAX_POLYNOMIAL_TERMS = 256
_MAX_POLYNOMIAL_DEGREE = 32
_MAX_DOMAIN_VALUES = _MAX_ASSIGNMENTS


class _UnsupportedExpression(ValueError):
    pass


def _safe_number(value: Any) -> Fraction | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Fraction(str(value))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if abs(number.numerator) > 10**18 or number.denominator > 10**18:
        return None
    return number


def _variable_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _parse(expression: str) -> ast.Expression:
    # This supports ordinary arithmetic notation only. In particular, it does
    # not rewrite TeX or invoke any parser that could evaluate model output.
    source = expression.replace("^", "**")
    if len(source) > _MAX_SOURCE_LENGTH:
        raise _UnsupportedExpression("expression exceeds the safe length limit")
    try:
        parsed = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise _UnsupportedExpression("unsupported notation or parse failure") from exc
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    if any(not isinstance(item, allowed) for item in ast.walk(parsed)):
        raise _UnsupportedExpression("expression uses unsupported syntax")
    nodes = list(ast.walk(parsed))
    if len(nodes) > _MAX_AST_NODES:
        raise _UnsupportedExpression("expression exceeds the safe syntax limit")
    if _ast_depth(parsed) > _MAX_AST_DEPTH:
        raise _UnsupportedExpression("expression exceeds the safe nesting limit")
    for item in ast.walk(parsed):
        if isinstance(item, ast.Constant) and _safe_number(item.value) is None:
            raise _UnsupportedExpression("expression contains a non-finite or non-numeric constant")
    return parsed


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((_ast_depth(child) for child in children), default=0)


def _evaluate(node: ast.AST, values: Mapping[str, Fraction]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, values)
    if isinstance(node, ast.Constant):
        number = _safe_number(node.value)
        if number is None:
            raise _UnsupportedExpression("non-numeric constant")
        return number
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise _UnsupportedExpression(f"no domain values supplied for variable {node.id!r}")
        return values[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, values)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, values)
        right = _evaluate(node.right, values)
        if isinstance(node.op, ast.Add):
            result = operator.add(left, right)
        elif isinstance(node.op, ast.Sub):
            result = operator.sub(left, right)
        elif isinstance(node.op, ast.Mult):
            result = operator.mul(left, right)
        elif isinstance(node.op, ast.Div):
            result = operator.truediv(left, right)
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > _MAX_POWER:
                raise _UnsupportedExpression("power exceeds the safe numeric check limit")
            if left == 0 and right.numerator == 0:
                raise _UnsupportedExpression("zero to the zero power is mathematically ambiguous")
            result = operator.pow(left, right.numerator)
        else:  # pragma: no cover - ast whitelist prevents this branch
            raise _UnsupportedExpression("unsupported operator")
        if abs(result.numerator) > 10**18 or result.denominator > 10**18:
            raise ArithmeticError("numeric result exceeds the safe exact-arithmetic limit")
        return result
    raise _UnsupportedExpression("unsupported expression node")


# A monomial is a sorted tuple of (variable, exponent) pairs. The polynomial
# representation is deliberately restricted to exact rational coefficients;
# no floating-point simplification or numeric sampling is used for proofs.
_Monomial = tuple[tuple[str, int], ...]
_Polynomial = dict[_Monomial, Fraction]
_CONSTANT_MONOMIAL: _Monomial = ()


def _bounded_coefficient(value: Fraction) -> Fraction:
    if abs(value.numerator) > 10**18 or value.denominator > 10**18:
        raise _UnsupportedExpression("polynomial coefficient exceeds the safe exact limit")
    return value


def _clean_polynomial(polynomial: _Polynomial) -> _Polynomial:
    cleaned: _Polynomial = {}
    for monomial, coefficient in polynomial.items():
        if coefficient == 0:
            continue
        if sum(power for _, power in monomial) > _MAX_POLYNOMIAL_DEGREE:
            raise _UnsupportedExpression("polynomial degree exceeds the safe identity limit")
        cleaned[monomial] = _bounded_coefficient(coefficient)
    if len(cleaned) > _MAX_POLYNOMIAL_TERMS:
        raise _UnsupportedExpression("polynomial exceeds the safe identity term limit")
    return cleaned


def _polynomial_add(left: _Polynomial, right: _Polynomial, factor: int = 1) -> _Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, Fraction(0)) + factor * coefficient
    return _clean_polynomial(result)


def _polynomial_multiply(left: _Polynomial, right: _Polynomial) -> _Polynomial:
    result: _Polynomial = {}
    for left_monomial, left_coefficient in left.items():
        for right_monomial, right_coefficient in right.items():
            powers: dict[str, int] = dict(left_monomial)
            for name, power in right_monomial:
                powers[name] = powers.get(name, 0) + power
            monomial = tuple(sorted((name, power) for name, power in powers.items() if power))
            if sum(power for _, power in monomial) > _MAX_POLYNOMIAL_DEGREE:
                raise _UnsupportedExpression("polynomial degree exceeds the safe identity limit")
            result[monomial] = (
                result.get(monomial, Fraction(0)) + left_coefficient * right_coefficient
            )
            # Bound intermediate growth before continuing through a potentially
            # combinatorial expansion.
            _bounded_coefficient(result[monomial])
            if len(result) > _MAX_POLYNOMIAL_TERMS:
                raise _UnsupportedExpression("polynomial exceeds the safe identity term limit")
    return _clean_polynomial(result)


def _constant_polynomial(value: Fraction) -> _Polynomial:
    return {} if value == 0 else {_CONSTANT_MONOMIAL: _bounded_coefficient(value)}


def _polynomial(node: ast.AST) -> _Polynomial:
    if isinstance(node, ast.Expression):
        return _polynomial(node.body)
    if isinstance(node, ast.Constant):
        number = _safe_number(node.value)
        if number is None:
            raise _UnsupportedExpression("non-numeric constant")
        return _constant_polynomial(number)
    if isinstance(node, ast.Name):
        return {((node.id, 1),): Fraction(1)}
    if isinstance(node, ast.UnaryOp):
        operand = _polynomial(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return _clean_polynomial({key: -value for key, value in operand.items()})
    if isinstance(node, ast.BinOp):
        left = _polynomial(node.left)
        right = _polynomial(node.right)
        if isinstance(node.op, ast.Add):
            return _polynomial_add(left, right)
        if isinstance(node.op, ast.Sub):
            return _polynomial_add(left, right, factor=-1)
        if isinstance(node.op, ast.Mult):
            return _polynomial_multiply(left, right)
        if isinstance(node.op, ast.Div):
            if len(right) > 1 or (right and _CONSTANT_MONOMIAL not in right):
                raise _UnsupportedExpression(
                    "polynomial identities allow division by constants only"
                )
            divisor = right.get(_CONSTANT_MONOMIAL, Fraction(0))
            if divisor == 0:
                raise _UnsupportedExpression("division by zero has no defined identity domain")
            return _clean_polynomial(
                {key: coefficient / divisor for key, coefficient in left.items()}
            )
        if isinstance(node.op, ast.Pow):
            if len(right) > 1 or (right and _CONSTANT_MONOMIAL not in right):
                raise _UnsupportedExpression("polynomial powers must be non-negative integers")
            exponent = right.get(_CONSTANT_MONOMIAL, Fraction(0))
            if exponent.denominator != 1 or not 0 <= exponent.numerator <= _MAX_POWER:
                raise _UnsupportedExpression("polynomial power exceeds the safe identity limit")
            if exponent.numerator == 0 and not left:
                raise _UnsupportedExpression("zero to the zero power is mathematically ambiguous")
            result: _Polynomial = _constant_polynomial(Fraction(1))
            for _ in range(exponent.numerator):
                result = _polynomial_multiply(result, left)
            return result
    raise _UnsupportedExpression("expression is outside the exact polynomial subset")


def _domain_values(
    names: set[str], variable_domains: Mapping[str, Any] | None
) -> tuple[dict[str, tuple[Fraction, ...]], str]:
    supplied = variable_domains or {}
    if not isinstance(supplied, Mapping):
        raise TypeError("variable_domains must be a mapping from names to numeric values")
    domains: dict[str, tuple[Fraction, ...]] = {}
    for name in sorted(names):
        raw = supplied.get(name)
        if raw is None:
            raise ValueError(f"explicit variable domain values are required for {name!r}")
        if isinstance(raw, (str, bytes)):
            raise TypeError(f"domain for {name!r} must be numeric values")
        if isinstance(raw, Sequence) and len(raw) > _MAX_DOMAIN_VALUES:
            raise ValueError(f"domain for {name!r} exceeds the {_MAX_DOMAIN_VALUES}-value limit")
        candidates = raw if isinstance(raw, Sequence) else (raw,)
        parsed = tuple(number for item in candidates if (number := _safe_number(item)) is not None)
        values = parsed
        if not values:
            raise ValueError(f"domain for {name!r} must contain at least one finite number")
        domains[name] = tuple(values)
    return domains, "declared_values" if names else "constant_expression"


def _display_number(value: Fraction) -> int | str:
    if value.denominator == 1:
        return value.numerator
    return f"{value.numerator}/{value.denominator}"


def compare_equations(
    prediction: str,
    reference: str,
    variable_domains: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``equivalent``, ``different``, or ``undetermined`` conservatively.

    ``variable_domains`` maps variable names to finite witness values, such as
    ``{"x": [-2, -1, 1, 2]}``. A differing value is a concrete counterexample
    within those supplied points. The result does not claim symbolic proof over
    a continuous interval.
    """
    if not isinstance(prediction, str) or not isinstance(reference, str):
        raise TypeError("prediction and reference must be strings")
    normalized_prediction = normalize_equation(prediction)
    normalized_reference = normalize_equation(reference)
    result_base = {
        "version": EQUIVALENCE_VERSION,
        "prediction_normalized": normalized_prediction,
        "reference_normalized": normalized_reference,
        "domain_mode": "unspecified",
        "witness": None,
        "reason": None,
    }
    if not normalized_prediction or not normalized_reference:
        return {
            **result_base,
            "status": "undetermined",
            "reason": "empty_expression",
        }
    try:
        parsed_prediction = _parse(normalized_prediction)
        parsed_reference = _parse(normalized_reference)
        prediction_polynomial = reference_polynomial = None
        polynomial_error: _UnsupportedExpression | None = None
        try:
            prediction_polynomial = _polynomial(parsed_prediction)
            reference_polynomial = _polynomial(parsed_reference)
        except _UnsupportedExpression as exc:
            # Numeric witnesses still apply to safe expressions outside the
            # polynomial identity subset, but they never establish identity.
            polynomial_error = exc
        if prediction_polynomial is not None and prediction_polynomial == reference_polynomial:
            return {
                **result_base,
                "status": "equivalent",
                "domain_mode": "exact_rational_polynomial_identity",
                "reason": "same_exact_polynomial",
            }
        if normalized_prediction == normalized_reference:
            # Identical supported non-polynomial expressions are equal on
            # their common defined domain. Still reject expressions that are
            # undefined everywhere, such as x / 0.
            if polynomial_error is not None and any(
                marker in str(polynomial_error) for marker in ("division by zero", "ambiguous")
            ):
                raise polynomial_error
            return {
                **result_base,
                "status": "equivalent",
                "domain_mode": "literal_normalized_identity",
                "reason": "same_normalized_expression",
            }
        names = _variable_names(parsed_prediction) | _variable_names(parsed_reference)
        domains, domain_mode = _domain_values(names, variable_domains)
    except (TypeError, ValueError, _UnsupportedExpression) as exc:
        return {
            **result_base,
            "status": "undetermined",
            "reason": str(exc),
        }

    combinations = math.prod(len(values) for values in domains.values()) if domains else 1
    if combinations > _MAX_ASSIGNMENTS:
        return {
            **result_base,
            "status": "undetermined",
            "domain_mode": domain_mode,
            "reason": f"domain has {combinations} combinations; limit is {_MAX_ASSIGNMENTS}",
        }
    assignments = itertools.product(*(domains[name] for name in domains)) if domains else [()]
    for assignment in assignments:
        values = dict(zip(domains, assignment, strict=True))
        try:
            predicted_value = _evaluate(parsed_prediction, values)
            reference_value = _evaluate(parsed_reference, values)
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            continue
        if predicted_value != reference_value:
            return {
                **result_base,
                "status": "different",
                "domain_mode": domain_mode,
                "witness": {
                    "variables": {name: _display_number(value) for name, value in values.items()},
                    "prediction_value": _display_number(predicted_value),
                    "reference_value": _display_number(reference_value),
                },
                "reason": "numeric_counterexample",
            }
    return {
        **result_base,
        "status": "undetermined",
        "domain_mode": domain_mode,
        "reason": "no_counterexample_found; finite evaluation cannot prove equivalence",
    }


# Singular spelling is convenient for callers and retained as the primary
# entry point used by the report evaluator.
compare_equation = compare_equations
