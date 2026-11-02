from vlm_bench.equivalence import compare_equations


def test_identical_valid_equation_is_equivalent_but_identical_unsupported_text_is_not():
    assert compare_equations(" x ^ 2 ", "x ^ 2")["status"] == "equivalent"
    assert compare_equations(r"\frac{x}{2}", r"\frac{x}{2}")["status"] == "undetermined"
    assert compare_equations("hello(", "hello(")["status"] == "undetermined"


def test_counterexample_requires_explicit_domains_and_never_proves_equivalence_by_sampling():
    assert compare_equations("x", "x + 1")["status"] == "undetermined"
    different = compare_equations("x", "x + 1", {"x": [0, 1, 2]})
    sampled_identity = compare_equations("x^2", "x", {"x": [0, 1]})

    assert different["status"] == "different"
    assert different["witness"]["variables"] == {"x": 0}
    assert sampled_identity["status"] == "undetermined"


def test_exact_rational_polynomial_identities_are_proved_without_sampling():
    expanded = compare_equations("(x + 1)^2", "x^2 + 2*x + 1")
    reordered = compare_equations("x + y", "y + x")
    divided = compare_equations("(x + 1)/2", "x/2 + 1/2")

    assert expanded["status"] == "equivalent"
    assert expanded["version"] == 2
    assert expanded["domain_mode"] == "exact_rational_polynomial_identity"
    assert reordered["status"] == "equivalent"
    assert divided["status"] == "equivalent"


def test_polynomial_proofs_reject_undefined_and_resource_exhausting_inputs():
    assert compare_equations("x / 0", "x / 0")["status"] == "undetermined"
    assert compare_equations("0^0", "0^0")["status"] == "undetermined"
    assert compare_equations("0^0", "2")["status"] == "undetermined"
    assert compare_equations("(x + y)^12", "(x + y)^12")["status"] == "equivalent"
    assert compare_equations("(x + y)^12", "(x^12 + y^12)")["status"] == "undetermined"


def test_equivalence_refuses_unbounded_or_unsupported_domains():
    huge = {"x": list(range(513))}
    assert compare_equations("x", "x + 1", huge)["status"] == "undetermined"
    assert compare_equations("x + y", "x - y", {"x": [1]})["status"] == "undetermined"
