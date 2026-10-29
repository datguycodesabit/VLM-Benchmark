from vlm_bench.equivalence import compare_equations


def test_identical_valid_equation_is_equivalent_but_identical_unsupported_text_is_not():
    assert compare_equations(" x ^ 2 ", "x ^ 2")["status"] == "equivalent"
    assert compare_equations(r"\frac{x}{2}", r"\frac{x}{2}")["status"] == "undetermined"
    assert compare_equations("hello(", "hello(")["status"] == "undetermined"


def test_counterexample_requires_explicit_domains_and_never_proves_equivalence_by_sampling():
    assert compare_equations("x", "x + 1")["status"] == "undetermined"
    different = compare_equations("x", "x + 1", {"x": [0, 1, 2]})
    sampled_identity = compare_equations("x + 1", "1 + x", {"x": [0, 1, 2]})

    assert different["status"] == "different"
    assert different["witness"]["variables"] == {"x": 0}
    assert sampled_identity["status"] == "undetermined"


def test_equivalence_refuses_unbounded_or_unsupported_domains():
    huge = {"x": list(range(513))}
    assert compare_equations("x", "x + 1", huge)["status"] == "undetermined"
    assert compare_equations("x + y", "x - y", {"x": [1]})["status"] == "undetermined"
