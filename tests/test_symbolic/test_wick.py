"""
Tests for the symbolic propagator and Gaussian Wick engine.
"""

import sympy as sp
import pytest

from tntlib.symbolic import SymbolicPropagator, WickEngine


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def symbols():
    i, j, k, t, q, tau = sp.symbols("i j k t q tau")
    return {
        "i": i,
        "j": j,
        "k": k,
        "t": t,
        "q": q,
        "tau": tau,
    }


@pytest.fixture
def wick():
    return WickEngine()


# ============================================================
# SymbolicPropagator: construction and access
# ============================================================

def test_scalar_propagator_construction(symbols):
    k, t = symbols["k"], symbols["t"]
    expr = sp.exp(-k**2 * t)

    prop = SymbolicPropagator(expr, momentum=k, time=t)

    assert prop.expr == expr
    assert prop.momentum == k
    assert prop.time == t
    assert prop.name == "G"
    assert prop.is_matrix is False


def test_matrix_propagator_construction():
    a, b = sp.symbols("a b")
    matrix = sp.Matrix([[a, b], [b, a]])

    prop = SymbolicPropagator(matrix)

    assert prop.is_matrix is True
    assert prop.expr == matrix
    assert prop.expr.shape == (2, 2)


def test_list_input_is_converted_to_matrix():
    prop = SymbolicPropagator([[1, 2], [3, 4]])

    assert prop.is_matrix is True
    assert prop.expr == sp.Matrix([[1, 2], [3, 4]])


def test_non_square_matrix_is_rejected():
    with pytest.raises(ValueError, match="square"):
        SymbolicPropagator(sp.Matrix([[1, 2, 3], [4, 5, 6]]))


def test_scalar_element_returns_explicit_expression():
    x = sp.symbols("x")
    prop = SymbolicPropagator(x**2 + 1)

    assert prop.element() == x**2 + 1


def test_matrix_element_returns_requested_entry():
    a, b, c, d = sp.symbols("a b c d")
    prop = SymbolicPropagator([[a, b], [c, d]])

    assert prop.element(0, 1) == b
    assert prop.element(1, 0) == c


def test_matrix_element_requires_both_indices():
    prop = SymbolicPropagator(sp.eye(2))

    with pytest.raises(ValueError, match="require i and j"):
        prop.element(0)

    with pytest.raises(ValueError, match="require i and j"):
        prop.element()


def test_symbolic_element_uses_configured_function_name():
    i, k, t = sp.symbols("i k t")
    prop = SymbolicPropagator(0, name="D")

    assert prop.symbolic_element(i, k, t) == sp.Function("D")(i, k, t)


# ============================================================
# SymbolicPropagator: registering and verifying symmetries
# ============================================================

@pytest.mark.parametrize(
    "sign, expression, expected",
    [
        (1, sp.cos(sp.Symbol("k")), True),
        (-1, sp.sin(sp.Symbol("k")), True),
        (1, sp.sin(sp.Symbol("k")), False),
        (-1, sp.cos(sp.Symbol("k")), False),
    ],
)
def test_momentum_parity_verification(sign, expression, expected):
    k = sp.Symbol("k")
    prop = SymbolicPropagator(expression, momentum=k)
    prop.add_momentum_parity(sign=sign)

    result = prop.verify_symmetries()

    assert result[0]["valid"] is expected
    assert result[0]["rule"]["kind"] == "momentum"


@pytest.mark.parametrize(
    "sign, expression, expected",
    [
        (1, sp.cos(sp.Symbol("t")), True),
        (-1, sp.sin(sp.Symbol("t")), True),
        (1, sp.sin(sp.Symbol("t")), False),
        (-1, sp.cos(sp.Symbol("t")), False),
    ],
)
def test_time_parity_verification(sign, expression, expected):
    t = sp.Symbol("t")
    prop = SymbolicPropagator(expression, time=t)
    prop.add_time_parity(sign=sign)

    assert prop.verify_symmetries()[0]["valid"] is expected


def test_matrix_momentum_parity_checks_every_component():
    k = sp.Symbol("k")
    prop = SymbolicPropagator(
        sp.Matrix([[sp.cos(k), sp.sin(k)], [0, sp.cos(k)]]),
        momentum=k,
    )
    prop.add_momentum_parity(+1)

    assert prop.verify_symmetries()[0]["valid"] is False


def test_matrix_momentum_parity_passes_when_all_components_match():
    k = sp.Symbol("k")
    prop = SymbolicPropagator(
        sp.Matrix([[sp.cos(k), k**2], [0, 1]]),
        momentum=k,
    )
    prop.add_momentum_parity(+1)

    assert prop.verify_symmetries()[0]["valid"] is True


def test_index_symmetry_verification_symmetric_matrix():
    a, b = sp.symbols("a b")
    prop = SymbolicPropagator([[a, b], [b, a]])
    prop.add_index_symmetry(0, 1, sign=+1)

    assert prop.verify_symmetries()[0]["valid"] is True


def test_index_symmetry_verification_antisymmetric_matrix():
    a = sp.symbols("a")
    prop = SymbolicPropagator([[0, a], [-a, 0]])
    prop.add_index_symmetry(0, 1, sign=-1)

    assert prop.verify_symmetries()[0]["valid"] is True


def test_invalid_missing_momentum_variable_is_rejected():
    prop = SymbolicPropagator(sp.Symbol("x"))

    with pytest.raises(ValueError, match="momentum variable"):
        prop.add_momentum_parity()


def test_invalid_missing_time_variable_is_rejected():
    prop = SymbolicPropagator(sp.Symbol("x"))

    with pytest.raises(ValueError, match="time variable"):
        prop.add_time_parity()


@pytest.mark.parametrize("sign", [0, 2, -2, "1", None])
def test_invalid_argument_symmetry_sign_is_rejected(sign):
    k = sp.Symbol("k")
    prop = SymbolicPropagator(k, momentum=k)

    with pytest.raises(ValueError, match=r"sign must be \+1 or -1"):
        prop.add_momentum_parity(sign=sign)


def test_index_symmetry_is_rejected_for_scalar_propagator():
    prop = SymbolicPropagator(sp.Symbol("x"))

    with pytest.raises(ValueError, match="requires a matrix"):
        prop.add_index_symmetry(0, 1)


def test_invalid_index_symmetry_sign_is_rejected():
    prop = SymbolicPropagator(sp.eye(2))

    with pytest.raises(ValueError, match=r"sign must be \+1 or -1"):
        prop.add_index_symmetry(0, 1, sign=0)


def test_multiple_registered_symmetries_are_all_reported():
    k, t = sp.symbols("k t")
    prop = SymbolicPropagator(sp.cos(k) * sp.cos(t), momentum=k, time=t)
    prop.add_momentum_parity(+1)
    prop.add_time_parity(+1)

    results = prop.verify_symmetries()

    assert len(results) == 2
    assert all(item["valid"] for item in results.values())


# ============================================================
# SymbolicPropagator: applying symmetries to abstract objects
# ============================================================

def test_apply_even_momentum_parity_canonicalizes_negative_argument():
    k, t = sp.symbols("k t")
    prop = SymbolicPropagator(sp.cos(k), momentum=k)
    prop.add_momentum_parity(+1)

    G = prop.symbolic_element
    expression = G(-k, t)

    assert prop.apply_symmetries(expression) == G(k, t)


def test_apply_odd_momentum_parity_adds_minus_sign():
    k, t = sp.symbols("k t")
    prop = SymbolicPropagator(sp.sin(k), momentum=k)
    prop.add_momentum_parity(-1)

    G = prop.symbolic_element

    assert prop.apply_symmetries(G(-k, t)) == -G(k, t)


def test_apply_even_time_parity_canonicalizes_negative_argument():
    k, t = sp.symbols("k t")
    prop = SymbolicPropagator(sp.cos(t), time=t)
    prop.add_time_parity(+1)

    G = prop.symbolic_element

    assert prop.apply_symmetries(G(k, -t)) == G(k, t)


def test_apply_odd_time_parity_adds_minus_sign():
    k, t = sp.symbols("k t")
    prop = SymbolicPropagator(sp.sin(t), time=t)
    prop.add_time_parity(-1)

    G = prop.symbolic_element

    assert prop.apply_symmetries(G(k, -t)) == -G(k, t)


def test_apply_index_symmetry_swaps_matrix_indices():
    i, j, k, t = sp.symbols("i j k t")
    prop = SymbolicPropagator(sp.eye(2))
    prop.add_index_symmetry(i, j, sign=+1)

    G = prop.symbolic_element

    assert prop.apply_symmetries(G(i, j, k, t)) == G(j, i, k, t)


def test_apply_antisymmetric_index_symmetry_adds_minus_sign():
    i, j, k, t = sp.symbols("i j k t")
    prop = SymbolicPropagator(sp.Matrix([[0, 1], [-1, 0]]))
    prop.add_index_symmetry(i, j, sign=-1)

    G = prop.symbolic_element

    assert prop.apply_symmetries(G(i, j, k, t)) == -G(j, i, k, t)


def test_apply_symmetries_leaves_unrelated_functions_unchanged():
    k, t, x = sp.symbols("k t x")
    prop = SymbolicPropagator(sp.cos(k), momentum=k)
    prop.add_momentum_parity(+1)

    H = sp.Function("H")
    expression = H(-k) + prop.symbolic_element(-k, t)

    expected = H(-k) + prop.symbolic_element(k, t)

    assert prop.apply_symmetries(expression) == expected


def test_apply_symmetries_preserves_expression_coefficients():
    k, t = sp.symbols("k t")
    prop = SymbolicPropagator(sp.sin(k), momentum=k)
    prop.add_momentum_parity(-1)

    G = prop.symbolic_element

    assert prop.apply_symmetries(3 * G(-k, t)) == -3 * G(k, t)


def test_propagator_display_returns_self(capsys):
    prop = SymbolicPropagator(sp.Symbol("x")**2, name="D")

    returned = prop.display()
    output = capsys.readouterr().out

    assert returned is prop
    assert "Propagator: D" in output
    assert "Registered symmetries" in output


# ============================================================
# WickEngine: fields and contractions
# ============================================================

def test_default_field_name_and_field_at(wick):
    i, k, t = sp.symbols("i k t")

    assert wick.field_at(i, k, t) == sp.Function("Psi")(i, k, t)


def test_custom_field_name():
    engine = WickEngine(field_name="phi")
    x = sp.Symbol("x")

    assert engine.field_at(x) == sp.Function("phi")(x)


def test_canonical_pair_is_independent_of_order(wick):
    a, b = sp.symbols("a b")

    assert wick._canonical_pair(a, b) == wick._canonical_pair(b, a)


def test_add_and_retrieve_contraction_in_both_orders(wick):
    a, b, C = sp.symbols("a b C")
    wick.add_contraction(a, b, C)

    assert wick.contraction(a, b) == C
    assert wick.contraction(b, a) == C


def test_contraction_is_sympified(wick):
    a, b = sp.symbols("a b")
    wick.add_contraction(a, b, "2")

    assert wick.contraction(a, b) == sp.Integer(2)


def test_missing_contraction_raises_key_error(wick):
    a, b = sp.symbols("a b")

    with pytest.raises(KeyError, match="No contraction registered"):
        wick.contraction(a, b)


def test_registering_new_contraction_clears_wick_cache(wick):
    a, b, c, d = sp.symbols("a b c d")
    wick.add_contraction(a, b, 2)
    wick._cache[()] = sp.Integer(99)

    wick.add_contraction(c, d, 3)

    assert wick._cache == {}


# ============================================================
# WickEngine: extraction and recursive pairings
# ============================================================

def test_extract_fields_from_product(wick):
    a, b = sp.symbols("a b")
    A, B = wick.field_at(a), wick.field_at(b)

    fields = wick._extract_fields(A**2 * B)

    assert fields == [A, A, B]


def test_extract_fields_returns_empty_for_term_without_fields(wick):
    x = sp.Symbol("x")

    assert wick._extract_fields(3 * x**2) == []


@pytest.mark.parametrize("power", [-1, sp.Rational(1, 2), sp.Rational(3, 2)])
def test_extract_fields_rejects_negative_or_fractional_powers(wick, power):
    a = sp.Symbol("a")
    A = wick.field_at(a)

    with pytest.raises(ValueError, match="nonnegative integer powers"):
        wick._extract_fields(A**power)


def test_wick_product_of_no_fields_is_one(wick):
    assert wick._wick_product([]) == 1


def test_wick_product_of_odd_number_of_fields_is_zero(wick):
    a, b, c = sp.symbols("a b c")
    A, B, C = (wick.field_at(x) for x in (a, b, c))
    wick.add_contraction(A, B, 7)
    wick.add_contraction(A, C, 11)
    wick.add_contraction(B, C, 13)

    assert wick._wick_product([A, B, C]) == 0


def test_wick_product_of_two_fields_is_the_covariance(wick):
    a, b, C = sp.symbols("a b C")
    A, B = wick.field_at(a), wick.field_at(b)
    wick.add_contraction(A, B, C)

    assert wick._wick_product([A, B]) == C


def test_wick_product_of_four_distinct_fields(wick):
    a, b, c, d = sp.symbols("a b c d")
    A, B, C, D = (wick.field_at(x) for x in (a, b, c, d))

    # Wick's theorem: AB CD + AC BD + AD BC.
    covariances = {
        (A, B): 2,
        (A, C): 3,
        (A, D): 5,
        (B, C): 7,
        (B, D): 11,
        (C, D): 13,
    }
    for (left, right), value in covariances.items():
        wick.add_contraction(left, right, value)

    expected = 2 * 13 + 3 * 11 + 5 * 7

    assert wick._wick_product([A, B, C, D]) == expected


def test_wick_product_of_four_identical_fields_is_three_sigma_squared(wick):
    a = sp.Symbol("a")
    A = wick.field_at(a)
    wick.add_contraction(A, A, sp.Symbol("C"))

    assert wick._wick_product([A, A, A, A]) == 3 * sp.Symbol("C")**2


def test_pairings_empty_input_has_one_empty_pairing(wick):
    assert wick.pairings([]) == [[]]


def test_pairings_odd_number_of_fields_is_empty(wick):
    a, b, c = sp.symbols("a b c")
    fields = [wick.field_at(x) for x in (a, b, c)]

    assert wick.pairings(fields) == []


@pytest.mark.parametrize(
    "number_of_fields, expected_count",
    [(2, 1), (4, 3), (6, 15), (8, 105)],
)
def test_pairings_count_matches_double_factorial(
    wick, number_of_fields, expected_count
):
    fields = [
        wick.field_at(sp.Symbol(f"x{n}"))
        for n in range(number_of_fields)
    ]

    pairings = wick.pairings(fields)

    assert len(pairings) == expected_count
    assert all(len(pairing) == number_of_fields // 2 for pairing in pairings)


def test_each_pairing_contains_every_field_exactly_once(wick):
    a, b, c, d = sp.symbols("a b c d")
    fields = [wick.field_at(x) for x in (a, b, c, d)]

    for pairing in wick.pairings(fields):
        flattened = [field for pair in pairing for field in pair]
        assert sorted(flattened, key=sp.default_sort_key) == sorted(
            fields, key=sp.default_sort_key
        )


# ============================================================
# WickEngine: Gaussian averages
# ============================================================

def test_average_of_constant_returns_constant(wick):
    x = sp.Symbol("x")

    assert wick.average(3 * x**2) == 3 * x**2


def test_average_of_single_field_is_zero(wick):
    a = sp.Symbol("a")
    A = wick.field_at(a)

    assert wick.average(A) == 0


def test_average_of_two_fields_is_covariance(wick):
    a, b, C = sp.symbols("a b C")
    A, B = wick.field_at(a), wick.field_at(b)
    wick.add_contraction(A, B, C)

    assert wick.average(A * B) == C


def test_average_of_two_fields_is_commutative(wick):
    a, b, C = sp.symbols("a b C")
    A, B = wick.field_at(a), wick.field_at(b)
    wick.add_contraction(A, B, C)

    assert wick.average(A * B) == wick.average(B * A)


def test_average_of_four_fields_uses_all_three_pairings(wick):
    a, b, c, d = sp.symbols("a b c d")
    A, B, C, D = (wick.field_at(x) for x in (a, b, c, d))

    cov = {
        (A, B): sp.Symbol("C_ab"),
        (A, C): sp.Symbol("C_ac"),
        (A, D): sp.Symbol("C_ad"),
        (B, C): sp.Symbol("C_bc"),
        (B, D): sp.Symbol("C_bd"),
        (C, D): sp.Symbol("C_cd"),
    }
    for (left, right), value in cov.items():
        wick.add_contraction(left, right, value)

    expected = (
        cov[(A, B)] * cov[(C, D)]
        + cov[(A, C)] * cov[(B, D)]
        + cov[(A, D)] * cov[(B, C)]
    )

    assert sp.expand(wick.average(A * B * C * D) - expected) == 0


def test_average_of_six_identical_fields_is_fifteen_sigma_cubed(wick):
    a = sp.Symbol("a")
    A = wick.field_at(a)
    C = sp.Symbol("C")
    wick.add_contraction(A, A, C)

    assert wick.average(A**6) == 15 * C**3


def test_average_of_odd_moment_is_zero(wick):
    a = sp.Symbol("a")
    A = wick.field_at(a)

    assert wick.average(A**5) == 0


def test_average_of_sum_is_linear(wick):
    a, b, C_ab, C_aa, C_bb = sp.symbols("a b C_ab C_aa C_bb")
    A, B = wick.field_at(a), wick.field_at(b)
    wick.add_contraction(A, A, C_aa)
    wick.add_contraction(B, B, C_bb)
    wick.add_contraction(A, B, C_ab)

    expression = 2 * A * B + 3 * A**2 + B**2

    expected = 2 * C_ab + 3 * C_aa + C_bb

    assert wick.average(expression) == expected


def test_average_preserves_non_field_prefactors(wick):
    a, x, C = sp.symbols("a x C")
    A = wick.field_at(a)
    wick.add_contraction(A, A, C)

    assert wick.average(x * A**2) == x * C


def test_average_of_sum_with_constant_term(wick):
    a, C, x = sp.symbols("a C x")
    A = wick.field_at(a)
    wick.add_contraction(A, A, C)

    assert wick.average(A**2 + x) == C + x


def test_average_raises_if_required_contraction_is_missing(wick):
    a, b = sp.symbols("a b")
    A, B = wick.field_at(a), wick.field_at(b)

    with pytest.raises(KeyError, match="No contraction registered"):
        wick.average(A * B)


def test_average_expand_flag_false_still_handles_a_product(wick):
    a, b, C = sp.symbols("a b C")
    A, B = wick.field_at(a), wick.field_at(b)
    wick.add_contraction(A, B, C)

    assert wick.average(A * B, expand=False) == C


def test_average_of_product_of_independent_pairs(wick):
    a, b, c, d, C1, C2 = sp.symbols("a b c d C1 C2")
    A, B, C, D = (wick.field_at(x) for x in (a, b, c, d))
    wick.add_contraction(A, B, C1)
    wick.add_contraction(C, D, C2)

    # The cross-pair contractions are not needed if the expression is
    # interpreted as a product whose only nonzero pairings are AB and CD.
    wick.add_contraction(A, C, 0)
    wick.add_contraction(A, D, 0)
    wick.add_contraction(B, C, 0)
    wick.add_contraction(B, D, 0)

    assert wick.average(A * B * C * D) == C1 * C2


def test_display_contractions_returns_self_and_prints(capsys, wick):
    a, b, C = sp.symbols("a b C")
    A, B = wick.field_at(a), wick.field_at(b)
    wick.add_contraction(A, B, C)

    returned = wick.display_contractions()
    output = capsys.readouterr().out

    assert returned is wick
    assert "Registered Gaussian contractions" in output
    assert str(C) in output
