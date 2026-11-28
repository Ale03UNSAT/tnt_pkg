"""
test_expressions.py
================

Pytest suite for expressions.py.
"""

import numpy as np
import pytest
import sympy as sp

from tntlib.symbolic import (
    SymbolicExpression,
    SymbolicMatrix,
    SymbolicSet,
    SeriesResult,
    LyapunovSolver,
)


# ============================================================
# 1. SYMBOLIC EXPRESSION
# ============================================================

class TestSymbolicExpression:

    @pytest.fixture
    def symbols(self):
        return sp.symbols("x y")

    def test_initialization_and_symbols(self, symbols):
        x, y = symbols
        expr = SymbolicExpression(x**2 + y)

        assert expr.expr == x**2 + y
        assert expr.symbols == {x, y}

    def test_arithmetic(self, symbols):
        x, y = symbols
        f = SymbolicExpression(x)

        assert sp.simplify((f + y).expr - (x + y)) == 0
        assert sp.simplify((f - y).expr - (x - y)) == 0
        assert sp.simplify((f * y).expr - x*y) == 0
        assert sp.simplify((f / y).expr - x/y) == 0
        assert sp.simplify((f**2).expr - x**2) == 0
        assert sp.simplify((-f).expr + x) == 0

    def test_reverse_arithmetic(self, symbols):
        x, _ = symbols
        f = SymbolicExpression(x)

        assert sp.simplify((2 + f).expr - (2 + x)) == 0
        assert sp.simplify((2 - f).expr - (2 - x)) == 0
        assert sp.simplify((2 * f).expr - 2*x) == 0
        assert sp.simplify((2 / f).expr - 2/x) == 0

    def test_substitution(self, symbols):
        x, y = symbols
        f = SymbolicExpression(x**2 + y)

        assert f.sub({x: 2, y: 3}).expr == 7

    def test_simultaneous_substitution(self):
        x, y = sp.symbols("x y")
        f = SymbolicExpression(x + y)

        result = f.sub(
            {x: y, y: 2},
            simultaneous=True
        )
        assert result.expr == y + 2

    def test_simplification_methods(self):
        x = sp.Symbol("x")
        f = SymbolicExpression((x**2 - 1)/(x - 1))

        assert sp.simplify(f.simplify().expr - (x + 1)) == 0
        assert sp.simplify(f.simplify("factor").expr - (x + 1)) == 0

        expanded = SymbolicExpression((x + 1)**2).simplify("expand")
        assert sp.expand(expanded.expr - (x**2 + 2*x + 1)) == 0

    def test_invalid_simplification_method(self):
        with pytest.raises(ValueError):
            SymbolicExpression(1).simplify("unknown")

    def test_copy_preserves_expression_and_name(self):
        f = SymbolicExpression(sp.Symbol("x")**2, name="test")
        copied = f.copy()

        assert copied.expr == f.expr
        assert copied is not f
        assert copied.name == "test"

    def test_latex(self):
        x = sp.Symbol("x")
        f = SymbolicExpression(x**2)

        assert f.tolatex() == sp.latex(x**2)

    def test_numpy_conversion_single_variable(self):
        x = sp.Symbol("x")
        f = SymbolicExpression(x**2 + 1)
        numerical = f.tonumpy()

        assert numerical(2) == pytest.approx(5)
        assert numerical(3) == pytest.approx(10)

    def test_numpy_conversion_multiple_variables(self):
        x, y = sp.symbols("x y")
        f = SymbolicExpression(x**2 + y)
        numerical = f.tonumpy()

        # Automatic variable ordering is alphabetical: x, y.
        assert numerical(2, 3) == pytest.approx(7)

    def test_series_at_zero(self):
        x = sp.Symbol("x")
        result = SymbolicExpression(sp.exp(x)).series(
            x, point=0, order=4
        )

        assert isinstance(result, SeriesResult)
        expected = 1 + x + x**2/2 + x**3/6
        assert sp.simplify(result.expanded.expr - expected) == 0

        assert result.coefficients[0].expr == 1
        assert result.coefficients[1].expr == 1
        assert result.coefficients[2].expr == sp.Rational(1, 2)
        assert result.coefficients[3].expr == sp.Rational(1, 6)

    def test_series_at_nonzero_point(self):
        x = sp.Symbol("x")
        result = SymbolicExpression(x**2).series(
            x, point=1, order=3
        )

        expected = 1 + 2*(x - 1) + (x - 1)**2
        assert sp.expand(result.expanded.expr - expected) == 0

        # Coefficients are in powers of (x - 1).
        assert result.coefficients[0].expr == 1
        assert result.coefficients[1].expr == 2
        assert result.coefficients[2].expr == 1

    def test_series_at_infinity(self):
        x = sp.Symbol("x")
        result = SymbolicExpression(1/(1 + x)).series(
            x, point=sp.oo, order=4
        )

        expected = 1/x - 1/x**2 + 1/x**3
        assert sp.simplify(result.expanded.expr - expected) == 0

        assert result.coefficients[-1].expr == 1
        assert result.coefficients[-2].expr == -1
        assert result.coefficients[-3].expr == 1

    def test_series_invalid_variable(self):
        with pytest.raises(TypeError):
            SymbolicExpression(1).series("x")

    def test_series_invalid_order(self):
        x = sp.Symbol("x")
        with pytest.raises(ValueError):
            SymbolicExpression(x).series(x, order=0)

        with pytest.raises(ValueError):
            SymbolicExpression(x).series(x, order=1.5)


# ============================================================
# 2. SYMBOLIC MATRIX
# ============================================================

class TestSymbolicMatrix:

    def test_initialization_and_shape(self):
        M = SymbolicMatrix([[1, 2], [3, 4]])
        assert M.shape == (2, 2)

    def test_transpose(self):
        M = SymbolicMatrix([[1, 2], [3, 4]])
        assert M.T.expr == M.expr.T

    def test_hermitian_conjugate(self):
        M = SymbolicMatrix([
            [1, sp.I],
            [2, 3]
        ])
        assert M.H.expr == M.expr.conjugate().T

    def test_determinant(self):
        M = SymbolicMatrix([[1, 2], [3, 4]])
        assert M.det.expr == -2

    def test_inverse(self):
        M = SymbolicMatrix([[1, 2], [3, 4]])
        assert M.expr * M.inv.expr == sp.eye(2)

    def test_trace(self):
        M = SymbolicMatrix([[1, 2], [3, 4]])
        assert M.trace.expr == 5

    def test_element(self):
        M = SymbolicMatrix([[1, 2], [3, 4]])
        assert M.element(0, 1).expr == 2

    def test_diagonal(self):
        M = SymbolicMatrix([[1, 2], [3, 4]])
        diagonal = M.diagonal()

        assert diagonal[0].expr == 1
        assert diagonal[1].expr == 4
        assert len(diagonal) == 2

    def test_numpy_conversion(self):
        x, y = sp.symbols("x y")
        M = SymbolicMatrix([[x, y], [y, x]])
        numerical = M.tonumpy()

        expected = np.array([[2, 3], [3, 2]], dtype=complex)
        np.testing.assert_allclose(numerical(2, 3), expected)

    def test_non_square_matrix(self):
        M = SymbolicMatrix([[1, 2, 3], [4, 5, 6]])
        assert M.shape == (2, 3)


# ============================================================
# 3. SYMBOLIC SET
# ============================================================

class TestSymbolicSet:

    @pytest.fixture
    def symbols(self):
        return sp.symbols("x y")

    @pytest.fixture
    def collection(self, symbols):
        x, y = symbols
        return SymbolicSet({
            "A": x**2 + y,
            "B": x*y,
            0: x + y
        }, name="Test collection")

    def test_initialization_and_length(self, collection):
        assert len(collection) == 3
        assert "A" in collection.keys()

    def test_indexing(self, collection, symbols):
        x, y = symbols
        assert collection["A"].expr == x**2 + y
        assert collection[0].expr == x + y

    def test_iteration(self, collection):
        assert list(collection) == ["A", "B", 0]

    def test_values_and_items(self, collection):
        assert len(list(collection.values())) == 3
        assert len(list(collection.items())) == 3

    def test_symbols(self, collection, symbols):
        x, y = symbols
        assert collection.symbols == {x, y}

    def test_substitution(self, collection, symbols):
        x, y = symbols
        result = collection.sub({x: 2, y: 3})

        assert result["A"].expr == 7
        assert result["B"].expr == 6

    def test_simplification(self):
        x = sp.Symbol("x")
        collection = SymbolicSet({
            "A": (x**2 - 1)/(x - 1)
        })
        result = collection.simplify()

        assert sp.simplify(result["A"].expr - (x + 1)) == 0

    def test_group_by(self):
        x, y = sp.symbols("x y")
        collection = SymbolicSet({
            "A": x**2 + 2*x*y + y**2
        })

        result = collection.group_by(x)
        expected = x**2 + 2*x*y + y**2

        assert sp.expand(result["A"].expr - expected) == 0
        assert result["A"].expr.has(x)

    def test_latex(self, collection):
        result = collection.tolatex()
        assert isinstance(result, dict)
        assert "A" in result

    def test_numpy_conversion(self):
        x, y = sp.symbols("x y")
        collection = SymbolicSet({
            "A": x**2 + y,
            "B": x*y
        })
        numerical = collection.tonumpy()

        assert numerical["A"](2, 3) == pytest.approx(7)
        assert numerical["B"](2, 3) == pytest.approx(6)

    def test_empty_collection(self):
        collection = SymbolicSet({})
        assert len(collection) == 0
        assert collection.symbols == set()


# ============================================================
# 4. LYAPUNOV SOLVER
# ============================================================

class TestLyapunovSolver:

    def test_general_solver_2x2(self):
        M = sp.diag(-1, -2)
        D = sp.eye(2)

        solver = LyapunovSolver(M, D, method="general")
        C = solver.solve(simplify=True)

        expected = sp.diag(
            sp.Rational(1, 2),
            sp.Rational(1, 4)
        )

        assert C.expr == expected
        assert solver.method_used == "general"
        assert solver.verify()

    def test_structured_solver_3x3(self):
        M = -sp.eye(3)
        D = sp.eye(3)

        solver = LyapunovSolver(M, D, method="structured")
        C = solver.solve(simplify=True)

        expected = sp.Rational(1, 2) * sp.eye(3)

        assert C.expr == expected
        assert solver.method_used == "structured"
        assert solver.verify()

    def test_auto_solver_uses_structured_method(self):
        M = -sp.eye(3)
        D = sp.eye(3)

        solver = LyapunovSolver(M, D, method="auto")
        C = solver.solve(simplify=True)

        assert C.expr == sp.Rational(1, 2) * sp.eye(3)
        assert solver.method_used == "structured"
        assert solver.verify()

    def test_auto_fallback_to_general(self):
        M = sp.diag(-1, -2)
        D = sp.eye(2)

        solver = LyapunovSolver(M, D, method="auto")
        C = solver.solve(simplify=True)

        expected = sp.diag(
            sp.Rational(1, 2),
            sp.Rational(1, 4)
        )

        assert C.expr == expected
        assert solver.method_used == "general"
        assert solver.verify()

    def test_residual(self):
        solver = LyapunovSolver(
            sp.diag(-1, -2),
            sp.eye(2),
            method="general"
        )
        solver.solve(simplify=True)

        assert solver.residual().expr == sp.zeros(2, 2)

    def test_verify_before_solve_raises(self):
        solver = LyapunovSolver(-sp.eye(2), sp.eye(2))

        with pytest.raises(RuntimeError):
            solver.verify()

    def test_residual_before_solve_raises(self):
        solver = LyapunovSolver(-sp.eye(2), sp.eye(2))

        with pytest.raises(RuntimeError):
            solver.residual()

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            LyapunovSolver(
                -sp.eye(2),
                sp.eye(2),
                method="invalid"
            )

    def test_non_square_matrix(self):
        with pytest.raises(ValueError):
            LyapunovSolver(
                sp.Matrix([[1, 2, 3], [4, 5, 6]]),
                sp.eye(2)
            )

    def test_incompatible_dimensions(self):
        with pytest.raises(ValueError):
            LyapunovSolver(sp.eye(2), sp.eye(3))


# ============================================================
# 5. DISPLAY METHODS
# ============================================================

class TestDisplay:

    def test_expression_display(self, capsys):
        f = SymbolicExpression(1 + sp.Symbol("x"), name="Test")
        assert f.display() is f

        output = capsys.readouterr().out
        assert "Test" in output

    def test_expression_latex_display(self, capsys):
        f = SymbolicExpression(sp.Symbol("x")**2)
        f.display(use_latex=True)

        output = capsys.readouterr().out
        assert "x^{2}" in output

    def test_set_display(self, capsys):
        collection = SymbolicSet(
            {"A": 1, "B": 2},
            name="Observables"
        )
        assert collection.display() is collection

        output = capsys.readouterr().out
        assert "Observables" in output
        assert "A:" in output

    def test_series_display(self, capsys):
        x = sp.Symbol("x")
        result = SymbolicExpression(sp.exp(x)).series(
            x, order=3
        )

        assert result.display() is result

        output = capsys.readouterr().out
        assert "SERIES EXPANSION" in output
        assert "COEFFICIENTS" in output
