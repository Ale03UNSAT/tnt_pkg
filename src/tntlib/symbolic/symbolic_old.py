"""
symbolic.py
===========

Symbolic computation tools for scientific physics.

Dependencies
------------
numpy
sympy
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from abc import ABC, abstractmethod


# ============================================================
# 1. BASE SYMBOLIC OBJECT
# ============================================================

class SymbolicObject(ABC):
    """
    Base class for symbolic objects.
    """

    def __init__(self, expression, name=None):
        self.expr = expression
        self.name = name

    @abstractmethod
    def _new(self, expression):
        """Construct a new object of the same type."""
        pass

    def copy(self):
        return self._new(self.expr)

    # --------------------------------------------------------
    # Substitution
    # --------------------------------------------------------

    def sub(self, substitutions, simultaneous=True):
        result = self.expr.subs(
            substitutions,
            simultaneous=simultaneous
        )
        return self._new(result)

    # --------------------------------------------------------
    # Simplification
    # --------------------------------------------------------

    def simplify(self, method="simplify"):
        methods = {
            "simplify": sp.simplify,
            "factor": sp.factor,
            "expand": sp.expand,
            "cancel": sp.cancel,
            "together": sp.together,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}")

        return self._new(methods[method](self.expr))

    def expand(self, **kwargs):
        return self._new(sp.expand(self.expr, **kwargs))

    def factor(self):
        return self._new(sp.factor(self.expr))

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def display(self, use_latex=False):
        """
        Pretty-print the expression in the terminal.

        Parameters
        ----------
        use_latex : bool
            Print LaTeX source instead of SymPy pretty output.
        """

        if self.name:
            print(f"\n{self.name}")
            print("=" * len(self.name))

        if use_latex:
            print(self.tolatex())
        else:
            print(sp.pretty(self.expr, use_unicode=True))

        return self

    # --------------------------------------------------------
    # LaTeX
    # --------------------------------------------------------

    def tolatex(self):
        return sp.latex(self.expr)

    # --------------------------------------------------------
    # Numerical conversion
    # --------------------------------------------------------

    def tonumpy(self, variables=None, modules="numpy"):
        """
        Convert expression into a numerical function.

        Automatically identifies free symbols if variables
        are not explicitly provided.
        """

        if variables is None:
            variables = sorted(
                self.expr.free_symbols,
                key=lambda x: x.name
            )

        variables = tuple(variables)

        return sp.lambdify(
            variables,
            self.expr,
            modules=modules
        )

    # --------------------------------------------------------
    # Symbol inspection
    # --------------------------------------------------------

    @property
    def symbols(self):
        return self.expr.free_symbols

    def __repr__(self):
        return f"{self.__class__.__name__}({self.expr})"


# ============================================================
# 2. SERIES RESULT
# ============================================================

class SeriesResult:
    """
    Stores a symbolic series expansion and its coefficients.

    Attributes
    ----------
    expanded : SymbolicExpression
        Expanded expression.

    coefficients : SymbolicSet
        Coefficients indexed by powers of the expansion variable.

    variable : sympy.Symbol
        Expansion variable.

    point : expansion point
    """

    def __init__(
        self,
        expanded,
        coefficients,
        variable,
        point,
        order
    ):

        self.expanded = SymbolicExpression(
            expanded,
            name="Series expansion"
        )

        self.coefficients = SymbolicSet(
            coefficients,
            name="Series coefficients"
        )

        self.variable = variable
        self.point = point
        self.order = order

    def display(self):
        print("\nSERIES EXPANSION")
        print("=" * 50)

        self.expanded.display()

        print("\nCOEFFICIENTS")
        print("=" * 50)

        self.coefficients.display()

        return self


# ============================================================
# 3. SYMBOLIC EXPRESSION
# ============================================================

class SymbolicExpression(SymbolicObject):
    """
    Scalar symbolic expression.
    """

    def __init__(self, expression, name=None):
        expression = sp.sympify(expression)

        if isinstance(expression, sp.MatrixBase):
            raise TypeError(
                "Use SymbolicMatrix for matrix expressions."
            )

        super().__init__(expression, name=name)

    def _new(self, expression):
        return SymbolicExpression(
            expression,
            name=self.name
        )

    # --------------------------------------------------------
    # Arithmetic
    # --------------------------------------------------------

    def __add__(self, other):
        return self._new(self.expr + _unwrap(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._new(self.expr - _unwrap(other))

    def __rsub__(self, other):
        return self._new(_unwrap(other) - self.expr)

    def __mul__(self, other):
        return self._new(self.expr * _unwrap(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._new(self.expr / _unwrap(other))

    def __pow__(self, power):
        return self._new(self.expr**power)

    def __neg__(self):
        return self._new(-self.expr)

    # --------------------------------------------------------
    # Series expansion
    # --------------------------------------------------------

    def series(
        self,
        variable,
        point=0,
        order=6,
        simplify=False
    ):
        """
        Expand around an arbitrary point.

        Examples
        --------
        expr.series(k, point=0, order=6)
        expr.series(k, point=sp.oo, order=5)
        expr.series(k, point=1, order=4)

        For infinity, SymPy returns an asymptotic expansion
        in inverse powers of the variable.

        Parameters
        ----------
        variable : sympy.Symbol
            Expansion variable.

        point : number or sympy expression
            Expansion point. Use sp.oo for infinity.

        order : int
            Series order (exclusive).

        simplify : bool
            Optionally simplify the final expansion.
        """

        if not isinstance(variable, sp.Symbol):
            raise TypeError(
                "variable must be a SymPy Symbol."
            )

        if not isinstance(order, int) or order < 1:
            raise ValueError(
                "order must be a positive integer."
            )

                t = sp.Dummy("t")

        if point == sp.oo:
            # Asymptotic expansion at infinity
            result = sp.series(
                self.expr,
                variable,
                sp.oo,
                order
            ).removeO().expand()

            expansion_variable = variable

        else:
            # Expand in t = variable - point
            local_expr = self.expr.subs(
                variable,
                point + t
            )

            local_series = sp.series(
                local_expr,
                t,
                0,
                order
            ).removeO().expand()

            # Return expression in the original variable
            result = local_series.subs(
                t,
                variable - point
            ).expand()

            expansion_variable = t

        if simplify:
            result = sp.simplify(result)

        coefficients = {}

        for term in sp.Add.make_args(
            local_series if point != sp.oo else result
        ):

            coefficient, power = term.as_coeff_exponent(
                expansion_variable
            )

            if not power.is_Integer:
                raise ValueError(
                    "Non-integer powers are not supported."
                )

            power = int(power)

            coefficients[power] = (
                coefficients.get(power, 0) + coefficient
            )

        coefficients = {
            n: c for n, c in coefficients.items()
            if c != 0
        }


# ============================================================
# 4. SYMBOLIC MATRIX
# ============================================================

class SymbolicMatrix(SymbolicObject):
    """
    General symbolic matrix.
    """

    def __init__(self, expression, name=None):
        super().__init__(
            sp.Matrix(expression),
            name=name
        )

    def _new(self, expression):
        return SymbolicMatrix(
            expression,
            name=self.name
        )

    @property
    def shape(self):
        return self.expr.shape

    @property
    def T(self):
        return self._new(self.expr.T)

    @property
    def H(self):
        return self._new(self.expr.conjugate().T)

    @property
    def det(self):
        return SymbolicExpression(
            self.expr.det(),
            name="Determinant"
        )

    @property
    def inv(self):
        return self._new(self.expr.inv())

    @property
    def trace(self):
        return SymbolicExpression(
            sp.trace(self.expr),
            name="Trace"
        )

    def element(self, i, j):
        return SymbolicExpression(
            self.expr[i, j]
        )

    def diagonal(self):
        return SymbolicSet({
            i: self.expr[i, i]
            for i in range(min(self.shape))
        })

    def tonumpy(self, variables=None, modules="numpy"):
        function = super().tonumpy(
            variables=variables,
            modules=modules
        )

        def numerical_function(*args):
            return np.asarray(
                function(*args),
                dtype=complex
            )

        return numerical_function


# ============================================================
# 5. SYMBOLIC COLLECTION
# ============================================================

class SymbolicSet:
    """
    Collection of named or indexed symbolic expressions.

    Keys can be strings or integers.

    Useful for:
        - Structure factors
        - Expansion coefficients
        - Physical observables
    """

    def __init__(self, expressions, name=None):

        self.expressions = {
            key: sp.sympify(_unwrap(value))
            for key, value in expressions.items()
        }

        self.name = name

    @property
    def expr(self):
        return self.expressions

    def __getitem__(self, key):
        return SymbolicExpression(
            self.expressions[key],
            name=str(key)
        )

    def __iter__(self):
        return iter(self.expressions)

    def __len__(self):
        return len(self.expressions)

    def keys(self):
        return self.expressions.keys()

    def values(self):
        return self.expressions.values()

    def items(self):
        return self.expressions.items()

    def _new(self, expressions):
        return self.__class__(
            expressions,
            name=self.name
        )

    def sub(self, substitutions, simultaneous=True):
        return self._new({
            key: value.subs(
                substitutions,
                simultaneous=simultaneous
            )
            for key, value in self.expressions.items()
        })

    def simplify(self, method="simplify"):

        methods = {
            "simplify": sp.simplify,
            "factor": sp.factor,
            "expand": sp.expand,
            "cancel": sp.cancel,
            "together": sp.together,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}")

        return self._new({
            key: methods[method](value)
            for key, value in self.expressions.items()
        })

    def group_by(self, variable, expand_first=True):
        """
        Collect every expression by powers of variable.

        The variable can be any symbolic parameter, not
        necessarily the original series expansion variable.
        """

        result = {}

        for key, value in self.expressions.items():

            if expand_first:
                value = sp.expand(value)

            result[key] = sp.collect(
                value,
                variable
            )

        return self._new(result)

    def display(self, use_latex=False):

        if self.name:
            print(f"\n{self.name}")
            print("=" * len(self.name))

        for key, value in self.expressions.items():

            print(f"\n{key}:")

            if use_latex:
                print(sp.latex(value))
            else:
                print(sp.pretty(value))

        return self

    def tolatex(self):
        return {
            key: sp.latex(value)
            for key, value in self.expressions.items()
        }

    def tonumpy(self, variables=None, modules="numpy"):
        return {
            key: SymbolicExpression(value).tonumpy(
                variables=variables,
                modules=modules
            )
            for key, value in self.expressions.items()
        }

    @property
    def symbols(self):
        return set().union(*(
            value.free_symbols
            for value in self.expressions.values()
        )) if self.expressions else set()


# ============================================================
# 6. INTERNAL HELPERS
# ============================================================

def _unwrap(value):

    if isinstance(value, SymbolicObject):
        return value.expr

    return value


def _vec_col(matrix):
    """Column-major vectorization."""
    return sp.Matrix([
        matrix[i, j]
        for j in range(matrix.cols)
        for i in range(matrix.rows)
    ])


def _unvec_col(vector, n):
    """Inverse column-major vectorization."""
    return sp.Matrix(
        n, n,
        lambda i, j: vector[j*n + i]
    )


def _linear_solve(equations, unknowns):
    """
    Solve a linear symbolic system.

    Avoids sp.solve(), which is a general-purpose equation
    solver and can be much slower for this problem.
    """

    A, b = sp.linear_eq_to_matrix(
        equations,
        unknowns
    )

    if A.rows != A.cols:
        raise ValueError(
            "The symbolic system is not square."
        )

    return A.LUsolve(b)


# ============================================================
# 7. LYAPUNOV SOLVER
# ============================================================

class LyapunovSolver:
    """
    Efficient symbolic continuous Lyapunov solver.

    Solves:

        M C + C M.H + D = 0

    Parameters
    ----------
    M : sympy.Matrix
        Dynamical matrix.

    D : sympy.Matrix
        Full noise covariance matrix.

    method : str
        "auto", "structured", or "general".

    The auto method attempts the six-real-variable
    3x3 covariance structure first, then falls back
    to the general Kronecker solver.
    """

    def __init__(self, M, D, method="auto"):

        self.M = sp.Matrix(_unwrap(M))
        self.D = sp.Matrix(_unwrap(D))

        if self.M.rows != self.M.cols:
            raise ValueError("M must be square.")

        if self.M.shape != self.D.shape:
            raise ValueError(
                "M and D must have identical dimensions."
            )

        if method not in ("auto", "structured", "general"):
            raise ValueError(
                "method must be 'auto', 'structured' or 'general'."
            )

        self.method = method
        self.method_used = None
        self.C = None
        self.solution = None

    # --------------------------------------------------------
    # Six-variable structured solver
    # --------------------------------------------------------

    def _solve_structured(self):
        """
        Solve the six-variable 3x3 Hermitian ansatz:

            C = [[A, iB, iC],
                 [-iB, D, H],
                 [-iC, H, F]]

        The ansatz must be physically valid for the model.
        The complete residual is checked afterwards.
        """

        if self.M.shape != (3, 3):
            raise ValueError(
                "The structured solver requires a 3x3 matrix."
            )

        A, B, Ccross, Dvar, H, F = sp.symbols(
            "_A _B _Ccross _D _H _F",
            real=True
        )

        unknowns = [A, B, Ccross, Dvar, H, F]

        Cmat = sp.Matrix([
            [A,          sp.I*B,       sp.I*Ccross],
            [-sp.I*B,    Dvar,         H],
            [-sp.I*Ccross, H,          F],
        ])

        E = (
            self.M*Cmat
            + Cmat*self.M.conjugate().T
            + self.D
        )

        # Six independent real equations.
        # This is the same physical reduction used
        # in the original notebook.

        equations = [
            sp.re(E[0, 0]),
            sp.re(E[0, 1] / sp.I),
            sp.re(E[0, 2] / sp.I),
            sp.re(E[1, 1]),
            sp.re(E[1, 2]),
            sp.re(E[2, 2]),
        ]

        #solution_vector = _linear_solve(
        #    equations,
        #    unknowns
        #)

        #solution = dict(zip(
        #    unknowns,
        #    solution_vector
        #))
        solutions = sp.solve(
            equations,
            unknowns,
            dict=True,
            simplify=False
        )
        
        if len(solutions) != 1:
            raise ValueError(
                "The six-variable ansatz did not produce "
                "one unique solution."
            )
        
        solution = solutions[0]
        
        C_solution = Cmat.subs(
            solution,
            simultaneous=True
        )

        # Validate against every entry of the original
        # Lyapunov equation.

        residual = (
            self.M*C_solution
            + C_solution*self.M.conjugate().T
            + self.D
        )

        residual = residual.applyfunc(sp.simplify)

        if not all(entry == 0 for entry in residual):
            raise ValueError(
                "The six-variable ansatz is not valid "
                "for this Lyapunov problem."
            )

        return C_solution, solution

    # --------------------------------------------------------
    # General Kronecker solver
    # --------------------------------------------------------

    def _solve_general(self):
        """
        General solver for arbitrary square matrices.

        Uses column-major vectorization:

        vec(M C + C M.H) =
        [I kron M + conjugate(M) kron I] vec(C)
        """

        n = self.M.rows
        identity = sp.eye(n)

        K = (
            sp.kronecker_product(identity, self.M)
            + sp.kronecker_product(
                self.M.conjugate(),
                identity
            )
        )

        rhs = -_vec_col(self.D)

        C_vector = K.LUsolve(rhs)

        return _unvec_col(C_vector, n)

    # --------------------------------------------------------
    # Main solve method
    # --------------------------------------------------------

    def solve(self, simplify=False):
        """
        Calculate the stationary covariance matrix.

        Returns
        -------
        SymbolicMatrix
            The covariance matrix.
        """

        self.solution = None

        if self.method == "structured":
            C, solution = self._solve_structured()

            self.solution = solution
            self.method_used = "structured"

        elif self.method == "general":
            C = self._solve_general()
            self.method_used = "general"

        else:
            # Try the optimized physical structure first.
            try:
                C, solution = self._solve_structured()

                self.solution = solution
                self.method_used = "structured"

            except (
                ValueError,
                sp.matrices.exceptions.NonInvertibleMatrixError
            ):
                # Fall back to the general solver.
                C = self._solve_general()
                self.method_used = "general"

        if simplify:
            C = C.applyfunc(sp.simplify)

        self.C = SymbolicMatrix(
            C,
            name="Correlation matrix"
        )

        return self.C

    # --------------------------------------------------------
    # Residual and verification
    # --------------------------------------------------------

    def residual(self):

        if self.C is None:
            raise RuntimeError(
                "Call solve() before residual()."
            )

        return SymbolicMatrix(
            self.M*self.C.expr
            + self.C.expr*self.M.conjugate().T
            + self.D,
            name="Lyapunov residual"
        )

    def verify(self):

        residual = self.residual().expr.applyfunc(
            sp.simplify
        )

        return all(
            entry == 0
            for entry in residual
        )

    def display(self):

        print(f"Lyapunov solver: {self.method_used}")

        if self.C is not None:
            self.C.display()

        return self
