"""
wick.py
=======

Symbolic propagators, physical symmetry rules,
and Gaussian Wick contractions.

Dependencies
------------
sympy
"""

import sympy as sp

from functools import lru_cache


# ============================================================
# 1. SYMBOLIC PROPAGATOR
# ============================================================

class SymbolicPropagator:
    """
    Symbolic scalar or matrix propagator.

    The user supplies the analytical expression.

    Symmetries are stored explicitly and can be:
        - verified against the supplied expression;
        - applied to abstract propagator components.

    Matrix propagator:
        G(i, j, k, t)

    Scalar propagator:
        G(k, t)
    """

    def __init__(
        self,
        expression,
        momentum=None,
        time=None,
        name="G"
    ):

        self.expr = (
            sp.Matrix(expression)
            if isinstance(expression, (sp.MatrixBase, list, tuple))
            else sp.sympify(expression)
        )

        self.momentum = momentum
        self.time = time
        self.name = name

        self.is_matrix = isinstance(
            self.expr, sp.MatrixBase
        )

        if self.is_matrix:
            if self.expr.rows != self.expr.cols:
                raise ValueError(
                    "A matrix propagator must be square."
                )

        # Abstract symbolic representation
        self.G = sp.Function(name)

        # Registered symmetry rules
        self.symmetries = []

    # --------------------------------------------------------
    # Access propagator
    # --------------------------------------------------------

    def element(self, i=None, j=None):
        """
        Return a propagator component.

        For matrix propagators:
            element(i, j)

        For scalar propagators:
            element()
        """

        if self.is_matrix:
            if i is None or j is None:
                raise ValueError(
                    "Matrix propagators require i and j."
                )

            return self.expr[i, j]

        return self.expr

    def symbolic_element(self, *args):
        """
        Return an abstract propagator component.

        Matrix:
            G(i, j, k, t)

        Scalar:
            G(k, t)
        """

        return self.G(*args)

    # --------------------------------------------------------
    # Register symmetries
    # --------------------------------------------------------

    def add_momentum_parity(self, sign=1):
        """
        G(k,t) = sign * G(-k,t)
        """

        self._add_argument_symmetry(
            variable=self.momentum,
            sign=sign,
            kind="momentum"
        )

    def add_time_parity(self, sign=1):
        """
        G(k,t) = sign * G(k,-t)
        """

        self._add_argument_symmetry(
            variable=self.time,
            sign=sign,
            kind="time"
        )

    def _add_argument_symmetry(
        self,
        variable,
        sign,
        kind
    ):

        if variable is None:
            raise ValueError(
                f"{kind} variable was not specified."
            )

        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1.")

        self.symmetries.append({
            "kind": kind,
            "variable": variable,
            "sign": sign
        })

    def add_index_symmetry(self, i, j, sign=1):
        """
        Register:

            G_ij = sign * G_ji

        Only applies to matrix propagators.
        """

        if not self.is_matrix:
            raise ValueError(
                "Index symmetry requires a matrix propagator."
            )

        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1.")

        self.symmetries.append({
            "kind": "index",
            "i": i,
            "j": j,
            "sign": sign
        })

    # --------------------------------------------------------
    # Verify symmetries
    # --------------------------------------------------------

    def verify_symmetries(self):
        """
        Verify registered symmetries against the
        explicitly supplied propagator.

        Returns a dictionary containing the results.
        """

        results = {}

        for n, rule in enumerate(self.symmetries):

            kind = rule["kind"]
            sign = rule["sign"]

            if kind in ("momentum", "time"):

                variable = rule["variable"]

                if self.is_matrix:
                    checks = []

                    for i in range(self.expr.rows):
                        for j in range(self.expr.cols):

                            original = self.expr[i, j]

                            transformed = original.subs(
                                variable,
                                -variable
                            )

                            checks.append(
                                sp.simplify(
                                    transformed - sign*original
                                ) == 0
                            )

                    valid = all(checks)

                else:

                    transformed = self.expr.subs(
                        variable,
                        -variable
                    )

                    valid = (
                        sp.simplify(
                            transformed - sign*self.expr
                        ) == 0
                    )

            elif kind == "index":

                i = rule["i"]
                j = rule["j"]

                valid = (
                    sp.simplify(
                        self.expr[i, j]
                        - sign*self.expr[j, i]
                    ) == 0
                )

            results[n] = {
                "rule": rule,
                "valid": valid
            }

        return results

    # --------------------------------------------------------
    # Apply symmetries to abstract propagators
    # --------------------------------------------------------

    
    def apply_symmetries(self, expression):
        """
        Apply registered rules to abstract G(...) objects.
        
        This operates on symbolic propagator components,
        not directly on arbitrary explicit expressions.
        """

        expression = sp.sympify(expression)
        replacements = {}

        for atom in expression.atoms(sp.Function):
            
            if atom.func != self.G:
                continue
        
            args = list(atom.args)
            factor = 1

            for rule in self.symmetries:
                
                kind = rule["kind"]
                sign = rule["sign"]

                if kind == "index" and self.is_matrix:

                    i = rule["i"]
                    j = rule["j"]

                    if len(args) >= 2 and args[0] == i and args[1] == j:
                        args[0], args[1] = j, i
                        factor *= sign

                elif kind in ("momentum", "time"):
                        
                    variable = rule["variable"]
                    
                    for pos, arg in enumerate(args):

                        # Match both k and -k, or t and -t.
                        if arg == -variable:
                            args[pos] = variable
                            factor *= sign
                            break
                            
            replacements[atom] = factor * self.G(*args)
                            
        return expression.xreplace(replacements)


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def display(self):

        print(f"\nPropagator: {self.name}")
        print("=" * 50)

        if self.is_matrix:
            sp.pprint(self.expr)
        else:
            sp.pprint(self.expr)

        print("\nRegistered symmetries:")

        for rule in self.symmetries:
            print(rule)

        return self


# ============================================================
# 2. GAUSSIAN WICK ENGINE
# ============================================================

class WickEngine:
    """
    Symbolic Gaussian Wick engine.

    Represents Gaussian fields through:

        Psi(i, k, t)

    Computes arbitrary Gaussian moments using
    recursive Wick pairings.

    The two-point covariance must be registered
    by the user.
    """

    def __init__(self, field_name="Psi"):

        self.field = sp.Function(field_name)

        self.contractions = {}

        # Cache Wick pairings
        self._cache = {}

    # --------------------------------------------------------
    # Create fields
    # --------------------------------------------------------

    def field_at(self, *args):
        """
        Construct a symbolic Gaussian field.

        Example:
            psi = wick.field_at(i, k, t)
        """

        return self.field(*args)

    # --------------------------------------------------------
    # Register two-point covariance
    # --------------------------------------------------------

    def _canonical_pair(self, field1, field2):
        """
        Gaussian classical fields commute.

        Therefore:
            <a b> = <b a>

        We use a canonical ordering for the cache.
        """

        return tuple(sorted(
            (field1, field2),
            key=sp.default_sort_key
        ))

    def add_contraction(
        self,
        field1,
        field2,
        covariance
    ):
        """
        Register:

            <field1 * field2> = covariance
        """

        self.contractions[
            self._canonical_pair(field1, field2)
        ] = sp.sympify(covariance)

        self._cache.clear()

    def contraction(self, field1, field2):
        """Retrieve a registered two-point covariance."""

        key = self._canonical_pair(field1, field2)

        if key not in self.contractions:
            raise KeyError(
                f"No contraction registered for:\n"
                f"{field1}, {field2}"
            )

        return self.contractions[key]

    # --------------------------------------------------------
    # Extract Gaussian fields from a product
    # --------------------------------------------------------

    def _extract_fields(self, term):
        """
        Extract field factors and their multiplicities.

        Example:
            a**2 * b

        returns:
            [a, a, b]
        """

        factors = []

        powers = term.as_powers_dict()

        for atom, power in powers.items():

            if atom.func != self.field:
                continue

            if not power.is_Integer or power < 0:
                raise ValueError(
                    "Only nonnegative integer powers "
                    "of Gaussian fields are supported."
                )

            factors.extend([atom] * int(power))

        return factors

    # --------------------------------------------------------
    # Recursive Wick pairings
    # --------------------------------------------------------

    def _wick_product(self, fields):
        """
        Compute the Gaussian average of a product.

        Recursive pairing algorithm.

        For an odd number of fields, the result is zero.
        """

        fields = tuple(fields)

        if len(fields) == 0:
            return sp.Integer(1)

        if len(fields) % 2:
            return sp.Integer(0)

        if fields in self._cache:
            return self._cache[fields]

        first = fields[0]

        result = 0

        for j in range(1, len(fields)):

            second = fields[j]

            remaining = (
                fields[1:j] + fields[j+1:]
            )

            pair = self.contraction(first, second)

            result += (
                pair * self._wick_product(remaining)
            )

        result = sp.expand(result)

        self._cache[fields] = result

        return result

    # --------------------------------------------------------
    # Gaussian average
    # --------------------------------------------------------

    def average(self, expression, expand=True):
        """
        Calculate the Gaussian expectation value.

        Handles sums, products and arbitrary even moments.

        Example:
            wick.average(a*b*c*d)
        """

        expression = sp.sympify(expression)

        if expand:
            expression = sp.expand(expression)

        result = 0

        for term in sp.Add.make_args(expression):

            fields = self._extract_fields(term)

            if not fields:
                result += term
                continue

            # Remove field factors to obtain the
            # non-Gaussian prefactor.
            prefactor = term

            for field in fields:
                prefactor = prefactor / field

            result += (
                prefactor *
                self._wick_product(fields)
            )

        return sp.expand(result)

    # --------------------------------------------------------
    # Explicit pairing generation
    # --------------------------------------------------------

    def pairings(self, fields):
        """
        Return all possible Wick pairings.

        Each pairing is represented as a list of pairs.
        """

        fields = tuple(fields)

        if len(fields) % 2:
            return []

        if len(fields) == 0:
            return [[]]

        first = fields[0]

        results = []

        for j in range(1, len(fields)):

            second = fields[j]

            remaining = (
                fields[1:j] + fields[j+1:]
            )

            for pairing in self.pairings(remaining):

                results.append(
                    [(first, second)] + pairing
                )

        return results

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    def display_contractions(self):

        print("\nRegistered Gaussian contractions")
        print("=" * 50)

        for (field1, field2), covariance in self.contractions.items():

            print(f"\n<{field1} {field2}> =")
            sp.pprint(covariance)

        return self
