"""
symbolic
========

Symbolic computation tools for scientific physics.

Modules
-------
expressions
    Symbolic expressions, matrices, sets and Lyapunov solvers.

wick
    Propagators and Wick's theorem for Gaussian averages.
"""

# ============================================================
# Symbolic expressions
# ============================================================

from .expressions import (
    SymbolicObject,
    SymbolicExpression,
    SymbolicMatrix,
    SymbolicSet,
    SeriesResult,
    LyapunovSolver,
)


# ============================================================
# Wick's theorem and propagators
# ============================================================

from .wick import (
    SymbolicPropagator,
    WickEngine,
)


# ============================================================
# Public API
# ============================================================

__all__ = [
    # Expressions
    "SymbolicObject",
    "SymbolicExpression",
    "SymbolicMatrix",
    "SymbolicSet",
    "SeriesResult",
    "LyapunovSolver",

    # Wick
    "SymbolicPropagator",
    "WickEngine",
]
