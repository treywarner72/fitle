"""param.py
=========
Symbolic parameter primitives used throughout the library
----------------------------------------------------------

Three mutually–exclusive roles exist:

THETA
    A free model parameter to be optimised.  Optionally carries
    ``min`` / ``max`` bounds and a starting value ``start``.
INPUT
    Placeholder for the data abscissa X.  There is exactly one;
    it never has bounds.
INDEX
    Integer loop control used by ``Reduction`` objects.  Holds a
    Python :class:`range` instead of numeric bounds.

Factory helpers create the most common bounded flavours::

    mu    = P()         # unbounded, start 0
    sigma = Q()         # >0, start 1
    w     = U()         # 0–1, start 0.5

Further tweaks are chainable in any order, e.g.::

    P('sigma')(2.0)[0.1, 10]    # name, start, bounds in one line
"""
from __future__ import annotations

import numpy as np
from enum import Enum, auto
from typing import Any, Optional, Tuple

__all__ = [
    "ParamKind",
    "Param",
    "ParamFactory",
    "P",
    "Q",
    "U",
    "X",
    "index",
]


class ParamKind(Enum):
    """Enumerates the recognised parameter roles."""

    THETA = auto()   #: free, optimisable parameter
    INPUT = auto()   #: singleton placeholder for X
    INDEX = auto()   #: integer loop variable (carries :class:`range`)


class Param:
    """Light‑weight symbolic leaf node used in :class:`Model` graphs.

    Parameters
    ----------
    min, max : float | None, optional
        Numerical bounds (THETA only).
    start : float | None, optional
        Initial value; defaults to 0 or the midpoint of bounds.
    name : str | None, optional
        Cosmetic identifier used in repr / str.
    range : range | None, keyword‑only
        Loop range (INDEX only).
    kind : ParamKind, default ``THETA``
        Role selector.
    """

    # convenience aliases
    theta = ParamKind.THETA
    input = ParamKind.INPUT
    index = ParamKind.INDEX

    def __init__(
        self,
        min: float | None = None,
        start: float | None = None,
        max: float | None = None,
        name: str | None = None,
        *,
        range: range | None = None,
        kind: ParamKind = ParamKind.THETA,
    ) -> None:
        self.kind = kind
        self.name = name

        # INPUT – no metadata allowed
        if kind is ParamKind.INPUT:
            if any(x is not None for x in (min, start, max, name, range)):
                raise ValueError("INPUT parameter must not carry bounds or name")
            return

        # INDEX – only a range
        if kind is ParamKind.INDEX:
            if any(x is not None for x in (min, start, max)):
                raise ValueError("INDEX parameter cannot have numeric bounds")
            self.range = range  # type: ignore[assignment]
            return

        # THETA – store optimisation metadata
        self.min = min
        self.start = start
        self.max = max
        self.value: float | None = self.start

    # ------------------------------------------------------------------
    #   Comparison & hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:  # noqa: D401
        if not isinstance(other, Param):
            return False
        if self.kind is self.input and other.kind is other.input:
            return True  # all INPUT placeholders are equivalent
        return self is other

    def __hash__(self) -> int:  # noqa: D401 – identity hash
        return id(self)

    # ------------------------------------------------------------------
    #   Printable helpers
    # ------------------------------------------------------------------
    def _prefix(self) -> str:
        return f"{self.name}=" if self.name else ""

    def __repr__(self) -> str:
        if self.kind is self.input:
            return "INPUT"
        if self.kind is self.index:
            return f"{self._prefix()}{self.range}"
        if self.min == self.max:
            return f"{self._prefix()}Param(value={self.value})"
        return (
            f"{self._prefix()}Param(min={self.min}, start={self.start}, "
            f"max={self.max}, value={self.value})"
        )

    def __str__(self) -> str:  # noqa: D401
        if self.kind is self.input:
            return "INPUT"
        if self.kind is self.index:
            return f"{self._prefix()}{self.range}"
        if hasattr(self, "error"):
            return f"{self._prefix()}{self.value:.4g} ± {self.error:.2g}"
        return f"{self._prefix()}{self.value:.4g}"

    # ------------------------------------------------------------------
    #   Mutating helpers
    # ------------------------------------------------------------------
    def __call__(self, arg: str | int | float | np.number):
        """Rename (str) or set start value (numeric)."""
        if isinstance(arg, str):
            self.name = arg
        elif isinstance(arg, (int, float, np.number)):
            if self.min is not None and arg < self.min:
                raise ValueError("start below min bound")
            if self.max is not None and arg > self.max:
                raise ValueError("start above max bound")
            self.start = float(arg)
            self.value = float(arg)
        else:
            raise TypeError("Param.__call__ expects str or numeric")
        return self

    def __getitem__(self, bounds: Tuple[float, float]):
        """Assign numeric bounds via slice syntax and adjust ``start``.

        A new default ``start`` is chosen if the existing one lies
        outside the interval.
        """
        self.min, self.max = bounds
        reset = self.start is None or self.start < self.min or self.start > self.max
        if reset:
            if self.min == -np.inf and self.max == np.inf:
                self.start = 0.0
            elif self.min == -np.inf:
                self.start = float(self.max)
            elif self.max == np.inf:
                self.start = float(self.min)
            else:
                self.start = 0.5 * (self.min + self.max)
            self.value = self.start
        return self


# -----------------------------------------------------------------------------
#   Factory helpers
# -----------------------------------------------------------------------------

class ParamFactory:
    """Callable/sliceable object that returns pre‑configured ``Param``s."""

    def __init__(self, default_limits: Tuple[float, float], default_start: float):
        self._limits = default_limits
        self._start = default_start

    def __getitem__(self, bounds: Tuple[float, float]):
        """Return a new ``Param`` with bounds applied."""
        return Param(self._start).__getitem__(bounds)

    def __call__(self, arg: str | int | float | np.number | None = None):
        p = Param().__getitem__(self._limits)(self._start)
        return p(arg) if arg is not None else p


P = ParamFactory((-np.inf, np.inf), 0)      # unbounded
Q = ParamFactory((1e-6, np.inf), 1)         # positive
U = ParamFactory((0, 1), 0.5)               # 0–1 interval


# -----------------------------------------------------------------------------
#   Module‑level helpers
# -----------------------------------------------------------------------------

X: Param = Param(kind=Param.input)          # singleton INPUT placeholder

def index(*args: int) -> Param:
    """Shortcut for creating an INDEX parameter with ``range(*args)``."""
    return Param(range=range(*args) if args else None, kind=Param.index)

I: Param = index() # default INDEX parameter.