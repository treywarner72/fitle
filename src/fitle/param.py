"""param.py
=========
Symbolic parameters used throughout the library
----------------------------------------------------------

Three mutually–exclusive roles exist:

THETA
    A free model parameter to be optimised.  Optionally carries
    ``min`` / ``max`` bounds and a starting value ``start``.
INPUT
    Represents the input, when a parameter is plugged into a model. 
    Often repaced with data for cost functions.
INDEX
    Integer loop control used by ``Reduction`` objects.  Holds a
    Python :class:`range` instead of numeric bounds.

Factory helpers create the most common bounded flavours::

    mu     = Param()              # unbounded, start 0
    sigma  = Param.positive()     # >0, start 1
    weight = Param.unit()         # 0–1, start 0.5

Further tweaks are chainable in any order, e.g.::

    P('sigma')(2.0)(0.1, 10)    # name, start, bounds in one line

So, using __call__, a string sets the name, a scalar the starting value, and two scalars for the bounds

"""
from __future__ import annotations

import numpy as np
from enum import Enum, auto
from typing import Any, Optional, Tuple

__all__ = [
    "ParamKind",
    "Param",
    "ParamFactory",
    "INPUT",
    "index",
    "INDEX"
]


class ParamKind(Enum):
    """Enumerates the recognised parameter roles."""

    THETA = auto()   #: free, optimisable parameter
    INPUT = auto()   #: singleton placeholder for X
    INDEX = auto()   #: integer loop variable (carries :class:`range`)


class Param:
    """Different kinds of parameters used in :class:`Model`s.

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
        arg0: float | str | None = None, 
        arg1: float | None = None,
        min: float | None = None,
        start: float | None = None,
        max: float | None = None,
        name: str | None = None,
        *,
        range: range | None = None,
        kind: ParamKind = ParamKind.THETA,
    ):
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
        self.min = min if min is not None else -np.inf
        self.max = max if max is not None else np.inf
        self.start = None
        self.value = None

        if arg0 is not None:
            if arg1 is not None:
                self(arg0,arg1)
            else:
                self(arg0)

        if self.start is None:
            self.start = start if start is not None else 0
            
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
            if self.range is None:
                return "INDEX"
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
            if self.range is None:
                return "INDEX"
            return f"{self._prefix()}{self.range}"
        if hasattr(self, "error"):
            return f"{self._prefix()}{self.value:.4g} ± {self.error:.2g}"
        return f"{self._prefix()}{self.value:.4g}"

    # ------------------------------------------------------------------
    #   Mutating helpers
    # ------------------------------------------------------------------
    def __call__(self, arg0: str | np.number, arg1: np.number | None = None):
        if arg1 is not None:
            self.min, self.max = arg0, arg1
            # Compute natural default for these bounds
            if self.min == -np.inf and self.max == np.inf:
                natural = 0.0
            elif self.min == -np.inf:
                natural = float(self.max)
            elif self.max == np.inf:
                natural = float(self.min)
            else:
                natural = 0.5 * (self.min + self.max)
            # Reset if: outside bounds, or start is at old default (0) and new default differs
            outside = self.start is None or self.start < self.min or self.start > self.max
            at_old_default = (self.start == 0 and natural != 0)
            if outside or at_old_default:
                self.start = natural
                self.value = self.start
            return self
        # Rename (str) or set start value (numeric).
        if isinstance(arg0, str):
            self.name = arg0
        elif isinstance(arg0, (int, float, np.number)):
            if self.min is not None and arg0 < self.min:
                raise ValueError("start below min bound")
            if self.max is not None and arg0 > self.max:
                raise ValueError("start above max bound")
            self.start = float(arg0)
            self.value = float(arg0)
        else:
            raise TypeError("Param.__call__ expects str or numeric")
        return self

# -----------------------------------------------------------------------------
#   Factory helpers
# -----------------------------------------------------------------------------

class ParamFactory:
    """Callable/sliceable object that returns pre‑configured ``Param``s."""

    def __init__(self, default_limits: Tuple[float, float], default_start: float):
        self._limits = default_limits
        self._start = default_start

    def __call__(self, arg0: str | np.number | None = None, arg1: np.number | None = None):
        p = Param()(*self._limits)(self._start)
        if arg0 is not None:
            if arg1 is not None:
                return p(arg0, arg1)
            return p(arg0)
        return p


Param.positive = ParamFactory((1e-6, np.inf), 1)         # positive
Param.unit = ParamFactory((0, 1), 0.5)               # 0–1 interval


# -----------------------------------------------------------------------------
#   Module‑level helpers
# -----------------------------------------------------------------------------

INPUT: Param = Param(kind=Param.input)          # singleton INPUT placeholder

def index(*args: int) -> Param:
    """Shortcut for creating an INDEX parameter with ``range(*args)``."""
    return Param(range=range(*args) if args else None, kind=Param.index)

INDEX: Param = index() # default INDEX parameter.
