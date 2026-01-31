"""param.py
=========
Symbolic parameters for model building.

The main interface is the `Param` builder which creates `_Param` instances:

    # Shorthand with unary operators (auto-named from variable)
    a = ~Param                # unbounded, auto-named 'a'
    sigma = +Param            # positive (>0), auto-named 'sigma'
    neg = -Param              # negative (<0), auto-named 'neg'

    # Explicit creation
    a = Param(5)              # unbounded, start=5, auto-named 'a'
    b = Param('b')            # unbounded, explicit name 'b'
    c = Param(0, 10)          # bounded [0,10], auto-named 'c'

    # Chaining works
    f = Param('f')(5)(0, 10)  # name='f', start=5, bounds=[0,10]

    # Constrained params (explicit)
    sigma = Param.positive('sigma')
    frac = Param.unit('frac')

    # Tuple unpacking with auto-naming
    mu, sigma = ~Param, +Param
"""
from __future__ import annotations

import numpy as np
from enum import Enum, auto
from typing import Tuple

__all__ = [
    "ParamKind",
    "Param",
    "_Param",
    "INPUT",
    "index",
    "INDEX",
]


class ParamKind(Enum):
    """Enumerates the recognised parameter roles."""
    THETA = auto()   #: free, optimisable parameter
    INPUT = auto()   #: singleton placeholder for X
    INDEX = auto()   #: integer loop variable (carries :class:`range`)


# -----------------------------------------------------------------------------
#   Variable name detection for auto-naming
# -----------------------------------------------------------------------------

_varname_cache: dict[tuple, int] = {}  # (filename, lineno) -> call index


def _split_top_level_commas(s: str) -> list[str]:
    """Split string by commas, but only at top level (not inside parens/brackets)."""
    parts = []
    current = []
    depth = 0
    for c in s:
        if c in '([':
            depth += 1
            current.append(c)
        elif c in ')]':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)
    if current:
        parts.append(''.join(current))
    return parts


def _is_simple_param_expr(rhs: str) -> bool:
    """Check if RHS contains only simple Param expressions.

    Matches: ~Param, +Param, -Param, Param(5), Param.positive('x'), (+Param)(5), etc.
    Does NOT match: Param(5) + Param(3), gaussian(Param()), etc.
    """
    import re
    # Pattern options:
    # 1. Optional unary op, identifier with optional .attr, followed by optional () chains
    # 2. Parenthesized unary expression like (+Param) followed by () chains
    pattern = r'^\s*([~+\-]?\s*\w+(\.\w+)?|\([~+\-]\s*\w+(\.\w+)?\))(\(.*?\))*\s*$'
    parts = _split_top_level_commas(rhs)
    return all(re.match(pattern, part.strip()) for part in parts)


def _detect_varname(depth: int = 2) -> str | None:
    """Detect variable name from assignment in calling frame.

    Only returns a name for simple direct assignments like:
        a = ~Param
        a, b = ~Param, +Param
        x = Param(5)

    Returns None for complex expressions like:
        y = ~Param + ~Param
        m = gaussian(~Param)
    """
    import inspect
    import re

    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is None:
                return None
            frame = frame.f_back
        if frame is None:
            return None

        info = inspect.getframeinfo(frame)
        if not info.code_context:
            return None

        line = info.code_context[0]
        key = (info.filename, info.lineno)

        # Strip comments (simple approach: remove # and everything after if not in string)
        # This handles common cases like: a = ~Param  # comment
        comment_pos = line.find('#')
        if comment_pos != -1:
            # Check it's not inside a string by counting quotes before it
            before = line[:comment_pos]
            if before.count("'") % 2 == 0 and before.count('"') % 2 == 0:
                line = before

        # Parse LHS and RHS of assignment
        match = re.match(r'\s*([\w\s,]+)\s*=\s*(.+)$', line)
        if not match:
            return None

        lhs = match.group(1)
        rhs = match.group(2)

        # Only auto-name for simple Param expressions
        if not _is_simple_param_expr(rhs):
            return None

        names = [n.strip() for n in lhs.split(',')]
        rhs_parts = _split_top_level_commas(rhs)

        # Must have same number of LHS vars and RHS parts
        if len(names) != len(rhs_parts):
            return None

        if len(names) == 1:
            return names[0]

        # Tuple unpacking: track position
        call_index = _varname_cache.get(key, 0)

        # Reset if we've gone past the end (re-execution of same line)
        if call_index >= len(names):
            call_index = 0

        _varname_cache[key] = call_index + 1

        return names[call_index]

    except Exception:
        return None


# -----------------------------------------------------------------------------
#   _Param: The actual parameter class
# -----------------------------------------------------------------------------

class _Param:
    """Internal parameter class. Use `Param` builder to create instances.

    Three mutually-exclusive roles:
    - THETA: Free parameter for optimization (has min/max/start/value)
    - INPUT: Placeholder for input data
    - INDEX: Integer loop variable (has range)
    """

    theta = ParamKind.THETA
    input = ParamKind.INPUT
    index = ParamKind.INDEX

    def __init__(
        self,
        name: str | None = None,
        min: float | None = None,
        max: float | None = None,
        start: float | None = None,
        *,
        kind: ParamKind = ParamKind.THETA,
        range: range | None = None,
    ):
        self.kind = kind
        self.name = name

        if kind is ParamKind.INPUT:
            return

        if kind is ParamKind.INDEX:
            self.range = range
            return

        # THETA
        self.range = None
        self.min = min if min is not None else -np.inf
        self.max = max if max is not None else np.inf

        # Compute default start
        if start is not None:
            self.start = start
        elif self.min == -np.inf and self.max == np.inf:
            self.start = 0.0
        elif self.min == -np.inf:
            self.start = float(self.max)
        elif self.max == np.inf:
            self.start = float(self.min)
        else:
            self.start = 0.5 * (self.min + self.max)

        self.value = self.start

    def __call__(self, arg0: str | float, arg1: float | None = None) -> '_Param':
        """Chain configuration: param('name'), param(5), param(0, 10)."""
        if arg1 is not None:
            # Set bounds
            new_min, new_max = float(arg0), float(arg1)
            # Recompute start if needed
            if self.start < new_min or self.start > new_max:
                if new_min == -np.inf:
                    new_start = float(new_max) if new_max != np.inf else 0.0
                elif new_max == np.inf:
                    new_start = float(new_min)
                else:
                    new_start = 0.5 * (new_min + new_max)
            else:
                new_start = self.start
            return _Param(
                name=self.name,
                min=new_min,
                max=new_max,
                start=new_start,
            )
        elif isinstance(arg0, str):
            # Set name
            return _Param(
                name=arg0,
                min=self.min,
                max=self.max,
                start=self.start,
            )
        elif isinstance(arg0, (int, float)):
            # Set start
            new_start = float(arg0)
            # Clamp to bounds
            if new_start < self.min:
                new_start = self.min
            if new_start > self.max:
                new_start = self.max
            return _Param(
                name=self.name,
                min=self.min,
                max=self.max,
                start=new_start,
            )
        else:
            raise TypeError(
                f"_Param(...) expects str (name), number (start), or two numbers (bounds), "
                f"got {type(arg0).__name__}"
            )

    # ------------------------------------------------------------------
    #   Identity-based equality (operators are added by bind_ops in model.py)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Param):
            return False
        if self.kind is self.input and other.kind is other.input:
            return True
        return self is other

    def __hash__(self) -> int:
        return id(self)

    # ------------------------------------------------------------------
    #   NumPy dispatch protocols
    # ------------------------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **ufunc_kwargs):
        if method != '__call__':
            return NotImplemented
        from .model import Model

        if ufunc_kwargs:
            def inner(*inner_args):
                return ufunc(*inner_args, **ufunc_kwargs)
            inner._has_kwargs = True
            inner._kwargs_info = tuple(ufunc_kwargs.keys())
        else:
            def inner(*inner_args):
                return ufunc(*inner_args)
        inner.__name__ = ufunc.__name__
        return Model(inner, list(inputs))

    def __array_function__(self, func, types, args, kwargs):
        from .model import Model

        if kwargs:
            def inner(*inner_args):
                return func(*inner_args, **kwargs)
            inner._has_kwargs = True
            inner._kwargs_info = tuple(kwargs.keys())
        else:
            def inner(*inner_args):
                return func(*inner_args)
        inner.__name__ = func.__name__
        return Model(inner, list(args))

    # ------------------------------------------------------------------
    #   Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """Select element from this param by index.

        INPUT[0] returns a Model that selects element 0 from the input array.
        INPUT[idx] where idx is an INDEX param returns a Model for dynamic indexing.
        """
        from .model import _indecise
        return _indecise(self, key)

    # ------------------------------------------------------------------
    #   String representations
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
            return f"{self._prefix()}_Param(value={self.value})"
        return (
            f"{self._prefix()}_Param(min={self.min}, start={self.start}, "
            f"max={self.max}, value={self.value})"
        )

    def __str__(self) -> str:
        if self.kind is self.input:
            return "INPUT"
        if self.kind is self.index:
            if self.range is None:
                return "INDEX"
            return f"{self._prefix()}{self.range}"
        if hasattr(self, "error"):
            return f"{self._prefix()}{self.value:.4g} ± {self.error:.2g}"
        return f"{self._prefix()}{self.value:.4g}"


# -----------------------------------------------------------------------------
#   Param: The builder class (public API)
# -----------------------------------------------------------------------------

_MISSING = object()  # sentinel for distinguishing Param() from Param(None)


class _ParamBuilder:
    """Builder for creating _Param instances.

    Use via the singleton `Param`:
        a = Param(5)           # _Param with start=5, auto-named 'a'
        b = Param('b')         # _Param with explicit name 'b'
        c = Param(0, 10)       # _Param with bounds [0, 10], auto-named 'c'
    """

    def __call__(self, arg0: str | float = _MISSING, arg1: float | None = None) -> _Param:
        """Create a _Param with configuration."""
        name = None
        start = None
        bounds = None

        if arg0 is _MISSING:
            # Param() - just auto-name
            name = _detect_varname(depth=2)
        elif arg1 is not None:
            # Param(min, max) - bounds
            bounds = (float(arg0), float(arg1))
            name = _detect_varname(depth=2)
        elif isinstance(arg0, str):
            # Param('name') - explicit name
            name = arg0
        elif isinstance(arg0, (int, float)):
            # Param(5) - start value
            start = float(arg0)
            name = _detect_varname(depth=2)
        else:
            raise TypeError(
                f"Param(...) expects str (name), number (start), or two numbers (bounds), "
                f"got {type(arg0).__name__}"
            )

        if bounds is not None:
            return _Param(name=name, min=bounds[0], max=bounds[1], start=start)
        else:
            return _Param(name=name, start=start)

    def _not_finalized_error(self):
        raise TypeError(
            "Param must be called before use in expressions.\n"
            "  Examples:\n"
            "    Param('x')       # named\n"
            "    Param(5)         # with start value\n"
            "    Param(0, 10)     # with bounds\n"
            "    Param()          # unbounded, auto-named"
        )

    def __add__(self, other): self._not_finalized_error()
    def __radd__(self, other): self._not_finalized_error()
    def __sub__(self, other): self._not_finalized_error()
    def __rsub__(self, other): self._not_finalized_error()
    def __mul__(self, other): self._not_finalized_error()
    def __rmul__(self, other): self._not_finalized_error()
    def __truediv__(self, other): self._not_finalized_error()
    def __rtruediv__(self, other): self._not_finalized_error()
    def __pow__(self, other): self._not_finalized_error()
    def __rpow__(self, other): self._not_finalized_error()

    def __invert__(self) -> _Param:
        """~Param: Create unbounded _Param with auto-naming."""
        return _Param(name=_detect_varname(depth=2))

    def __pos__(self) -> _Param:
        """+Param: Create positive _Param with auto-naming."""
        return _Param(name=_detect_varname(depth=2), min=1e-6, max=np.inf, start=1.0)

    def __neg__(self) -> _Param:
        """-Param: Create negative _Param with auto-naming."""
        return _Param(name=_detect_varname(depth=2), min=-np.inf, max=-1e-6, start=-1.0)

    def __repr__(self) -> str:
        return "Param"


# The singleton builder
Param = _ParamBuilder()


# -----------------------------------------------------------------------------
#   Constrained param builders
# -----------------------------------------------------------------------------

class _PositiveBuilder:
    """Builder for positive params: Param.positive('name') or Param.positive(start)"""
    def __call__(self, arg0: str | float | None = None) -> _Param:
        if arg0 is None:
            name = _detect_varname(depth=2)
            return _Param(name=name, min=1e-6, max=np.inf, start=1.0)
        elif isinstance(arg0, str):
            return _Param(name=arg0, min=1e-6, max=np.inf, start=1.0)
        elif isinstance(arg0, (int, float)):
            name = _detect_varname(depth=2)
            start = float(arg0)
            if start <= 0:
                start = 1.0
            return _Param(name=name, min=1e-6, max=np.inf, start=start)
        else:
            raise TypeError(f"Param.positive expects str or number, got {type(arg0).__name__}")

    def __repr__(self) -> str:
        return "Param.positive"


class _NegativeBuilder:
    """Builder for negative params: Param.negative('name') or Param.negative(start)"""
    def __call__(self, arg0: str | float | None = None) -> _Param:
        if arg0 is None:
            name = _detect_varname(depth=2)
            return _Param(name=name, min=-np.inf, max=-1e-6, start=-1.0)
        elif isinstance(arg0, str):
            return _Param(name=arg0, min=-np.inf, max=-1e-6, start=-1.0)
        elif isinstance(arg0, (int, float)):
            name = _detect_varname(depth=2)
            start = float(arg0)
            if start >= 0:
                start = -1.0
            return _Param(name=name, min=-np.inf, max=-1e-6, start=start)
        else:
            raise TypeError(f"Param.negative expects str or number, got {type(arg0).__name__}")

    def __repr__(self) -> str:
        return "Param.negative"


class _UnitBuilder:
    """Builder for unit [0,1] params: Param.unit('name') or Param.unit(start)"""
    def __call__(self, arg0: str | float | None = None) -> _Param:
        if arg0 is None:
            name = _detect_varname(depth=2)
            return _Param(name=name, min=0, max=1, start=0.5)
        elif isinstance(arg0, str):
            return _Param(name=arg0, min=0, max=1, start=0.5)
        elif isinstance(arg0, (int, float)):
            name = _detect_varname(depth=2)
            start = float(arg0)
            if start < 0:
                start = 0.0
            if start > 1:
                start = 1.0
            return _Param(name=name, min=0, max=1, start=start)
        else:
            raise TypeError(f"Param.unit expects str or number, got {type(arg0).__name__}")

    def __repr__(self) -> str:
        return "Param.unit"


Param.positive = _PositiveBuilder()
Param.negative = _NegativeBuilder()
Param.unit = _UnitBuilder()


# -----------------------------------------------------------------------------
#   Module-level helpers
# -----------------------------------------------------------------------------

INPUT: _Param = _Param(kind=ParamKind.INPUT)

def index(*args: int) -> _Param:
    """Create an INDEX parameter with range(*args)."""
    return _Param(range=range(*args) if args else None, kind=ParamKind.INDEX)

INDEX: _Param = index()
