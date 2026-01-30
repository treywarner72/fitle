# tests/test_param.py
import pytest
import numpy as np
from fitle import Param, _Param, INPUT, INDEX, index


class TestThetaCreation:
    """Basic THETA parameter creation."""

    def test_empty_param(self):
        """Param() creates unbounded param with start=0 and auto-naming."""
        p = Param()
        assert p.min == -np.inf
        assert p.max == np.inf
        assert p.start == 0
        assert p.value == 0
        # Now auto-named to 'p' from variable assignment
        assert p.name == 'p'

    def test_param_with_name(self):
        p = Param('mu')
        assert p.name == 'mu'

    def test_param_with_start(self):
        p = Param(5.0)
        assert p.start == 5.0
        assert p.value == 5.0

    def test_param_with_bounds(self):
        p = Param()(0, 10)
        assert p.min == 0
        assert p.max == 10

    def test_param_with_all(self):
        p = Param('x')(0, 10)(5)
        assert p.name == 'x'
        assert p.min == 0
        assert p.max == 10
        assert p.start == 5


class TestChainableAPI:
    """Chainable __call__ works in any order."""

    def test_name_bounds_start(self):
        p = Param('x')(0, 10)(5)
        assert p.name == 'x' and p.min == 0 and p.max == 10 and p.start == 5

    def test_start_name(self):
        p = Param(5)('x')
        assert p.name == 'x' and p.start == 5

    def test_bounds_then_name(self):
        p = Param()(0, 10)('x')
        assert p.name == 'x' and p.min == 0 and p.max == 10

    def test_bounds_then_start(self):
        p = Param()(0, 10)(5)
        assert p.min == 0 and p.max == 10 and p.start == 5

    def test_start_then_bounds(self):
        """Start set first, then bounds that contain it."""
        p = Param(5)(0, 10)
        assert p.start == 5 and p.min == 0 and p.max == 10


class TestFactories:
    """Param.positive() and Param.unit() factories."""

    def test_positive_defaults(self):
        p = Param.positive()
        assert p.min == 1e-6
        assert p.max == np.inf
        assert p.start == 1

    def test_positive_with_name(self):
        p = Param.positive('sigma')
        assert p.name == 'sigma'
        assert p.min == 1e-6

    def test_positive_with_start(self):
        p = Param.positive(5)
        assert p.start == 5
        assert p.min == 1e-6

    def test_positive_with_name_and_start(self):
        p = Param.positive('sigma')(5)
        assert p.name == 'sigma' and p.start == 5

    def test_unit_defaults(self):
        p = Param.unit()
        assert p.min == 0
        assert p.max == 1
        assert p.start == 0.5

    def test_unit_with_name(self):
        p = Param.unit('frac')
        assert p.name == 'frac'

    def test_unit_with_start(self):
        p = Param.unit(0.3)
        assert p.start == 0.3


class TestDefaultStart:
    """Start value defaults based on bounds."""

    def test_unbounded_default_zero(self):
        p = Param()
        assert p.start == 0

    def test_bounded_default_midpoint(self):
        p = Param(0, 10)
        assert p.start == 5

    def test_min_only_default_to_min(self):
        p = Param(5, np.inf)
        assert p.start == 5

    def test_max_only_default_to_max(self):
        p = Param(-np.inf, 5)
        assert p.start == 5


class TestBoundsValidation:
    """Start outside bounds behavior."""

    def test_start_outside_bounds_clamps(self):
        """Start outside bounds gets clamped (new behavior)."""
        p = Param(0, 10)(-5)
        # Now clamps instead of raising
        assert p.start == 0  # clamped to min

    def test_start_above_bounds_clamps(self):
        p = Param(0, 10)(15)
        assert p.start == 10  # clamped to max

    def test_start_at_min_ok(self):
        p = Param(0, 10)(0)
        assert p.start == 0

    def test_start_at_max_ok(self):
        p = Param(0, 10)(10)
        assert p.start == 10


class TestThetaEquality:
    """THETA params use identity-based equality."""

    def test_same_object_equal(self):
        p = Param('x')
        assert p == p

    def test_different_objects_not_equal(self):
        p1 = Param('x')
        p2 = Param('x')
        assert p1 != p2

    def test_hash_is_identity(self):
        p = Param('x')
        assert hash(p) == id(p)

    def test_can_use_in_set(self):
        p1 = Param('x')
        p2 = Param('y')
        s = {p1, p2, p1}
        assert len(s) == 2


class TestInput:
    """INPUT placeholder tests."""

    def test_input_singleton_equality(self):
        assert INPUT == INPUT

    def test_input_kind(self):
        assert INPUT.kind == _Param.input

    def test_input_repr(self):
        assert repr(INPUT) == 'INPUT'


class TestIndex:
    """INDEX parameter tests."""

    def test_index_no_range(self):
        i = index()
        assert i.kind == _Param.index
        assert i.range is None

    def test_index_with_stop(self):
        i = index(10)
        assert i.range == range(10)

    def test_index_with_start_stop(self):
        i = index(5, 10)
        assert i.range == range(5, 10)

    def test_index_with_step(self):
        i = index(0, 10, 2)
        assert i.range == range(0, 10, 2)

    def test_INDEX_is_default(self):
        assert INDEX.kind == _Param.index
        assert INDEX.range is None


class TestEdgeCases:
    """Numeric and type edge cases."""

    def test_fixed_param_min_equals_max(self):
        p = Param(5, 5)
        assert p.min == p.max == 5
        assert p.start == 5

    def test_numpy_scalar_start(self):
        p = Param(np.float64(3.14))
        assert p.start == 3.14

    def test_int_start_becomes_float(self):
        p = Param(5)
        assert isinstance(p.start, float)

    def test_very_small_bounds(self):
        p = Param(1e-10, 2e-10)
        assert p.start == 1.5e-10

    def test_very_large_bounds(self):
        p = Param(1e10, 2e10)
        assert p.start == 1.5e10

    def test_value_mutation(self):
        p = Param(0, 10)(5)
        p.value = 7
        assert p.value == 7
        assert p.start == 5  # start unchanged
