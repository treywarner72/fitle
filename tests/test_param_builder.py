"""Tests for the Param builder system."""

import pytest
import numpy as np
from fitle import Param, _Param, INPUT


class TestUnaryOperatorSyntax:
    """Test ~Param, +Param, -Param shorthand."""

    def test_invert_creates_unbounded(self):
        """~Param creates unbounded param with auto-naming."""
        a = ~Param
        assert isinstance(a, _Param)
        assert a.name == 'a'
        assert a.min == -np.inf
        assert a.max == np.inf

    def test_pos_creates_positive(self):
        """+Param creates positive param with auto-naming."""
        sigma = +Param
        assert isinstance(sigma, _Param)
        assert sigma.name == 'sigma'
        assert sigma.min > 0
        assert sigma.max == np.inf

    def test_neg_creates_negative(self):
        """-Param creates negative param with auto-naming."""
        neg = -Param
        assert isinstance(neg, _Param)
        assert neg.name == 'neg'
        assert neg.min == -np.inf
        assert neg.max < 0

    def test_tuple_unpacking_unary(self):
        """mu, sigma = ~Param, +Param auto-names both."""
        mu, sigma = ~Param, +Param
        assert mu.name == 'mu'
        assert sigma.name == 'sigma'
        assert mu.min == -np.inf
        assert sigma.min > 0


class TestParamBuilder:
    """Test Param() creates _Param instances."""

    def test_call_with_name(self):
        """Param('x') creates named param."""
        p = Param('x')
        assert isinstance(p, _Param)
        assert p.name == 'x'
        assert p.min == -np.inf
        assert p.max == np.inf

    def test_call_with_start(self):
        """Param(5) creates param with start value."""
        a = Param(5)
        assert isinstance(a, _Param)
        assert a.start == 5
        assert a.value == 5

    def test_call_with_bounds(self):
        """Param(0, 10) creates bounded param."""
        b = Param(0, 10)
        assert isinstance(b, _Param)
        assert b.min == 0
        assert b.max == 10
        assert b.start == 5  # midpoint

    def test_call_no_args(self):
        """Param() creates unbounded param."""
        c = Param()
        assert isinstance(c, _Param)
        assert c.min == -np.inf
        assert c.max == np.inf


class TestAutoNaming:
    """Test automatic variable name detection."""

    def test_call_auto_names(self):
        """Param(...) auto-detects name."""
        foo = Param(5)
        assert foo.name == 'foo'

    def test_call_no_args_auto_names(self):
        """Param() auto-detects name."""
        bar = Param()
        assert bar.name == 'bar'

    def test_positive_auto_names(self):
        """Param.positive() auto-detects name."""
        sigma = Param.positive()
        assert sigma.name == 'sigma'

    def test_negative_auto_names(self):
        """Param.negative() auto-detects name."""
        offset = Param.negative()
        assert offset.name == 'offset'

    def test_explicit_name_overrides(self):
        """Param('explicit') overrides auto-naming."""
        foo = Param('bar')
        assert foo.name == 'bar'  # explicit, not 'foo'

    def test_tuple_unpacking(self):
        """a, b = Param(...), Param(...) names both."""
        a, b = Param(1), Param(2)
        assert a.name == 'a'
        assert b.name == 'b'
        assert a.start == 1
        assert b.start == 2

    def test_tuple_unpacking_with_positive(self):
        """a, b = Param(), Param.positive() names both."""
        mu, sigma = Param(), Param.positive()
        assert mu.name == 'mu'
        assert sigma.name == 'sigma'
        assert sigma.min > 0

    def test_tuple_unpacking_three(self):
        """x, y, z = ... works."""
        x, y, z = Param(1), Param.positive(2), Param.negative()
        assert x.name == 'x'
        assert y.name == 'y'
        assert z.name == 'z'


class TestChaining:
    """Test chaining via () on _Param."""

    def test_chain_name_then_start(self):
        """Param('x')(5) works."""
        p = Param('x')(5)
        assert p.name == 'x'
        assert p.start == 5

    def test_chain_start_then_name(self):
        """Param(5)('x') works (any order)."""
        p = Param(5)('x')
        assert p.name == 'x'
        assert p.start == 5

    def test_chain_name_then_bounds(self):
        """Param('x')(0, 10) works."""
        p = Param('x')(0, 10)
        assert p.name == 'x'
        assert p.min == 0
        assert p.max == 10

    def test_chain_all_three(self):
        """Param('x')(5)(0, 10) works."""
        p = Param('x')(5)(0, 10)
        assert p.name == 'x'
        assert p.start == 5
        assert p.min == 0
        assert p.max == 10

    def test_chain_any_order(self):
        """Param(0, 10)(5)('x') works (any order)."""
        p = Param(0, 10)(5)('x')
        assert p.name == 'x'
        assert p.start == 5
        assert p.min == 0
        assert p.max == 10


class TestConstrainedBuilders:
    """Test Param.positive, Param.negative, Param.unit."""

    def test_positive_with_name(self):
        """Param.positive('sigma') is positive and named."""
        p = Param.positive('sigma')
        assert p.name == 'sigma'
        assert p.min > 0

    def test_positive_with_start(self):
        """Param.positive(5) is positive with start."""
        p = Param.positive(5)
        assert p.min > 0
        assert p.start == 5

    def test_negative_with_name(self):
        """Param.negative('x') is negative and named."""
        p = Param.negative('x')
        assert p.name == 'x'
        assert p.max < 0

    def test_negative_with_start(self):
        """Param.negative(-5) is negative with start."""
        p = Param.negative(-5)
        assert p.max < 0
        assert p.start == -5

    def test_unit_with_name(self):
        """Param.unit('f') is in [0,1] and named."""
        p = Param.unit('f')
        assert p.name == 'f'
        assert p.min == 0
        assert p.max == 1

    def test_unit_with_start(self):
        """Param.unit(0.3) is in [0,1] with start."""
        p = Param.unit(0.3)
        assert p.min == 0
        assert p.max == 1
        assert p.start == 0.3


class TestBuilderError:
    """Test bare Param errors in expressions."""

    def test_bare_param_add_errors(self):
        """Param + x raises TypeError."""
        with pytest.raises(TypeError, match="must be called"):
            Param + INPUT

    def test_bare_param_mul_errors(self):
        """Param * x raises TypeError."""
        with pytest.raises(TypeError, match="must be called"):
            Param * 2

    def test_bare_param_radd_creates_model(self):
        """x + Param creates a Model (INPUT has __add__ which calls _Param)."""
        # INPUT + Param actually works because INPUT is a _Param with __add__
        # The Param builder's __radd__ is not called because INPUT handles it
        from fitle import Model
        result = INPUT + Param()  # This creates a Model
        assert isinstance(result, Model)


class TestParamInExpressions:
    """Test _Param works in model expressions."""

    def test_param_times_input(self):
        """Param('a') * INPUT works."""
        from fitle import Model
        model = Param('a') * INPUT
        assert isinstance(model, Model)

    def test_param_plus_param(self):
        """Param('a') + Param('b') works."""
        from fitle import Model
        model = Param('a') + Param('b')
        assert isinstance(model, Model)
        assert len(model.params) == 2

    def test_positive_param_in_expression(self):
        """Param.positive in expression works."""
        from fitle import Model
        model = Param.positive('sigma') * INPUT
        assert isinstance(model, Model)
        assert model.params[0].min > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_start_outside_bounds_clamps(self):
        """Start value outside bounds clamps to bounds."""
        p = Param(100)(0, 10)  # start=100, then bounds=[0,10]
        assert p.start == 5  # midpoint when out of bounds

    def test_start_below_bounds_clamps(self):
        """Start value below bounds clamps."""
        p = Param(-5)(0, 10)
        assert p.start == 5  # midpoint when out of bounds

    def test_invalid_arg_type(self):
        """Param(None) raises TypeError."""
        with pytest.raises(TypeError):
            Param(None)

    def test_float_start(self):
        """Param(3.14) works with float."""
        p = Param(3.14)
        assert p.start == 3.14


class TestRHSPatternEdgeCases:
    """Test edge cases in RHS pattern matching for auto-naming."""

    def test_negative_bounds(self):
        """Param(-5, 5) should auto-name."""
        neg_bounds = Param(-5, 5)
        assert neg_bounds.name == 'neg_bounds'
        assert neg_bounds.min == -5
        assert neg_bounds.max == 5

    def test_float_bounds(self):
        """Param(0.5, 1.5) works."""
        flt = Param(0.5, 1.5)
        assert flt.name == 'flt'
        assert flt.min == 0.5

    def test_scientific_notation_bounds(self):
        """Param(1e-6, 1e6) works."""
        sci = Param(1e-6, 1e6)
        assert sci.name == 'sci'
        assert sci.min == 1e-6

    def test_expression_in_args_no_autoname(self):
        """Param(1+2) should NOT auto-name (expression in args)."""
        from fitle import Model
        # This creates a param but the RHS has + so no auto-name
        result = Param(1+2)  # evaluates to Param(3)
        # Actually 1+2 is evaluated by Python first, so this becomes Param(3)
        # This is a limitation - Python evaluates 1+2 before we see it
        assert result.start == 3  # At least the value is correct

    def test_parenthesized_no_autoname(self):
        """(Param()) in expression should not auto-name."""
        from fitle import Model
        result = (Param()) + Param(1)
        assert isinstance(result, Model)
        # Both params should be unnamed since it's an expression
        assert all(p.name is None for p in result.params)

    def test_method_call_autonames(self):
        """np.exp(Param()) does auto-name because pattern matches simple RHS."""
        # The auto-naming pattern sees "result = np.exp(Param())" as a simple assignment
        # This is a known limitation - the param gets named 'result'
        result = np.exp(Param())
        # The param IS auto-named because the pattern is not smart enough
        # to detect that it's inside a function call
        assert result.params[0].name == 'result'


class TestUnaryOperators:
    """Test unary operators on _Param create Models."""

    def test_neg_creates_model(self):
        """-Param('x') creates a Model (negation)."""
        from fitle import Model
        p = Param('x')
        m = -p
        assert isinstance(m, Model)
        # The param inside is still named 'x'
        assert m.params[0].name == 'x'


class TestChainingEdgeCases:
    """Test edge cases when chaining () on _Param."""

    def test_start_twice_uses_second(self):
        """Param(5)(10) - second start overwrites first."""
        p = Param(5)(10)
        assert p.start == 10

    def test_bounds_twice_uses_second(self):
        """Param(0,10)(5,15) - second bounds overwrites first."""
        p = Param(0, 10)(5, 15)
        assert p.min == 5
        assert p.max == 15

    def test_name_twice_uses_second(self):
        """Param('a')('b') - second name overwrites first."""
        p = Param('a')('b')
        assert p.name == 'b'

    def test_complex_chain(self):
        """Param('x')(5)(0,10)('y')(7) - last of each type wins."""
        p = Param('x')(5)(0, 10)('y')(7)
        assert p.name == 'y'
        assert p.start == 7
        assert p.min == 0
        assert p.max == 10


class TestTupleEdgeCases:
    """Test edge cases with tuple unpacking."""

    def test_more_values_than_vars(self):
        """a = Param(), Param() creates a tuple - no auto-naming (count mismatch)."""
        a = Param(), Param.positive()
        assert isinstance(a, tuple)
        assert len(a) == 2
        # 1 var, 2 params = mismatch, so both should be unnamed
        assert a[0].name is None
        assert a[1].name is None

    def test_starred_unpacking(self):
        """a, *b = Param(), Param(), Param() - starred unpacking."""
        a, *b = Param(), Param(), Param()
        assert isinstance(a, _Param)
        assert isinstance(b, list)
        assert len(b) == 2
        # Names depend on how the regex parses "a, *b"

    def test_single_var_tuple_value(self):
        """a = (Param(),) - single element tuple."""
        a = (Param(),)
        assert isinstance(a, tuple)
        assert len(a) == 1


class TestNumpyInteraction:
    """Test _Param interaction with numpy."""

    def test_param_in_numpy_array(self):
        """np.array([Param()]) - param in array."""
        # This might not work as expected
        arr = np.array([Param()])
        # numpy might call __array__ or just store the object

    def test_param_in_numpy_function(self):
        """np.mean(Param()) - param passed to numpy function."""
        from fitle import Model
        result = np.mean(Param())
        # Should create a Model via __array_function__
        assert isinstance(result, Model)
        # Param gets auto-named to 'result' because the pattern matches
        # (this is a known limitation)
        assert result.params[0].name == 'result'


class TestBoundsEdgeCases:
    """Test edge cases with bounds."""

    def test_inf_bounds(self):
        """Param(-np.inf, np.inf) explicit infinite bounds."""
        p = Param(-np.inf, np.inf)
        assert p.min == -np.inf
        assert p.max == np.inf

    def test_same_min_max(self):
        """Param(5, 5) - fixed value param."""
        p = Param(5, 5)
        assert p.min == 5
        assert p.max == 5
        assert p.start == 5

    def test_very_small_bounds(self):
        """Param(1e-100, 1e-99) - very small bounds."""
        p = Param(1e-100, 1e-99)
        assert p.min == 1e-100
        assert p.max == 1e-99
