# tests/test_model.py
"""Comprehensive tests for Model - the core of fitle."""
import pytest
import numpy as np
import operator
from fitle import Param, INPUT, INDEX, index, Model, const, identity, Reduction
from fitle.model import _indecise


# =============================================================================
# Model Creation
# =============================================================================

class TestModelCreation:
    """Basic Model construction."""

    def test_model_with_fn_and_args(self):
        fn = lambda x: x * 2
        m = Model(fn, [5])
        assert m.fn is fn
        assert m.args == [5]

    def test_model_with_param_arg(self):
        p = Param('x')
        m = Model(lambda x: x, [p])
        assert p in m.args

    def test_model_with_nested_model(self):
        inner = Model(lambda x: x, [Param('a')])
        outer = Model(lambda x: x + 1, [inner])
        assert outer.args[0] is inner

    def test_const_scalar(self):
        m = const(42)
        assert m() == 42

    def test_const_array(self):
        arr = np.array([1, 2, 3])
        m = const(arr)
        np.testing.assert_array_equal(m(), arr)

    def test_identity_passthrough(self):
        p = Param('x')(5)
        m = identity(p)
        assert m() == 5


# =============================================================================
# Arithmetic Operations - Model with Model
# =============================================================================

class TestModelArithmetic:
    """Arithmetic between Models."""

    def test_add_models(self):
        a = Param('a')(2)
        b = Param('b')(3)
        m = a + b
        assert m() == 5

    def test_sub_models(self):
        a = Param('a')(5)
        b = Param('b')(3)
        m = a - b
        assert m() == 2

    def test_mul_models(self):
        a = Param('a')(4)
        b = Param('b')(3)
        m = a * b
        assert m() == 12

    def test_div_models(self):
        a = Param('a')(10)
        b = Param('b')(2)
        m = a / b
        assert m() == 5

    def test_pow_models(self):
        a = Param('a')(2)
        b = Param('b')(3)
        m = a ** b
        assert m() == 8

    def test_neg_model(self):
        a = Param('a')(5)
        m = -a
        assert m() == -5


# =============================================================================
# Arithmetic Operations - Model with Scalar
# =============================================================================

class TestModelScalarArithmetic:
    """Arithmetic between Model/Param and scalars."""

    def test_model_add_scalar(self):
        a = Param('a')(2)
        m = a + 3
        assert m() == 5

    def test_scalar_add_model(self):
        a = Param('a')(2)
        m = 3 + a
        assert m() == 5

    def test_model_sub_scalar(self):
        a = Param('a')(5)
        m = a - 2
        assert m() == 3

    def test_scalar_sub_model(self):
        a = Param('a')(2)
        m = 5 - a
        assert m() == 3

    def test_model_mul_scalar(self):
        a = Param('a')(4)
        m = a * 3
        assert m() == 12

    def test_scalar_mul_model(self):
        a = Param('a')(4)
        m = 3 * a
        assert m() == 12

    def test_model_div_scalar(self):
        a = Param('a')(10)
        m = a / 2
        assert m() == 5

    def test_scalar_div_model(self):
        a = Param('a')(2)
        m = 10 / a
        assert m() == 5

    def test_model_pow_scalar(self):
        a = Param('a')(2)
        m = a ** 3
        assert m() == 8

    def test_scalar_pow_model(self):
        a = Param('a')(3)
        m = 2 ** a
        assert m() == 8


# =============================================================================
# Arithmetic with INPUT
# =============================================================================

class TestInputArithmetic:
    """Arithmetic involving INPUT placeholder."""

    def test_input_add_scalar(self):
        m = INPUT + 1
        np.testing.assert_array_equal(m([0, 1, 2]), [1, 2, 3])

    def test_scalar_add_input(self):
        m = 1 + INPUT
        np.testing.assert_array_equal(m([0, 1, 2]), [1, 2, 3])

    def test_input_mul_param(self):
        a = Param('a')(2)
        m = INPUT * a
        np.testing.assert_array_equal(m([1, 2, 3]), [2, 4, 6])

    def test_complex_expression(self):
        a = Param('a')(2)
        b = Param('b')(1)
        m = a * INPUT + b  # y = 2x + 1
        np.testing.assert_array_equal(m([0, 1, 2]), [1, 3, 5])

    def test_polynomial(self):
        a = Param('a')(1)
        b = Param('b')(0)
        c = Param('c')(0)
        m = a * INPUT**2 + b * INPUT + c  # y = x^2
        np.testing.assert_array_equal(m([0, 1, 2, 3]), [0, 1, 4, 9])


# =============================================================================
# Substitution (% operator)
# =============================================================================

class TestSubstitution:
    """Model substitution with % operator."""

    def test_substitute_input_with_dict(self):
        m = INPUT * 2
        result = m % {INPUT: np.array([1, 2, 3])}
        np.testing.assert_array_equal(result(), [2, 4, 6])

    def test_substitute_input_shorthand(self):
        m = INPUT * 2
        result = m % np.array([1, 2, 3])
        np.testing.assert_array_equal(result(), [2, 4, 6])

    def test_substitute_param_with_value(self):
        a = Param('a')
        m = a * 2
        result = m % {a: 5}
        assert result() == 10

    def test_substitute_param_with_model(self):
        a = Param('a')
        b = Param('b')(3)
        m = a * 2
        result = m % {a: b}
        assert result() == 6

    def test_nested_substitution(self):
        a = Param('a')
        m = (a + 1) * 2
        result = m % {a: 4}
        assert result() == 10

    def test_substitute_preserves_other_params(self):
        a = Param('a')(2)
        b = Param('b')
        m = a + b
        result = m % {b: 3}
        assert result() == 5


# =============================================================================
# Evaluation (__call__)
# =============================================================================

class TestEvaluation:
    """Model evaluation via __call__."""

    def test_call_with_input(self):
        m = INPUT * 2
        result = m(np.array([1, 2, 3]))
        np.testing.assert_array_equal(result, [2, 4, 6])

    def test_call_without_input(self):
        a = Param('a')(5)
        m = a * 2
        assert m() == 10

    def test_call_with_input_raises_when_no_input(self):
        a = Param('a')(5)
        m = a * 2
        with pytest.raises(Exception):
            m(np.array([1, 2, 3]))

    def test_call_without_input_raises_when_input_present(self):
        m = INPUT * 2
        with pytest.raises(Exception):
            m()

    def test_scalar_input(self):
        m = INPUT * 2
        assert m(5) == 10

    def test_empty_array_input(self):
        m = INPUT * 2
        result = m(np.array([]))
        assert len(result) == 0


# =============================================================================
# Properties
# =============================================================================

class TestModelProperties:
    """Model properties: params, free, shape, components."""

    def test_params_single(self):
        a = Param('a')
        m = a * 2
        assert m.params == [a]

    def test_params_multiple(self):
        a = Param('a')
        b = Param('b')
        m = a + b
        assert set(m.params) == {a, b}

    def test_params_nested(self):
        a = Param('a')
        b = Param('b')
        m = (a * 2) + (b * 3)
        assert set(m.params) == {a, b}

    def test_params_no_duplicates(self):
        a = Param('a')
        m = a + a + a
        assert m.params == [a]

    def test_free_with_input(self):
        m = INPUT * 2
        assert INPUT in m.free

    def test_free_without_input(self):
        a = Param('a')
        m = a * 2
        assert m.free == []

    def test_free_with_index(self):
        i = index(10)
        arr = const(np.arange(10))
        m = _indecise(arr, i)
        assert i in m.free

    def test_components_no_identity(self):
        a = Param('a')
        m = a * 2
        assert m.components == [m]

    def test_components_with_identity(self):
        a = Param('a')
        b = Param('b')
        m = identity(a) + identity(b)
        assert len(m.components) == 2


# =============================================================================
# Hashing and Equality
# =============================================================================

class TestHashingEquality:
    """Model hashing and equality for caching."""

    def test_same_structure_same_hash(self):
        a1 = Param('a')
        a2 = Param('a')
        m1 = a1 * 2 + 1
        m2 = a2 * 2 + 1
        assert hash(m1) == hash(m2)

    def test_different_structure_different_hash(self):
        a = Param('a')
        m1 = a * 2
        m2 = a + 2
        assert hash(m1) != hash(m2)

    def test_different_const_different_hash(self):
        a = Param('a')
        m1 = a * 2
        m2 = a * 3
        assert hash(m1) != hash(m2)

    def test_model_equality_same_param(self):
        # Same Param object -> models are equal
        a = Param('a')
        m1 = a * 2
        m2 = a * 2
        assert m1 == m2

    def test_model_equality_different_params_not_equal(self):
        # Different Param objects -> models not equal (identity-based)
        a1 = Param('a')
        a2 = Param('a')
        m1 = a1 * 2
        m2 = a2 * 2
        assert m1 != m2  # Different param instances

    def test_model_equality_different_structure(self):
        a = Param('a')
        m1 = a * 2
        m2 = a + 2
        assert m1 != m2

    def test_model_equality_self(self):
        a = Param('a')
        m = a * 2
        assert m == m


# =============================================================================
# copy()
# =============================================================================

class TestModelCopy:
    """Deep copy of models."""

    def test_copy_creates_new_model(self):
        a = Param('a')(5)
        m = a * 2
        m2 = m.copy()
        assert m is not m2

    def test_copy_clones_params(self):
        a = Param('a')(5)
        m = a * 2
        m2 = m.copy()
        # Params should be different objects
        assert m.params[0] is not m2.params[0]

    def test_copy_preserves_values(self):
        a = Param('a')(5)
        m = a * 2
        m2 = m.copy()
        assert m2() == 10

    def test_copy_params_independent(self):
        a = Param('a')(5)
        m = a * 2
        m2 = m.copy()
        # Changing original param doesn't affect copy
        a.value = 10
        assert m() == 20
        assert m2() == 10

    def test_copy_shares_input(self):
        m = INPUT * 2
        m2 = m.copy()
        # INPUT should be shared
        assert m.free[0] is m2.free[0]


# =============================================================================
# freeze()
# =============================================================================

class TestFreeze:
    """Freezing models substitutes params with current values."""

    def test_freeze_removes_params(self):
        a = Param('a')(5)
        m = a * 2
        frozen = m.freeze()
        assert frozen.params == []

    def test_freeze_preserves_value(self):
        a = Param('a')(5)
        m = a * 2
        frozen = m.freeze()
        assert frozen() == 10

    def test_freeze_with_input(self):
        a = Param('a')(2)
        m = a * INPUT
        frozen = m.freeze()
        assert frozen.params == []
        np.testing.assert_array_equal(frozen([1, 2, 3]), [2, 4, 6])

    def test_freeze_captures_current_value(self):
        a = Param('a')(5)
        m = a * 2
        frozen = m.freeze()
        a.value = 10
        assert m() == 20
        assert frozen() == 10  # Still uses old value


# =============================================================================
# simplify()
# =============================================================================

class TestSimplify:
    """Algebraic simplification."""

    def test_add_zero_left(self):
        a = Param('a')
        m = (0 + a).simplify()
        # Should simplify to just a
        assert m() == a.value

    def test_add_zero_right(self):
        a = Param('a')
        m = (a + 0).simplify()
        assert m() == a.value

    def test_mul_one_left(self):
        a = Param('a')(5)
        m = (1 * a).simplify()
        assert m() == 5

    def test_mul_one_right(self):
        a = Param('a')(5)
        m = (a * 1).simplify()
        assert m() == 5

    def test_mul_zero(self):
        a = Param('a')(5)
        m = (a * 0).simplify()
        assert m == 0

    def test_sub_zero(self):
        a = Param('a')(5)
        m = (a - 0).simplify()
        assert m() == 5

    def test_div_one(self):
        a = Param('a')(5)
        m = (a / 1).simplify()
        assert m() == 5

    def test_pow_zero(self):
        a = Param('a')(5)
        m = (a ** 0).simplify()
        assert m == 1

    def test_pow_one(self):
        a = Param('a')(5)
        m = (a ** 1).simplify()
        assert m() == 5


# =============================================================================
# Reduction
# =============================================================================

class TestReduction:
    """Reduction (summation over index)."""

    def test_reduction_sum(self):
        i = index(5)
        arr = const(np.array([1, 2, 3, 4, 5]))
        elem = _indecise(arr, i)
        r = Reduction(elem, i, operator.add)
        assert r() == 15

    def test_reduction_params(self):
        a = Param('a')
        i = index(3)
        m = a * _indecise(const(np.array([1, 2, 3])), i)
        r = Reduction(m, i)
        assert a in r.params

    def test_reduction_free_excludes_index(self):
        i = index(3)
        arr = const(np.array([1, 2, 3]))
        r = Reduction(_indecise(arr, i), i)
        assert i not in r.free

    def test_reduction_with_input(self):
        i = index(3)
        weights = const(np.array([1, 1, 1]))
        w = _indecise(weights, i)
        # Sum of (INPUT * weight) for each index
        m = w * INPUT
        r = Reduction(m, i)
        assert r(2) == 6  # 2*1 + 2*1 + 2*1


# =============================================================================
# _indecise
# =============================================================================

class TestIndecise:
    """Index selection with _indecise."""

    def test__indecise_array(self):
        arr = const(np.array([10, 20, 30]))
        i = index(3)
        m = _indecise(arr, i)
        # Evaluate at index 1
        result = m % {i: 1}
        assert result() == 20

    def test__indecise_default_index(self):
        arr = const(np.array([10, 20, 30]))
        m = _indecise(arr)  # Uses default INDEX
        result = m[1]  # Evaluates immediately when no INPUT
        assert result == 20

    def test__indecise_with_input(self):
        # Index into INPUT array
        m = _indecise(INPUT)
        result = m[0]
        arr = np.array([5, 10, 15])
        assert result(arr) == 5

    def test__indecise_2d_matrix(self):
        # Multidimensional indexing via chained _indecise
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        i = index(3)
        j = index(3)
        # A[i][j]
        elem = _indecise(_indecise(const(A), i), j)
        result = elem % {i: 1, j: 2}
        assert result() == 6  # A[1][2]

    def test__indecise_2d_reduction(self):
        # Sum all elements of 2D matrix
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        i = index(3)
        j = index(3)
        elem = _indecise(_indecise(const(A), i), j)
        sum_j = Reduction(elem, j, operator.add)
        sum_ij = Reduction(sum_j, i, operator.add)
        assert sum_ij() == 45


class TestIndexingBehavior:
    """Test interaction between __getitem__ and different index types."""

    def test_getitem_with_global_INDEX(self):
        # m[1] works with global INDEX
        arr = const(np.array([10, 20, 30]))
        m = _indecise(arr, INDEX)  # Uses global INDEX
        assert m[1] == 20

    def test_getitem_with_custom_index_requires_dict(self):
        # m[1] with custom index needs dict syntax
        arr = const(np.array([10, 20, 30]))
        i = index(3)
        m = _indecise(arr, i)
        # Dict syntax works
        result = m % {i: 1}
        assert result() == 20

    def test_getitem_shorthand_only_for_global_INDEX(self):
        # m[1] shorthand only substitutes global INDEX, not custom indices
        arr = const(np.array([10, 20, 30]))
        i = index(3)
        m = _indecise(arr, i)
        # m[1] tries to substitute INDEX (not i), so fails
        with pytest.raises(KeyError):
            m[1]

    def test_getitem_with_INPUT_and_global_INDEX(self):
        # When INPUT present, m[1] returns substituted model
        m = _indecise(INPUT, INDEX)
        result = m[1]  # Substitutes INDEX, returns model
        assert result([5, 10, 15]) == 10

    def test_getitem_with_INPUT_and_custom_index(self):
        # Custom index with INPUT needs dict
        i = index(3)
        m = _indecise(INPUT, i)
        result = m % {i: 1}  # Dict syntax
        assert result([5, 10, 15]) == 10


# =============================================================================
# Gradient
# =============================================================================

class TestGradient:
    """Automatic differentiation."""

    def test_grad_linear(self):
        a = Param('a')(2)
        m = a * INPUT  # y = a*x, dy/da = x
        grad = m.grad(a)
        # Gradient should be INPUT
        np.testing.assert_array_equal(grad([1, 2, 3]), [1, 2, 3])

    def test_grad_quadratic(self):
        a = Param('a')(3)
        m = a * a  # y = a^2, dy/da = 2a
        grad = m.grad(a)
        assert grad() == 6

    def test_grad_sum(self):
        a = Param('a')(2)
        b = Param('b')(3)
        m = a + b
        assert m.grad(a)() == 1
        assert m.grad(b)() == 1

    def test_grad_product(self):
        a = Param('a')(2)
        b = Param('b')(3)
        m = a * b  # d/da = b, d/db = a
        assert m.grad(a)() == 3
        assert m.grad(b)() == 2

    def test_grad_division(self):
        a = Param('a')(4)
        b = Param('b')(2)
        m = a / b  # d/da = 1/b = 0.5
        assert m.grad(a)() == 0.5

    def test_grad_power(self):
        a = Param('a')(2)
        m = a ** 3  # d/da = 3*a^2 = 12
        assert m.grad(a)() == 12

    def test_grad_vector(self):
        a = Param('a')(2)
        b = Param('b')(3)
        m = a * b
        grad = m.grad()  # Gradient w.r.t. all params
        result = grad()
        np.testing.assert_array_equal(result, [3, 2])  # [d/da, d/db]

    def test_grad_chain_rule(self):
        a = Param('a')(2)
        m = (a + 1) ** 2  # d/da = 2*(a+1) = 6
        assert m.grad(a)() == 6

    def test_grad_exp(self):
        import numpy as np
        a = Param('a')(0)
        m = np.exp(a)  # d/da = exp(a) = 1
        assert abs(m.grad(a)() - 1.0) < 1e-10

    def test_grad_nested(self):
        a = Param('a')(1)
        b = Param('b')(2)
        m = (a + b) * (a - b)  # a^2 - b^2, d/da = 2a = 2
        assert m.grad(a)() == 2


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and potential failure modes."""

    def test_deeply_nested(self):
        a = Param('a')(2)
        m = a
        for _ in range(50):
            m = m + 1
        assert m() == 52

    def test_many_params(self):
        params = [Param(f'p{i}')(i) for i in range(20)]
        m = params[0]
        for p in params[1:]:
            m = m + p
        assert m() == sum(range(20))

    def test_zero_division_model(self):
        a = Param('a')(0)
        m = 1 / a
        with pytest.raises(ZeroDivisionError):
            m()

    def test_negative_power(self):
        a = Param('a')(2)
        m = a ** -1
        assert m() == 0.5

    def test_float_power(self):
        a = Param('a')(4)
        m = a ** 0.5
        assert m() == 2.0

    def test_large_array(self):
        m = INPUT * 2
        arr = np.arange(10000)
        result = m(arr)
        np.testing.assert_array_equal(result, arr * 2)

    def test_nan_propagation(self):
        a = Param('a')(np.nan)
        m = a * 2
        assert np.isnan(m())

    def test_inf_arithmetic(self):
        a = Param('a')(np.inf)
        m = a * 2
        assert np.isinf(m())

    def test_model_with_numpy_ufunc(self):
        m = Model(np.sin, [INPUT])
        result = m(np.array([0, np.pi/2, np.pi]))
        np.testing.assert_array_almost_equal(result, [0, 1, 0])

    def test_empty_params_model(self):
        m = const(5) + const(3)
        assert m.params == []
        assert m() == 8


# =============================================================================
# Representation
# =============================================================================

class TestRepresentation:
    """String representation of models."""

    def test_repr_param(self):
        a = Param('a')(5)
        m = a * 2
        s = repr(m)
        assert 'a' in s or '5' in s

    def test_repr_input(self):
        m = INPUT * 2
        s = repr(m)
        assert 'INPUT' in s

    def test_repr_addition(self):
        a = Param('a')
        m = a + 1
        s = repr(m)
        assert '+' in s

    def test_repr_nested(self):
        a = Param('a')
        b = Param('b')
        m = (a + b) * 2
        s = repr(m)
        # Should have both + and *
        assert '+' in s or '*' in s


# =============================================================================
# Compilation Consistency
# =============================================================================

class TestCompilation:
    """Compiled vs non-compiled consistency."""

    def test_compiled_matches_uncompiled(self):
        a = Param('a')(2)
        b = Param('b')(3)
        m = a * INPUT + b
        x = np.array([1.0, 2.0, 3.0])

        # Uncompiled result
        uncompiled = m(x).copy()

        # Compile and run
        m.compile()
        compiled = m(x)

        np.testing.assert_array_almost_equal(uncompiled, compiled)

    def test_compilation_caching_same_param(self):
        a = Param('a')
        m1 = a * 2 + 1
        m2 = a * 2 + 1  # Same structure, same param

        m1.compile()
        # m2 should use cached compiled function
        assert m2.compiled

    def test_compilation_caching_different_params(self):
        # Different Param objects but same structure should share compiled code
        a1 = Param('a')(2)
        a2 = Param('a')(3)
        m1 = a1 * 2 + 1
        m2 = a2 * 2 + 1

        assert hash(m1) == hash(m2)  # Same structure = same hash
        m1.compile()
        assert m2.compiled  # Shares cached function

        # But they give correct different results
        assert m1() == 5  # 2*2+1
        assert m2() == 7  # 3*2+1


# =============================================================================
# Comparison Operators
# =============================================================================

class TestComparisons:
    """Comparison operators create Models."""

    def test_less_than(self):
        a = Param('a')(3)
        m = a < 5
        assert m() == True

    def test_greater_than(self):
        a = Param('a')(3)
        m = a > 5
        assert m() == False

    def test_less_equal(self):
        a = Param('a')(5)
        m = a <= 5
        assert m() == True

    def test_greater_equal(self):
        a = Param('a')(5)
        m = a >= 5
        assert m() == True

    def test_not_equal(self):
        a = Param('a')(3)
        m = a != 5
        assert m() == True

    def test_comparison_with_input(self):
        m = INPUT > 0
        result = m(np.array([-1, 0, 1]))
        np.testing.assert_array_equal(result, [False, False, True])
