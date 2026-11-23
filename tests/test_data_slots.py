# tests/test_data_slots.py
"""DATA params: runtime data slots that are passed to compiled code, not baked in."""
import warnings

import numpy as np
import pytest

from fitle import Param, INPUT, Model, Reduction, Cost, fit, gaussian, const, index
from fitle.model import _indecise


def _chi2_model(data, bins=10, rng=(-3, 3)):
    N = Param.positive("N")(len(data))
    mu = Param("mu")(-2, 2)(0.1)
    sigma = Param.positive("sigma")(1.2)
    return (N * gaussian(mu, sigma)) | Cost.chi2(data, bins, range=rng)


@pytest.fixture(autouse=True)
def _clean_cache():
    Model.clear_cache()
    yield
    Model.clear_cache()


class TestCompileReuse:
    """Same model structure + different data -> one compiled function."""

    def test_same_hash_for_different_data(self):
        r = np.random.default_rng(1)
        m1 = _chi2_model(r.normal(0, 1, 50000))
        m2 = _chi2_model(r.normal(0.5, 1, 50000))
        assert hash(m1) == hash(m2)

    def test_one_cache_entry_across_datasets(self):
        r = np.random.default_rng(2)
        for _ in range(5):
            _chi2_model(r.normal(0, 1, 50000)).compile()
        assert Model.cache_info()["size"] == 1

    def test_data_not_baked_into_code(self):
        m = _chi2_model(np.random.default_rng(3).normal(0, 1, 50000))
        m.compile()
        assert "def compiled(x, data, indices, theta):" in m.code
        assert "data[0]" in m.code
        assert "_arr_" not in m.code and "const_" not in m.code

    def test_reused_function_uses_new_data(self):
        r = np.random.default_rng(4)
        m1 = _chi2_model(r.normal(0, 1, 50000))
        m2 = _chi2_model(r.normal(1.0, 1, 50000))
        m1.compile()
        assert m2.compiled                      # cache hit, nothing compiled
        assert m2() == pytest.approx(m2.eval_py(m2.memory['_eval_x'], {}))
        assert m1() != pytest.approx(m2())

    def test_fits_match_python_path(self):
        r = np.random.default_rng(5)
        for true_mu in (-0.5, 0.0, 0.7):
            data = r.normal(true_mu, 1, 50000)
            with warnings.catch_warnings():
                warnings.simplefilter("error")      # no silent numba/grad fallback
                res_nb = fit(_chi2_model(data))
            res_py = fit(_chi2_model(data), numba=False)
            assert res_nb.success and res_py.success
            assert res_nb.values["mu"] == pytest.approx(res_py.values["mu"], abs=1e-6)
            assert res_nb.values["mu"] == pytest.approx(true_mu, abs=0.05)
        assert Model.cache_info()["size"] <= 2   # cost model + its gradient

    def test_binned_nll_reuses(self):
        r = np.random.default_rng(6)
        for _ in range(3):
            data = r.normal(0, 1, 50000)
            N, mu, sigma = Param.positive("N")(5000), Param("mu")(0.1), Param.positive("sigma")(1.1)
            fit(N * gaussian(mu, sigma) | Cost.binnedNLL(data, 20, range=(-4, 4)))
        assert Model.cache_info()["size"] <= 2

    def test_chi2_zero_method_absolute(self):
        data = np.random.default_rng(7).normal(0, 1, 300)
        with pytest.raises(ValueError):   # empty bins: default zero_method refuses
            Cost.chi2(data, 40, range=(-6, 6))
        N, mu, sigma = Param.positive("N")(300), Param("mu")(0.1), Param.positive("sigma")(1.1)
        cm = N * gaussian(mu, sigma) | Cost.chi2(data, 40, range=(-6, 6), zero_method="absolute")
        res = fit(cm)
        assert res.values["mu"] == pytest.approx(fit(cm, numba=False).values["mu"], abs=1e-5)


class TestSlotIdentity:
    """Slots hash by position, so aliasing is distinguished correctly."""

    def test_aliased_vs_distinct_slots(self):
        a = Param.data(np.array([1.0, 2.0]))
        b = Param.data(np.array([3.0, 4.0]))
        same = np.sum(a * a * INPUT)
        diff = np.sum(a * b * INPUT)
        assert hash(same) != hash(diff)
        same.compile()
        x = np.ones(2)
        assert diff(x) == pytest.approx(1 * 3 + 2 * 4)   # not served same's code
        assert same(x) == pytest.approx(1 + 4)

    def test_dtype_and_ndim_in_hash(self):
        f = np.sum(Param.data(np.array([1.0, 2.0])) * INPUT)
        i = np.sum(Param.data(np.array([1, 2])) * INPUT)
        assert hash(f) != hash(i)

    def test_different_lengths_share_code(self):
        k = Param("k")(2.0)
        m3 = np.sum(k * Param.data(np.arange(3.0)))
        m5 = np.sum(k * Param.data(np.arange(5.0)))
        assert hash(m3) == hash(m5)
        m3.compile()
        assert m5() == pytest.approx(2.0 * 10)


class TestPublicDataParam:
    """Param.data(...) as a user-facing building block."""

    def test_call_returns_array_and_is_not_theta(self):
        w = Param.data(np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(w(), [1.0, 2.0, 3.0])
        assert w.name == "w"
        a = Param("a")(1.0)
        m = a * w
        assert m.params == [a]
        assert m.data_params == [w]
        assert m.free == []
        with pytest.raises(TypeError):
            w(5)

    def test_rejects_non_arrays(self):
        with pytest.raises(TypeError):
            Param.data(3.0)

    def test_weighted_custom_cost(self):
        r = np.random.default_rng(8)
        x = np.linspace(0, 10, 200)
        for true_a in (1.5, 2.5):
            y = true_a * x + r.normal(0, 0.3, x.size)
            ydat = Param.data(y)
            wdat = Param.data(np.full(x.size, 2.0))
            a = Param("a")(0, 5)(1)
            cm = np.sum(wdat * (ydat - a * INPUT) ** 2)
            cm.memory["_eval_x"] = x
            cm.compile()
            assert cm() == pytest.approx(cm.eval_py(x, {}))
        assert Model.cache_info()["size"] == 1

    def test_copy_shares_data_slots(self):
        d = Param.data(np.array([1.0, 2.0]))
        m = Param("a")(1.0) * d
        c = m.copy()
        assert c.data_params[0] is d
        assert c.params[0] is not m.params[0]

    def test_str(self):
        d = Param.data(np.zeros(4), "counts")
        assert str(d) == "counts"
        assert "DATA(shape=(4,)" in repr(d)


class TestCompilerPaths:
    """Reductions and sub-model reuse with data slots."""

    def test_reduction_with_data_shaped_accumulator(self):
        i = index(3)
        coef = _indecise(const(np.array([1.0, 2.0, 3.0])), i)
        k = Param("k")(1.0)
        def build(arr):
            return Reduction(coef * k * Param.data(arr), i)
        m_a = build(np.array([1.0, 1.0]))
        m_b = build(np.array([1.0, 2.0, 3.0, 4.0]))
        assert hash(m_a) == hash(m_b)
        m_a.compile()
        assert "data[0].shape" in m_a.code
        np.testing.assert_allclose(m_b(), 6.0 * np.array([1.0, 2.0, 3.0, 4.0]))

    def test_subcompiled_model_with_data(self):
        d1 = Param.data(np.array([1.0, 2.0, 3.0]))
        d2 = Param.data(np.array([10.0, 20.0, 30.0]))
        a = Param("a")(2.0)
        b = Param("b")(3.0)
        inner = np.sum(a * d2)                 # compiled first, then reused
        inner.compile()
        outer = np.sum(b * d1) + inner
        outer.compile()
        assert "_compiled_" in outer.code       # really went through reuse
        assert outer() == pytest.approx(3 * 6 + 2 * 60)
        assert outer() == pytest.approx(outer.eval_py(None, {}))


class TestThetaSlotIdentity:
    """Regression: the cache key used to treat all THETA params as one,
    so a*b was served the compiled code of a*a."""

    def test_aliased_vs_distinct_theta(self):
        a, b = Param("a")(2.0), Param("b")(3.0)
        same = a * a * INPUT
        diff = a * b * INPUT
        assert hash(same) != hash(diff)
        same.compile()
        x = np.array([1.0])
        assert diff(x) == pytest.approx([6.0])
        assert diff(x) == pytest.approx(diff.eval_py(x, {}))

    def test_order_of_use_matters(self):
        a, b = Param("a")(2.0), Param("b")(3.0)
        m1 = a - b * a          # params [a, b]: theta0 - theta1*theta0
        m2 = a - b * b          # params [a, b]: theta0 - theta1*theta1
        assert hash(m1) != hash(m2)
        m1.compile()
        assert m2() == pytest.approx(2.0 - 9.0)

    def test_fresh_params_same_structure_still_share(self):
        def build():
            return Param("a")(1.0) * INPUT + Param("b")(2.0)
        assert hash(build()) == hash(build())
