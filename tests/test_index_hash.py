# tests/test_index_hash.py
"""INDEX params hash by slot + range, not identity.

Regression: convolve() creates a fresh index() on every call, and the hash
used id(index), so every MC toy recompiled and LLVM eventually ran out of
memory (segfault / "Unable to allocate section memory" after ~400 toys).
"""
import warnings

import numpy as np
import pytest

from fitle import Param, Model, Cost, fit, convolve, exponential, index, const, Reduction
from fitle.model import _indecise


KX = np.linspace(-5, 5, 41)
KC = np.exp(-0.5 * (KX / 2.0) ** 2)


def _conv_cost(data):
    n1, n2, nb = Param.positive("n1")(4e4), Param.positive("n2")(5e4), Param.positive("nb")(3e4)
    mu1, mu2 = Param("mu1")(-15, 15)(0.5), Param("mu2")(85, 115)(100.5)
    s = Param.positive("s")(3.0)
    tau = Param.positive("tau")(80)
    model = (n1 * convolve(KX, KC, 0.0, mu1, s)
             + n2 * convolve(KX, KC, 0.0, mu2, s)
             + nb * exponential(tau, -50, 150))
    return model | Cost.chi2(data, 100, range=(-50, 150))


def _toy(rng, shift=0.0):
    return np.concatenate([
        rng.normal(0 + shift, 4, 40000),
        rng.normal(100 + shift, 4, 50000),
        rng.uniform(-50, 150, 30000),
    ])


@pytest.fixture(autouse=True)
def _clean_cache():
    Model.clear_cache()
    yield
    Model.clear_cache()


def test_fresh_convolve_models_hash_equal():
    k = Param("k")(1.0)
    assert hash(convolve(KX, KC, 0.0, k, 1.0)) == hash(convolve(KX, KC, 0.0, k, 1.0))


def test_index_range_in_hash():
    k = Param("k")(1.0)
    short = convolve(KX[:20], KC[:20], 0.0, k, 1.0)
    full = convolve(KX, KC, 0.0, k, 1.0)
    assert hash(short) != hash(full)


def test_reduction_over_fresh_indices_shares_code():
    arr = const(np.array([1.0, 2.0, 3.0]))
    k = Param("k")(2.0)
    def build():
        i = index(3)
        return Reduction(_indecise(arr, i) * k, i)
    m1, m2 = build(), build()
    assert hash(m1) == hash(m2)
    m1.compile()
    assert m2.compiled
    assert m2() == pytest.approx(12.0)


def test_mc_loop_with_convolve_compiles_once():
    rng = np.random.default_rng(11)
    results = []
    for shift in (0.0, 1.0, -0.5, 0.3):
        data = _toy(rng, shift)
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # no silent numba/grad fallback
            res = fit(_conv_cost(data))
        res_py = fit(_conv_cost(data), numba=False)
        assert res.values["mu1"] == pytest.approx(res_py.values["mu1"], abs=1e-5)
        results.append(res.values["mu1"])
    assert Model.cache_info()["size"] <= 2      # cost model + its gradient
    assert results[1] - results[0] == pytest.approx(1.0, abs=0.15)  # really refit new data
