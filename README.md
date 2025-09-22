# fitle

A tiny, fast library for building statistical models, compiling them to efficient [Numba] code, and fitting to data with [iminuit].

> Core ideas:
> - Compose models from simple pieces (`Param`, arithmetic, `numpy` ops).
> - Contextualize it to data with built-in cost factories (MSE, (un)binned NLL, χ²).
> - Generate optimized numba code.
> - Fit with `iminuit`.

---

## Install

This is currently a local/package source library.

```bash
# From the repository root (where setup.py/pyproject would live if you add one)
# Editable install (recommended while developing)
pip install -e .
```

### Dependencies

- Python 3.10+
- `numpy`
- `numba`
- `iminuit`
- (optional) `matplotlib` for plotting in the examples

---

## Quick start


```py
import numpy as np
import fitle as fl

data = np.random.normal(1850, 10, 1000) # random data, mu=1850, sigma=10, 1000 entries

# A Gaussian PDF model in X with fit parameters mu, sigma
mu    = fl.Param('mu')(1800, 2000)(1870)   # name, (min,max), start
sigma = fl.Param.positive('sigma')(5)  # >0 constraint, start=5

model = fl.gaussian(mu, sigma)         # returns a Model

cost = fl.Cost.NLL(data)               # factory
loss = model | cost                    # pipe the model into the cost

# Fit with iminuit (uses Numba-compiled model and gradients by default)
result = fl.fit(loss)       # -> FitResult
print(result)
# <FitResult fval=3713.952, success=True>
# sigma: 9.925 ± 0.31
# mu: 1850 ± 0.44

# Evaluate the best-fit prediction quickly
result.predict([1851,1852,1853,1854])        
# or, as the values of the parameters adjust to the best fit
model([1851,1852,1853,1854]) # which is usually convenient
```

---

## Building blocks

### Parameters

```py
# Free fit parameter (THETA) with bounds and starting value
a = Param()
b = Param('add a name')
c = Param(5) # or a starting value
d = Param(0)('name')(-5,5) # in any order, or with a bound too
tau = Param.positive() # or use a default bound
f = Param.unit() # this one is (0,1)

# The independent variable
linear_model = a * INPUT + b

# Integer index for reductions
i = index(10)   # i over range(0, 10)
```

### Models & composition

```py
g1 = gaussian(Param('mu1')(1870), Param.positive('s1')(5))
g2 = gaussian(Param('mu2')(1970), Param.positive('s2')(10))
tail = (1.0 / tau) * np.e ** (-(INPUT - 1840) / tau)

composite = Param.positive('n_gauss_1')(5e5)*g1 + Param.positive('n_gauss_2')(5e5)*g2 + Q(6e4)*Param.positive('n_tail')
```

---

## Cost functions

Cost functions are easy to define with the framework, for instance:

```py
data_x, data_y = [...], [...]

cost1 = linear_model % {INPUT : data_x} # % replaces elements in the model according to a dictionary
cost1() # returns an array of the prediction for each data_x value
cost2 = (data_y - cost1) ** 2 # represents the error squared
cost = mnp.sum(cost2) # represents the sum of squared errors

fit(cost) # minimizes the sum of squared errors with respect to a and b
```

There are of couse built in functions to do this. Let `model` be a placeholder model to be fitted.

```py
# Mean squared error on paired (x, y)
loss = model | Cost.MSE(x, y)

# Unbinned negative log-likelihood on a sample x (no y)
loss = model | Cost.NLL(x)

# Binned χ² and binned NLL
#   Either supply raw data with binning instructions:
chi2_loss   = model | Cost.chi2(data=sample, bins=200, range=(1840, 2040))
binned_loss = model | Cost.binnedNLL(data=sample, bins=200, range=(1840, 2040))

#   …or (x, y) as bin centers and counts directly:
chi2_loss2   = model | Cost.chi2(x=centers, y=counts)
binned_loss2 = model | Cost.binnedNLL(x=centers, y=counts)
```

All costs return a `Model` that **evaluates to a scalar**. Pipe your baseline model with `|` into the cost factory.

---

## Fitting

```py
# Compile and minimize
res = fit(loss)

print(res.fval, res.success)
print(res.values)  # dict: {param_name: value}
print(res.errors)  # dict: {param_name: error}
```

**Notes**
- `fit()` uses:
  - `model.compile()` (Numba JIT). Set `numba=False` to use base Python.
  - `model.grad()`. Set `grad=False` to not let the program differentiate and use that gradient.
- Parameter limits are taken from each `Param`’s `(min, max)`. Starts from `Param.start`.

---

## Speed

There is no need to define all the steps. Here is an example:

```py
#define model
model = Param.postiive('n_signal') * gaussian() + Param.positive('n_background') * (Param('a')*INPUT + Param(B))

# fit the model with respect to the data
res = fit(model | Cost.NLL(data))

plt.hist(data)
plt.plot(x, model(x))
plt.text(f"signal count: {res.values['n_signal']}")

# or
n_signal = Param.postiive('n_signal')
model = n_signal * ...
fit(model | Cost.NLL(data))
...
plt.text(f"signal count: {n_signal.value}")
```
---

## Reductions & indexing

Reductions let you do operations over an integer `index`.

- Create an index: `i = index(N)` (→ `range(0, N)`).
- Build a per-bin expression that depends on `i`.
- Wrap with `Reduction(expr, i, operation=operator.add)

Helper: **`indecise(container, index)`** – represents the picking an element at integer `index` from a Python list/tuple or a NumPy array, as a `Model`.

### Example: discrete convolution of MC histogram with a Gaussian

```py

# assume we have a histogram of a distribution that we want to convolve with the gaussian.
# this is in the context of fitting a particle mass distribution using a convolutional approach

def convolve(i, centers, counts, mother_mass, mu, sigma):
    w    = indecise(const(counts), i)         # weight per bin
    c    = indecise(const(centers), i)        # bin center
    xsh  = INPUT + mother_mass - mu           # shift into MC frame
    g    = gaussian(c, sigma) % {INPUT: xsh}  # Gaussian at center c
    term = w * g
    return Reduction(term, i)                 # Σ_i w_i * g(x; c_i)

# Example usage
i     = index(len(Dp_c))
mass1 = Param('mDp')(1800, 2000)(1870)
sig5  = Param.positive('s5')(5)
sig10 = Param.positive('s10')(10)

signal = Param.postiive(5e5)*convolve(i, histogram_centers, histogram_counts, 1869.65, mass1, sig5) \
       + Param.postiive(5e5)*convolve(i, histogram_centers, histogram_counts, 1869.65, mass1, sig10)

background = Q(4e4) * exponential(tau=Param.positive('tau')(100)) % (INPUT-1840)
model_conv = signal + background

loss = model_conv | Cost.chi2(data=particle_mass_distribution, bins=200, range=(1840, 2040))
result = fit(loss)
```

**Current limitations**
- **Nested indices** are not yet supported (one index per reduction body).

---

## Compilation details

```py
# Compile explicitly (usually done for you inside fit())
model.compile()       # stores generated source on model.code
print(model.code)     # see the code that the program generated, useful for troubleshooting
model(x)              # now uses numba JIT compiled version.
```

The compiler also caches the structure of the model, so if you have a same model with the same structure, you don't need to compile again.

---

## Plotting

An execellent characteristic of this package is the ease of plotting. Let's say we have a model composed of two signals and a background. It's very easy to plot the components seperately.

```py
model = signal_1 + signal_2 + background
fit(model | NLL(data))

x = np.linspace(0,100,100)
plt.plot(x, model(x))
plt.plot(x, signal_1(x))
plt.plot(x, signal_2(x))
plt.plot(x, background(x))
```

The compiler also caches the structure of the model, so if you have a same model with the same structure, you don't need to compile again.

---

## API (brief)

```py
# Params
Param(name=None)(min, max)(start)   # chainable setters
Param.positive(name=None)           # factory with (1e-6, ∞), start=1
Param.unit(name=None)               # factory with [0,1], start=0.5
INPUT                                # placeholder for x
index(*args)                         # INDEX param with range(*args)
INDEX                                 # a default index placeholder

# Model helpers
const(value)                         # wrap constant as Model
indecise(container_or_array, index)  # pick element at index (as Model)
gaussian(mu=None, sigma=None)
exponential(tau=None)

# Model methods
Model.grad(wrt=None)                 # -> Model of gradients
Model.simplify()                     # basic algebraic simplifications
Model.shape(obj)                     # infer output shape
Model.compile()                      # JIT compile; code at .code
Model.freeze()                       # substitute θ with current .value
Model.__mod__(subs)                  # substitute by dict, supports {INPUT: x}
Model.__call__(x=None)               # evaluate, if INPUT is free then x required

# Cost
Cost.MSE(x, y)
Cost.NLL(x)                          # alias: Cost.unbinnedNLL
Cost.chi2(data,bins,range) or Cost.chi2(x=centers, y=counts)
Cost.binnedNLL(…)                    # same forms as chi2

# Fitting
fit(model_or_cost_model, grad=True, ncall=9999999, options={}) -> FitResult

class FitResult:
    .fval          # best objective value
    .success       # iminuit valid flag
    .values        # {name: value}
    .errors        # {name: error}
    .predict       # frozen Model of the *base* (un-costed) model
    .minimizer     # underlying iminuit.Minuit
```

---

## Tips

- `Model.__mod__` does substitution:
  - `{INPUT: x_array}` feeds data,
  - `x_array` is shorthand for above,
  - `{param_obj: value_or_model}` replaces parameters or subgraphs.
- If you pass arrays into `const`, they are hoisted, which is necessary sometimes (I need to fix this).
- χ² requires **non-zero** counts; zeros will raise `ValueError`.

---

## License

MIT

---

## Acknowledgements

- Uses [NumPy] and [Numba] for speed; [iminuit] for minimization.
- Inspired by common HEP fitting workflows.

[NumPy]: https://numpy.org/
[Numba]: https://numba.pydata.org/
[iminuit]: https://iminuit.readthedocs.io/
