# fitle

A fast library for building statistical models, compiling them to efficient [Numba] code, and fitting to data with [iminuit].

> - Compose models from pieces (`Param`, arithmetic, `numpy` ops).
> - Tie it to data with cost functions (MSE, (un)binned NLL, chi-squared).
> - Fit with `iminuit`.

---

## Install

```bash
pip install -e .
# or
pip3 install git+https://github.com/treywarner72/fitle.git
```

### Dependencies

- Python 3.10+
- `numpy`
- `numba`
- `iminuit`

---

## Quick Start

```python
import numpy as np
import fitle as fl

# Generate sample data
data = np.random.normal(1850, 10, 1000)

# Define parameters with bounds and starting values
mu    = fl.Param('mu')(1800, 2000)(1870)   # name, (min,max), start
sigma = fl.Param.positive('sigma')(5)       # >0 constraint, start=5

# Build a Gaussian PDF model
model = fl.gaussian(mu, sigma)

# Fit using unbinned negative log-likelihood
result = fl.fit(model | fl.Cost.NLL(data))
print(result)
# <FitResult fval=3713.952, success=True>
# mu: 1850 +/- 0.44
# sigma: 9.925 +/- 0.31

# Evaluate the fitted model
model([1851, 1852, 1853, 1854])
```

---

## Building Blocks

### Parameters

```python
from fitle import Param, INPUT, index, INDEX

# Free fit parameter (THETA) with bounds and starting value
a = Param()                           # unbounded, start=0
b = Param('name')                     # with name
c = Param(5.0)                        # with starting value
d = Param('name')(0, 10)(5.0)         # name, bounds, start (chainable)

# Factory helpers for common constraints
tau = Param.positive()                # >0, start=1
f   = Param.unit()                    # [0,1], start=0.5

# The independent variable (INPUT)
linear = a * INPUT + b

# Integer index for reductions
i = index(10)                         # range(0, 10)
j = index(5, 20)                      # range(5, 20)
k = index(0, 100, 2)                  # range(0, 100, 2)
```

### Models & Composition

```python
from fitle import gaussian, exponential, Param, INPUT

# Pre-built PDFs
g1 = gaussian(Param('mu1')(1870), Param.positive('s1')(5))
g2 = gaussian(Param('mu2')(1970), Param.positive('s2')(10))
exp_tail = exponential(Param.positive('tau')(100))

# Composite model with normalization parameters
composite = (
    Param.positive('n1')(5e5) * g1 +
    Param.positive('n2')(5e5) * g2 +
    Param.positive('n_bg')(6e4) * exp_tail
)

# Arithmetic with INPUT
polynomial = Param('a') * INPUT**2 + Param('b') * INPUT + Param('c')
```

---

## Cost Functions

Cost functions transform a model into a scalar loss for optimization.

```python
from fitle import Cost

# Mean Squared Error for regression
loss = model | Cost.MSE(x_data, y_data)

# Unbinned Negative Log-Likelihood (for PDFs)
loss = model | Cost.NLL(data)
loss = model | Cost.unbinnedNLL(data)  # same as above

# Binned Chi-squared
loss = model | Cost.chi2(data=sample, bins=100, range=(0, 10))
loss = model | Cost.chi2(x=centers, y=counts, bin_widths=widths)

# Binned Negative Log-Likelihood
loss = model | Cost.binnedNLL(data=sample, bins=100, range=(0, 10))
loss = model | Cost.binnedNLL(x=centers, y=counts, bin_widths=widths)
```

All cost functions return a `Model` that evaluates to a scalar.

---

## Fitting

```python
from fitle import fit

result = fit(loss)

print(result.fval)      # minimum cost value
print(result.success)   # True if converged
print(result.values)    # {'param_name': value, ...}
print(result.errors)    # {'param_name': error, ...}

# Access the frozen prediction model
y_pred = result.predict(x_new)

# Plot for binned fits
result.plot_data()      # data with error bars
result.plot_fit()       # fitted curve
```

### Fit Options

```python
# Disable Numba compilation (use pure Python)
result = fit(loss, numba=False)

# Disable analytical gradients (use numerical)
result = fit(loss, grad=False)

# Pass options to iminuit
result = fit(loss, options={'print_level': 2})
```

---

## Reductions & Indexing

Reductions sum expressions over an integer index.

```python
from fitle import index, indecise, const, Reduction, gaussian, INPUT

# Create an index over range(N)
i = index(10)

# indecise picks element at index from an array
weights = indecise(const([0.1, 0.2, 0.3, ...]), i)
centers = indecise(const([1.0, 2.0, 3.0, ...]), i)

# Build expression and reduce
term = weights * gaussian(centers, sigma)
total = Reduction(term, i)  # sum over i
```

### Example: Discrete Convolution

```python
def convolve_hist(centers, counts, mu, sigma):
    i = index(len(counts))
    w = indecise(const(counts), i)
    c = indecise(const(centers), i)
    g = gaussian(c, sigma)
    return Reduction(w * g, i)

model = convolve_hist(hist_centers, hist_counts, Param('mu'), Param.positive('sigma'))
```

---

## Compilation

Models are automatically compiled when fitting. You can also compile manually:

```python
model.compile()           # JIT compile with Numba
print(model.code)         # inspect generated source
print(model.compiled)     # True if compiled

# Cache management
Model.cache_info()        # {'size': N, 'limit': 128}
Model.clear_cache()       # free memory
```

---

## Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Composite model
model = signal_1 + signal_2 + background
result = fit(model | Cost.NLL(data))

# Plot components separately (they share fitted parameters)
x = np.linspace(0, 100, 200)
plt.plot(x, model(x), label='Total')
plt.plot(x, signal_1(x), label='Signal 1')
plt.plot(x, signal_2(x), label='Signal 2')
plt.plot(x, background(x), label='Background')
plt.legend()
```

---

## Tips

- **Substitution** with `%`:
  ```python
  model % x_array              # substitute INPUT with data
  model % {INPUT: x_array}     # explicit form
  model % {param: 5.0}         # fix a parameter to a value
  model % {param: other_model} # replace with another model
  ```

- **Chi-squared with zero bins**: Use `zero_method='absolute'` to handle bins with zero counts.

- **Gradients**: If symbolic differentiation fails, fit automatically falls back to numerical gradients.

- **Freezing**: Use `model.freeze()` to replace all parameters with their current values.

---

## Full API Reference

### Module: `fitle.param`

#### `class Param`

Symbolic parameters for model building.

```python
Param(arg0=None, arg1=None, min=None, start=None, max=None, name=None, *, range=None, kind=ParamKind.THETA)
```

**Attributes:**
- `kind` - Parameter role: `THETA` (optimizable), `INPUT` (data), or `INDEX` (loop variable)
- `name` - Optional string identifier
- `min`, `max` - Bounds for optimization (THETA only)
- `start` - Initial value for optimization
- `value` - Current value (updated during fitting)
- `error` - Uncertainty estimate (set after fitting)
- `range` - Loop range (INDEX only)

**Methods:**
- `__call__(arg0, arg1=None)` - Chainable setter: string sets name, number sets start, two numbers set bounds. Returns `self`.

**Class Methods:**
- `Param.positive(name=None)` - Factory for positive parameters (min=1e-6, start=1)
- `Param.unit(name=None)` - Factory for [0,1] parameters (start=0.5)

#### `INPUT`

Singleton `Param` representing the independent variable (x-data).

#### `index(*args) -> Param`

Create an INDEX parameter with `range(*args)`.

```python
i = index(10)        # range(0, 10)
j = index(5, 20)     # range(5, 20)
k = index(0, 10, 2)  # range(0, 10, 2)
```

#### `INDEX`

Default INDEX parameter (unbound range).

---

### Module: `fitle.model`

#### `class Model`

Represents a symbolic expression tree.

```python
Model(fn, args)
```

**Attributes:**
- `fn` - The function this node applies
- `args` - List of arguments (may include Param, Model, constants)
- `params` - List of all THETA parameters in the tree
- `free` - List of INPUT and INDEX parameters
- `compiled` - True if a compiled version exists in cache
- `code` - Generated source code (after compilation)
- `memory` - Dict for metadata (used by Cost)
- `components` - List of sub-models (split at `identity()` boundaries)

**Methods:**

| Method | Description |
|--------|-------------|
| `__call__(x=None)` | Evaluate the model or perform substitution |
| `__mod__(subs)` | Substitute parameters/INPUT with new values |
| `__getitem__(index)` | Evaluate or substitute at given index |
| `copy()` | Deep copy with independent THETA params |
| `copy_with_args(args)` | Shallow copy with new arguments |
| `freeze()` | Replace all params with current values |
| `grad(wrt=None)` | Symbolic gradient w.r.t. parameters |
| `simplify()` | Apply algebraic simplifications |
| `shape()` | Infer output shape |
| `compile()` | JIT compile with Numba |
| `eval(x, index_map, numba=False)` | Low-level evaluation |
| `eval_py(x, index_map)` | Pure Python evaluation |

**Arithmetic:** Models support `+`, `-`, `*`, `/`, `**`, unary `-`, and comparisons (`<`, `<=`, `>`, `>=`, `!=`).

**Class Methods:**
- `Model.clear_cache()` - Clear the compilation cache
- `Model.cache_info()` - Return cache size and limit

#### `class Reduction(Model)`

Reduce an expression over an INDEX parameter.

```python
Reduction(model, index_param, op=operator.add)
```

Currently only `operator.add` is supported for compilation.

#### `const(val) -> Model`

Wrap a constant value as a Model node.

```python
c = const(5.0)
arr = const(np.array([1, 2, 3]))
```

#### `indecise(obj, index=INDEX) -> Model`

Create a Model that indexes into an array.

```python
weights = indecise(const([0.1, 0.2, 0.3]), i)
# Evaluates to obj[int(index)]
```

#### `identity(val) -> Model`

Wrap a value in an identity node (useful for component boundaries).

#### `vector(*models) -> Model`

Stack multiple models into an array output.

```python
gradients = vector(model.grad(p1), model.grad(p2))
```

---

### Module: `fitle.cost`

#### `class Cost`

Factory for cost functions.

**Class Methods:**

| Method | Description |
|--------|-------------|
| `Cost.MSE(x, y)` | Mean Squared Error: `sum((y - model(x))^2)` |
| `Cost.NLL(x)` | Unbinned Negative Log-Likelihood: `sum(-log(model(x)))` |
| `Cost.unbinnedNLL(x)` | Alias for `NLL` |
| `Cost.chi2(...)` | Chi-squared for binned data |
| `Cost.binnedNLL(...)` | Binned Negative Log-Likelihood |

**Chi-squared / Binned NLL signatures:**

```python
# From raw data
Cost.chi2(data=array, bins=int, range=(min, max))
Cost.chi2(data=array, bins=edges_array)

# From histogram
Cost.chi2(x=centers, y=counts, bin_widths=widths)

# Chi2 options
Cost.chi2(..., zero_method='error')     # raise on zero counts (default)
Cost.chi2(..., zero_method='absolute')  # use absolute residuals for zeros
```

**Usage:** Pipe a model into a cost with `|`:

```python
loss = model | Cost.NLL(data)
```

---

### Module: `fitle.fitting`

#### `fit(model, numba=True, grad=True, ncall=9999999, options=None) -> FitResult`

Minimize a cost model using iminuit's MIGRAD.

**Parameters:**
- `model` - Cost model (created with `model | Cost.xxx(...)`)
- `numba` - Use Numba compilation (default: True)
- `grad` - Use analytical gradients (default: True)
- `ncall` - Maximum function calls
- `options` - Additional kwargs for `iminuit.Minuit`

#### `class FitResult`

Container for fit results.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `Model` | The cost model that was fitted |
| `fval` | `float` | Minimum cost value |
| `success` | `bool` | True if fit converged |
| `values` | `dict[str, float]` | Fitted parameter values |
| `errors` | `dict[str, float]` | Parameter uncertainties |
| `predict` | `Model` | Frozen base model for predictions |
| `x` | `ndarray` | Input data (bin centers) |
| `y` | `ndarray | None` | Output data (counts), None for unbinned |
| `minimizer` | `Minuit` | Underlying iminuit object |

**Methods:**

| Method | Description |
|--------|-------------|
| `plot_data()` | Plot data with Poisson error bars (binned only) |
| `plot_fit()` | Plot fitted model curve (binned only) |
| `dof()` | Degrees of freedom: n_bins - n_params (binned only) |

---

### Module: `fitle.pdfs`

Pre-built probability density functions.

#### `gaussian(mu=None, sigma=None) -> Model`

Normalized Gaussian PDF.

```python
# Default parameters (auto-created)
g = gaussian()

# Custom parameters
g = gaussian(Param('mean')(5.0), Param.positive('width')(1.0))
```

#### `exponential(tau=None) -> Model`

Normalized exponential PDF: `(1/tau) * exp(-x/tau)`.

```python
e = exponential()
e = exponential(Param.positive('lifetime')(2.5))
```

#### `crystalball(alpha, n, mu, sigma) -> Model`

Crystal Ball function (Gaussian core + power-law tail).

**Parameters:**
- `alpha` - Transition point in sigma units (typical: 0.5-5)
- `n` - Tail power (typical: 1.5-10, must be >1)
- `mu` - Mean
- `sigma` - Width

#### `convolve(d_x, c, mass_mother, mu, sigma, idx=None) -> Model`

Discrete convolution of Gaussians with weights.

**Parameters:**
- `d_x` - Center positions for each component
- `c` - Weights for each component
- `mass_mother` - Reference mass for shift
- `mu` - Mean offset parameter
- `sigma` - Common width for all components
- `idx` - Optional INDEX parameter

---

### NumPy Integration

NumPy functions work directly with Model and Param objects via `__array_ufunc__` and `__array_function__` protocols:

```python
import numpy as np
from fitle import INPUT as x, Param

# NumPy functions create Model nodes automatically
model = np.exp(-x**2 / 2)
model = np.where(x > 0, x, -x)
model = np.sum(weights * values)

# Works with Params too
mu = Param('mu')
sigma = Param.positive('sigma')
gaussian = np.exp(-0.5 * ((x - mu) / sigma)**2)
```

**Note:** Keyword arguments (e.g., `np.mean(x, where=...)`) are not supported for compilation. Use positional arguments or equivalent expressions.

---

### Module: `fitle.compiler`

#### `class Compiler`

Compiles Model trees to Numba JIT functions.

```python
c = Compiler(model)
fn = c.compile()      # Returns @njit function
print(c.code)         # Generated source code
```

The compilation process:
1. **DAG Construction** - Build IR with common subexpression detection
2. **Node Partitioning** - Separate global vs loop-local expressions
3. **Temporary Allocation** - Assign names to multi-use expressions
4. **Source Emission** - Generate Python code for `@njit`

---

## License

MIT

---

## Acknowledgements

Built on [NumPy], [Numba], and [iminuit].

[NumPy]: https://numpy.org/
[Numba]: https://numba.pydata.org/
[iminuit]: https://iminuit.readthedocs.io/
