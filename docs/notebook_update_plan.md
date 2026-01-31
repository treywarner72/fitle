# Notebook Update Plan

## Overview
Update `notebooks_remote/` notebooks to use the latest fitle API and improve documentation.

## Notebooks to Update
1. `real_data_kkp.ipynb` - KKP channel fits (largest, ~40 cells)
2. `real_data_kpp.ipynb` - KPP channel fits (~25 cells)
3. `fitle.ipynb` - Installation helper (trivial)
4. `binning_schemes.ipynb` - Review for fitle usage
5. `monte_carlo.ipynb` - Review for fitle usage
6. `monte_carlo_analysis.ipynb` - Review for fitle usage

## Changes to Apply

### 1. Import Style
**Before:**
```python
import fitle as fl
```

**After:**
```python
from fitle import (
    Param, INPUT, fit, Cost,
    gaussian, exponential, crystalball, convolve, identity
)
```

### 2. Param Syntax Modernization
Use auto-naming with shorthand operators where possible.

**Before:**
```python
fl.Param(1869)('mu1')
fl.Param.positive(5)
fl.Param(3.6, min=0.5, max=5)('alpha1')
fl.Param.positive('N_background')(6e4)
```

**After:**
```python
mu1 = Param('mu1')(1869)
sigma = (+Param)(5)  # auto-named 'sigma'
alpha1 = Param('alpha1')(0.5, 5)(3.6)
N_background = (+Param)(6e4)  # auto-named 'N_background'
```

### 3. Exponential with start/end
**Before:**
```python
fl.exponential(tau=fl.Param.positive(100)) % (fl.INPUT - 1840)
```

**After:**
```python
exponential(tau=(+Param)(100), start=1840, end=2040)
```

### 4. Convolve with bin_width
Calculate bin_width from the data range and number of bins.

**Before:**
```python
fl.convolve(Dp_x, Dp_c, Dp_mother_mass, conv1_mass1, fl.Param.positive(5))
```

**After:**
```python
# For 200 bins over [1840, 2040]:
bin_width = (2040 - 1840) / 200  # = 1.0

convolve(Dp_x, Dp_c, Dp_mother_mass, mass1, (+Param)(5), bin_width=bin_width)
```

### 5. Remove fl. Prefix Throughout
Replace all `fl.` prefixes with direct references.

### 6. Delete Unused Custom Functions
Remove the custom `convolve` function in `real_data_kkp.ipynb` (cell f9b2e328) since built-in is available.

### 7. Add Markdown Explanations
Convert raw cells to markdown and add explanatory text:
- Before each model definition: explain what's being modeled
- Before each fit: explain the cost function choice
- After each fit result: interpret the results
- For complex models: explain the components

### 8. Ensure All Params Are Named
Every parameter should have an explicit name or use auto-naming correctly:
```python
# Good - explicit name:
mu = Param('mu')(1869)

# Good - auto-named from variable:
sigma = +Param  # auto-named 'sigma'
N_signal = (+Param)(1e6)  # auto-named 'N_signal'

# Bad - anonymous:
fl.Param.positive(5)  # no name!
```

## File-by-File Tasks

### real_data_kkp.ipynb
- [ ] Update imports (cell 1)
- [ ] Update `model_sum` helper function
- [ ] Delete custom `convolve` function (cell f9b2e328)
- [ ] Update double gaussian model (mu1, mu2, dp, ds, tail)
- [ ] Update conv1 model with proper param names and bin_width
- [ ] Update conv2 model (varying tau exponential)
- [ ] Update conv3 model (triple gaussian convolution)
- [ ] Update conv4 model
- [ ] Update crys1 model (crystal ball fits)
- [ ] Update crys2 model
- [ ] Add markdown explanations throughout
- [ ] Verify all params are named

### real_data_kpp.ipynb
- [ ] Update imports
- [ ] Update `model_sum` helper
- [ ] Update conv_kkp model
- [ ] Update conv1, conv2 models
- [ ] Update crys1, crys2 models
- [ ] Add markdown explanations
- [ ] Verify all params are named

### binning_schemes.ipynb
- [ ] Update imports (uses `import fitle as fl`)
- [ ] Modernize any fitle usage
- [ ] Add markdown explanations

### monte_carlo.ipynb & monte_carlo_analysis.ipynb
- [ ] Check for fitle usage and update if present

## Execution Order
1. Start with `real_data_kkp.ipynb` (most complex, establishes patterns)
2. Apply same patterns to `real_data_kpp.ipynb`
3. Review and update remaining notebooks

## Notes
- Keep `model_sum` helper function but modernize its internals
- The bin_width for chi2 with 200 bins over [1840, 2040] is exactly 1.0
- For KPP data range [1790, 1970], bin_width = (1970-1790)/200 = 0.9
- Auto-naming requires variable assignment on same line as Param creation
