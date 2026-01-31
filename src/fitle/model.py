"""model.py
=========
Representation of models in the library, also the main user interface
----------------------------------------------------------

Define a model building on built in types or by using arithmetic with Params. For instance:

linear_model = Param('a') * Param.INPUT + Param('b')
gaussian_model = gaussian(Param('mu')(5,10)) # a gaussian model with a mu between 5 and 10 and a default sigma

We can also add models together:

composite_model = Param('n_gaussian') * gaussian_model + linear_model

To run a model, all you need to do is run __call__:

composite_model(0) or composite_model(ndarray)

You can also compile it with composite_model.compile(), after which any model with the same shape of operations will use a numba-compiled version


"""
from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import Dict
import typing
from .param import Param, _Param, INPUT, INDEX, ParamKind
from collections.abc import Iterable, Callable
from collections import defaultdict, OrderedDict
import operator
import functools
from typing import Any
from numpy.typing import NDArray
import copy as _copy


__all__ = [
    "Model",
    "_indecise",
    "identity",
    "vector",
    "formula",
]

class Model:
    """Represents a function that returns an output from INPUT, THETA parameters, and sometimes INDEX's.

    Parameters
    ----------
    fn : function
        A function that takes in input scalars and returns an output
    args : list
        A list of parameters that correspond to the scalars of the input of fn.
    memory : dictionary
        Acted on by other classes, for now just so that a cost function knows what its based on.

    """
    _compiled_cache = OrderedDict()  # LRU cache for compiled functions
    _cache_limit = 128

    @classmethod
    def clear_cache(cls):
        """Clear the compiled function cache to free memory."""
        cls._compiled_cache.clear()

    @classmethod
    def cache_info(cls):
        """Return info about the compilation cache."""
        return {"size": len(cls._compiled_cache), "limit": cls._cache_limit}

    @property
    def compiled(self) -> bool:
        """Return True if this Model has a compiled numba function in the cache."""
        return hash(self) in Model._compiled_cache

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        self.memory = {}

    def __array_ufunc__(self, ufunc, method, *inputs, **ufunc_kwargs):
        """Handle numpy ufuncs like np.exp, np.sin, np.add, etc."""
        if method != '__call__':
            return NotImplemented

        # Numba can't handle **kwargs - only include if non-empty
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
        """Handle numpy functions like np.dot, np.matmul, np.sum, etc."""
        # Numba can't handle **kwargs - only include if non-empty
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

    # for selecting models in trees
    @property
    def components(self):
        """
        Return a 1D list of components obtained by walking the tree
        and cutting at nodes with fn == iden.

        - Traverse the expression tree depth-first.
        - Whenever we see a Model with fn is iden, we treat its *inner*
          expression as a single component and do not recurse further.
        - If there are no iden boundaries at all, return [self].
        """
        comps = []

        def walk(node):
            # Only care about Model nodes structurally
            if not isinstance(node, Model):
                return

            # If this is a boundary: stop and record the inner expression
            if node.fn is iden or getattr(node.fn, "__name__", "") == "iden":
                if node.args:
                    comps.append(node.args[0])
                else:
                    comps.append(node)
                return

            # Otherwise recurse into children
            for a in node.args:
                walk(a)

        walk(self)

        # If no iden nodes found, the whole model is a single component
        return comps or [self]

    @property
    def params(self):
        """A list of all the THETA params that act on the model.
        """
        # Return cached params if available (models are effectively immutable)
        if hasattr(self, '_cached_params'):
            return self._cached_params

        flat = []
        seen = set()

        def walk(m):
            if isinstance(m, Model):
                for arg in m.args:
                    walk(arg)
            elif isinstance(m, Iterable) and not isinstance(m, (str, bytes, np.ndarray)):
                for item in m:
                    walk(item)
            elif isinstance(m, (int, float, np.number)):
                return
            elif isinstance(m, _Param) and m.kind not in (_Param.input, _Param.index):
                if m not in seen:
                    seen.add(m)
                    flat.append(m)

        walk(self)
        self._cached_params = flat
        return flat

    @property
    def free(self):
        """A list of all the free params, which are INPUT and INDEX that act on the model.
        """
        # Return cached free params if available (models are effectively immutable)
        if hasattr(self, '_cached_free'):
            return self._cached_free

        flat = []
        seen = set()

        def walk(m):
            if isinstance(m, Model):
                if m is not self:
                    for f in m.free:
                        if f not in seen:
                            seen.add(f)
                            flat.append(f)
                else:
                    for arg in m.args:
                        walk(arg)
            elif isinstance(m, Iterable) and not isinstance(m, (str, bytes, np.ndarray)):
                for item in m:
                    walk(item)
            elif isinstance(m, (int, float, np.number)):
                return
            elif isinstance(m, _Param) and m.kind in (_Param.input, _Param.index):
                if m not in seen:
                    seen.add(m)
                    flat.append(m)

        walk(self)
        self._cached_free = flat
        return flat

    def __hash__(self):
        """Two equal-hashed models may have different parameters, but if you use the same scalars for those parameters, they will give the same output.
        """
        # Return cached hash if available (models are effectively immutable)
        if hasattr(self, '_cached_hash'):
            return self._cached_hash

        def walk(m):
            if isinstance(m, _Param):
                return ("param", m.kind)
            elif isinstance(m, Reduction):
                return ("reduce", walk(m.model), hash(m.index), m.op.__name__)
            elif isinstance(m, Model):
                fn = m.fn

                # --- special case: zero-arg const() ---
                if not m.args and getattr(fn, "__name__", "") == "const":
                    val = fn()
                    if isinstance(val, np.ndarray):
                        return ("const_array", val.shape, val.dtype.str, tuple(val.flatten().tolist()))
                    else:
                        return ("const_scalar", val)

                # --- otherwise: normal Model node ---
                def _fn_key(f):
                    # Prefer __name__ since named() sets it explicitly
                    if hasattr(f, "__name__"):
                        if hasattr(f, "__module__"):
                            return (f.__module__, f.__name__)
                        return f.__name__
                    if hasattr(f, "__qualname__"):
                        return (f.__module__, f.__qualname__)
                    return repr(f)

                return ("model", _fn_key(fn), tuple(walk(arg) for arg in m.args))
            elif isinstance(m, np.ndarray):
                return ("ndarray", m.shape, m.dtype.str, tuple(m.flatten().tolist()))
            elif isinstance(m, (list, tuple)):
                return (type(m).__name__, tuple(walk(x) for x in m))
            else:
                try:
                    hash(m)
                    return ("const", m)
                except TypeError:
                    return ("unhashable", type(m).__name__)
        self._cached_hash = hash(walk(self))
        return self._cached_hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Model):
            return False

        def walk(a, b):
            if isinstance(a, _Param) and isinstance(b, _Param):
                if a.kind != b.kind:
                    return False
                if a.kind != _Param.input:
                    return a is b
                else:
                    return True
            if isinstance(a, Reduction) and isinstance(b, Reduction):
                return (a.index == b.index and
                        a.op is b.op and
                        walk(a.model, b.model))
            if isinstance(a, Model) and isinstance(b, Model):
                fn_a = getattr(a.fn, "__name__", repr(a.fn))
                fn_b = getattr(b.fn, "__name__", repr(b.fn))
                if fn_a != fn_b or len(a.args) != len(b.args):
                    return False
                return all(walk(x, y) for x, y in zip(a.args, b.args))
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return (a.shape == b.shape and
                        a.dtype == b.dtype and
                        np.array_equal(a, b))
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return (type(a) is type(b) and
                        len(a) == len(b) and
                        all(walk(x, y) for x, y in zip(a, b)))
            return a == b

        return walk(self, other)

    def eval_py(self, x: NDArray[np.floating] | None, index_map: dict[_Param, int]) -> Any:
        """Evaluate the model using pure Python (no Numba).

        Recursively evaluates the expression tree by calling each
        function node with its evaluated arguments.

        Parameters
        ----------
        x : ndarray | None
            Input data array, or None if the model has no INPUT dependency.
        index_map : dict[_Param, int]
            Mapping from INDEX parameters to their current integer values.

        Returns
        -------
        Any
            The evaluated result (typically ndarray or scalar).
        """
        import warnings

        # Collect warnings with their model context
        collected_warnings = []
        seen_base_messages = set()

        def _eval(m):
            if isinstance(m, Reduction):
                index = m.index
                bound = index.range
                return functools.reduce(
                    m.op,
                    [_eval(m.model % {index: i}) for i in bound]
                )

            if isinstance(m, Model):
                # Capture numpy warnings and add model context
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = m.fn(*[_eval(arg) for arg in m.args])

                for w in caught:
                    base_msg = str(w.message).split("\n")[0]
                    # Only keep the first occurrence (innermost expression)
                    if base_msg not in seen_base_messages:
                        seen_base_messages.add(base_msg)
                        collected_warnings.append((base_msg, m, w.category))

                return result
            if isinstance(m, _Param):
                if m.kind == _Param.theta:
                    return m.value
                if m.kind == _Param.input:
                    return x
                if m.kind == _Param.index:
                    return index_map[m]
                raise ValueError(
                    f"Unknown parameter kind: {m.kind!r}.\n"
                    f"  Expected: _Param.theta, _Param.input, or _Param.index.\n"
                    f"  This usually indicates a corrupted Param object."
                )
            return m

        result = _eval(self)

        # Emit all collected warnings after evaluation completes
        for base_msg, model_expr, category in collected_warnings:
            msg = f"{base_msg}\n  In expression: {model_expr}"
            warnings.warn(msg, category, stacklevel=4)

        return result

    def eval(self, x: NDArray[np.floating] | None, index_map: dict[_Param, int], numba: bool = False) -> Any:
        """Evaluate the model, using compiled code if available.

        Checks the compilation cache for a Numba-compiled version of this
        model's structure. If found, uses the fast compiled path; otherwise
        falls back to Python evaluation.

        Parameters
        ----------
        x : ndarray | None
            Input data array, or None if the model has no INPUT dependency.
        index_map : dict[_Param, int]
            Mapping from INDEX parameters to their current integer values.
        numba : bool, default False
            If True, raises an error when no compiled function is available.

        Returns
        -------
        Any
            The evaluated result (typically ndarray or scalar).

        Raises
        ------
        AttributeError
            If ``numba=True`` and no compiled function exists in cache.
        """
        if not self.params:
            theta = None
        else:
            theta = np.array([p.value for p in self.params])

        index_params = [p for p in self.free if p.kind == _Param.index]
        index_list = [index_map[p] for p in index_params] if index_params else None

        current_hash = hash(self) # Hash depends on symbolic structure
        if current_hash in Model._compiled_cache:
            return Model._compiled_cache[current_hash](x, index_list, theta)
        else:
            if numba:
                raise AttributeError("No compiled function in cache")
            return self.eval_py(x, index_map)

    def __call__(self, x: NDArray[np.floating] | _Param | Model | None = None) -> Any:
        """Evaluate the model or perform substitution.

        The behavior depends on the argument type:
        - If ``x`` is a Param or Model, performs substitution (same as ``%``).
        - If ``x`` is array-like, evaluates the model with that input data.
        - If ``x`` is None, evaluates a model with no INPUT dependency.

        Parameters
        ----------
        x : ndarray | _Param | Model | None, optional
            Input data for evaluation, or a Param/Model for substitution.

        Returns
        -------
        Any
            Evaluation result (ndarray/scalar) or substituted Model.

        Raises
        ------
        TypeError
            If the model requires INPUT but none provided, or vice versa.

        Examples
        --------
        >>> model = Param("a") * INPUT
        >>> model(np.array([1, 2, 3]))  # Evaluate
        >>> model(other_param)  # Substitute (same as model % other_param)
        """
        # If x is a Param/Model, treat as substitution, not evaluation
        if isinstance(x, (_Param, Model)):
            return self % x
        if x is not None and not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if INPUT in self.free and x is None:
            raise TypeError(
                f"Model requires input data but none provided.\n"
                f"  Call as: model(x) where x is your data array.\n"
                f"  Model: {self}"
            )
        if INPUT not in self.free and x is not None:
            raise TypeError(
                f"Model doesn't accept input data.\n"
                f"  This model has no INPUT dependency, call as: model()\n"
                f"  Model: {self}"
            )
        if not self.free:
            return self.eval(None, {})
        if self.free == [INPUT]:
            return self.eval(x, {})
        else:
            return self % x
    def __getitem__(self, key: dict | int | _Param) -> Any:
        """Access model values by index or make model indexable.

        Provides convenient syntax for working with INDEX parameters:
        - ``model[index_param]``: Make model indexable by that INDEX param
        - ``model[2]``: If model has no INDEX, wrap and evaluate at index 2
        - ``model[{idx: 5}]``: Substitute specific index param with value

        Parameters
        ----------
        key : dict | int | _Param
            An INDEX parameter, an integer index value, or a dictionary
            mapping INDEX params to values.

        Returns
        -------
        Model | Any
            A Model if INPUT is present or wrapping in _indecise,
            otherwise the evaluated result.

        Examples
        --------
        >>> arr = const([10, 20, 30])
        >>> arr[INDEX]  # Make indexable by INDEX
        >>> arr[1]      # Get element 1 (implicitly wraps in _indecise)
        >>> indexed_model[{my_index: 5}]  # Substitute specific index
        """
        # If key is an INDEX param, wrap in _indecise
        if isinstance(key, _Param) and key.kind is ParamKind.INDEX:
            return _indecise(self, key)

        # Check if model already has INDEX dependencies
        has_index = any(
            isinstance(p, _Param) and p.kind is ParamKind.INDEX
            for p in self.free
        )

        if isinstance(key, int) and not has_index:
            # No INDEX params - wrap in _indecise with global INDEX first
            indexed = _indecise(self, INDEX)
            if INPUT in indexed.free:
                return indexed % {INDEX: key}
            return indexed.eval(None, {INDEX: key})

        # Current behavior: substitute INDEX
        if INPUT in self.free:
            if not isinstance(key, dict):
                return self % {INDEX: key}
            return self % key
        if not isinstance(key, dict):
            return self.eval(None, {INDEX: key})
        return self.eval(None, key)

    def copy_with_args(self, args: list) -> Model:
        """Create a shallow copy of this Model with new arguments.

        Parameters
        ----------
        args : list
            New argument list to use instead of the current args.

        Returns
        -------
        Model
            A new Model with the same function but different arguments.
        """
        return type(self)(self.fn, args)

    def copy(self) -> Model:
        """Create a deep copy of the model expression tree.

        Creates a new Model with:
        - All Model/Reduction nodes recreated as new instances
        - THETA Params cloned into new Param instances (preserving values)
        - INPUT/INDEX Params shared (same singletons)
        - Lists/tuples inside args recursively copied

        Returns
        -------
        Model
            A deep copy of this model with independent THETA parameters.
        """
        model_memo = {}
        param_memo = {}

        def walk(node):
            # --- Models (including Reduction via copy_with_args) ---
            if isinstance(node, Model):
                obj_id = id(node)
                if obj_id in model_memo:
                    return model_memo[obj_id]

                new_args = [walk(a) for a in node.args]
                new_node = node.copy_with_args(new_args)
                model_memo[obj_id] = new_node
                return new_node

            # --- Params ---
            if isinstance(node, _Param):
                # Keep INPUT/INDEX singletons shared
                if node.kind in (_Param.input, _Param.index):
                    return node

                # Clone θ-params, reusing mapping so they stay consistent
                if node in param_memo:
                    return param_memo[node]

                new_p = _copy.copy(node)   # new Param instance, same attributes
                param_memo[node] = new_p
                return new_p

            # --- Lists / tuples inside args ---
            if isinstance(node, (list, tuple)):
                return type(node)(walk(a) for a in node)

            # --- Scalars, ndarrays, etc. ---
            return node

        return walk(self)


    def __mod__(self, subs: dict | Any) -> Model:
        """Substitute parameters or sub-expressions in this model.

        The ``%`` operator performs symbolic substitution, replacing
        occurrences of specified parameters or models with new values.

        Parameters
        ----------
        subs : dict | Any
            Either a dictionary mapping old nodes to new values, or a
            single value to substitute for INPUT.

        Returns
        -------
        Model
            A new Model with substitutions applied.

        Examples
        --------
        >>> model = Param("a") * INPUT + Param("b")
        >>> model % 5  # Substitute INPUT with 5
        >>> model % {INPUT: x_data}  # Same as above
        >>> model % {some_param: 2.0}  # Fix a parameter to a constant
        """
        if not isinstance(subs, dict):
            subs = {INPUT: subs}

        def walk(m):
            # 1) If this node is exactly one of the keys, replace it wholesale
            if m in subs:
                return subs[m]

            # 2) Otherwise recurse structurally
            if isinstance(m, Model):
                return m.copy_with_args([walk(a) for a in m.args])

            elif isinstance(m, Iterable) and not isinstance(m, (str, bytes, np.ndarray)):
                return type(m)(walk(x) for x in m)

            # 3) Fallback: leave as is
            return m

        return walk(self)




#REPRESENTATION

    def __repr__(self):
        #fn_name = getattr(self.fn, '__name__', 'λ')
        #if fn_name == '<lambda>':
        #    fn_name = 'λ'
        #arg_strs = [repr(arg) for arg in self.args]
        #return f"Model<{fn_name}>({', '.join(arg_strs)})"
        return self._format()

    def __str__(self):
        return self._format()

    def _format(self, parent_prec: int = 0, side: str | None = None) -> str:
        """Format the model as a human-readable expression string.

        Recursively formats the expression tree with proper operator
        precedence and parenthesization.

        Parameters
        ----------
        parent_prec : int, default 0
            Precedence level of the parent expression (for parentheses).
        side : str | None, default None
            Which side of a binary op this is ('left' or 'right').

        Returns
        -------
        str
            A formatted string representation of the model.
        """
        if isinstance(self, Reduction):
            inner = self.model._format() if isinstance(self.model, Model) else str(self.model)
            return f"{self.op.__name__}_reduce({inner}, index={self.index})"
        fn_name = getattr(self.fn, '__name__', 'λ')
        if fn_name == '<lambda>':
            fn_name = 'λ'

        op_info = {
            'add': ('+', 1, 'left'),
            'radd': ('+', 1, 'left'),
            'sub': ('-', 1, 'left'),
            'rsub': ('-', 1, 'left'),
            'mul': ('*', 2, 'left'),
            'rmul': ('*', 2, 'left'),
            'div': ('/', 2, 'left'),
            'rdiv': ('/', 2, 'left'),
            'pow': ('**', 3, 'right'),
            'rpow': ('**', 3, 'right'),
            'eq': ('==', 0, 'left'),
            'ne': ('!=', 0, 'left'),
            'lt': ('<', 0, 'left'),
            'le': ('<=', 0, 'left'),
            'gt': ('>', 0, 'left'),
            'ge': ('>=', 0, 'left'),
        }

        def parens_needed(p_prec, p_assoc, c_prec, is_left):
            if c_prec < p_prec:
                return True
            if c_prec > p_prec:
                return False
            if p_assoc == 'left' and not is_left:
                return True
            if p_assoc == 'right' and is_left:
                return True
            return False

        def get_prec(m):
            """Return precedence for formatting a Model node."""
            if isinstance(m, Model):
                fn_name = getattr(m.fn, "__name__", "")
                if fn_name == "const":
                    return 100   # const literals are atomic
                if fn_name not in op_info:
                    return 100   # function calls (sum, exp, log, etc.) are atomic
                return op_info[fn_name][1]
            return 100  # numbers, arrays, etc. are atomic

        if fn_name in op_info and len(self.args) == 2:
            op, prec, assoc = op_info[fn_name]
            if fn_name.startswith('r'):
                right, left = self.args
            else:
                left, right = self.args

            if isinstance(left, Model):
                lstr = left._format(prec, 'left')
            else:
                lstr = str(left)
            if isinstance(right, Model):
                rstr = right._format(prec, 'right')
            else:
                rstr = str(right)

            if isinstance(left, Model):
                if parens_needed(prec, assoc, get_prec(left), True):
                    lstr = f'({lstr})'
            if isinstance(right, Model):
                if parens_needed(prec, assoc, get_prec(right), False):
                    rstr = f'({rstr})'

            return f"{lstr} {op} {rstr}"

        elif fn_name == 'neg' and len(self.args) == 1:
            arg = self.args[0]
            astr = arg._format(4) if isinstance(arg, Model) else str(arg)
            return f"-{astr}"

        elif fn_name == 'const' and len(self.args) == 0:
            try:
                val = self.fn()
            except Exception:
                return "const(?)"
            if isinstance(val, np.ndarray):
                # Show small arrays fully, large arrays with first/last elements
                if val.size <= 6:
                    return f"{val}"
                elif val.ndim == 1:
                    return f"[{val[0]:.4g}, {val[1]:.4g}, ..., {val[-2]:.4g}, {val[-1]:.4g}]"
                else:
                    return f"const(array, shape={val.shape}, dtype={val.dtype})"
            else:
                return f"{val!r}"

        # generic case

        arg_strs = [arg._format() if isinstance(arg, Model) else str(arg) for arg in self.args]
        return f"{fn_name}({', '.join(arg_strs)})"

    def freeze(self) -> Model:
        """Replace all THETA parameters with their current values.

        Creates a new Model where every optimizable parameter is
        substituted with its current numeric value, effectively
        "freezing" the model at its current parameter state.

        Returns
        -------
        Model
            A new Model with all parameters replaced by constants.

        Examples
        --------
        >>> fitted_model.freeze()  # Lock in fitted parameter values
        """
        return self % {p : p.value for p in self.params}

# Operations

def named(fn, name):
    fn.__name__ = name
    return fn

_op_map = {
    '__add__': named(lambda a, b: a + b, 'add'),
    '__radd__': named(lambda a, b: b + a, 'radd'),
    '__sub__': named(lambda a, b: a - b, 'sub'),
    '__rsub__': named(lambda a, b: b - a, 'rsub'),
    '__mul__': named(lambda a, b: a * b, 'mul'),
    '__rmul__': named(lambda a, b: b * a, 'rmul'),
    '__truediv__': named(lambda a, b: a / b, 'div'),
    '__rtruediv__': named(lambda a, b: b / a, 'rdiv'),
    '__pow__': named(lambda a, b: a ** b, 'pow'),
    '__rpow__': named(lambda a, b: b ** a, 'rpow'),
    '__neg__': named(lambda a: -a, 'neg'),
    #DO NOT UNCOMMENT '__eq__': named(lambda a, b: a == b, 'eq'),
    '__ne__': named(lambda a, b: a != b, 'ne'),
    '__lt__': named(lambda a, b: a < b, 'lt'),
    '__le__': named(lambda a, b: a <= b, 'le'),
    '__gt__': named(lambda a, b: a > b, 'gt'),
    '__ge__': named(lambda a, b: a >= b, 'ge'),
}

def bind_ops(cls):
    for name, fn in _op_map.items():
        def op_method(self, other=None, fn=fn):
            if other is None:
                return Model(fn, [self])
            if not isinstance(other, typing.Hashable):
                other = const(other)
            return Model(fn, [self, other])# if 'r' not in name else Model(fn, [other, self])
        setattr(cls, name, op_method)

bind_ops(Model)
bind_ops(_Param)


# Reduction

class Reduction(Model):
    """A Model that reduces over an INDEX parameter using an operator.

    Represents a summation (or other reduction) over a loop variable,
    e.g., ``sum_{i=0}^{n} f(i)``. Currently only ``operator.add`` is
    supported for Numba compilation.

    Parameters
    ----------
    model : Model
        The expression to reduce over.
    index_param : Param
        An INDEX parameter with a ``range`` attribute defining the
        iteration bounds.
    op : callable, default operator.add
        The reduction operator. Must be ``operator.add``.

    Attributes
    ----------
    model : Model
        The inner expression being reduced.
    index : Param
        The INDEX parameter controlling the loop.
    op : callable
        The reduction operator.

    Examples
    --------
    >>> i = index(10)  # Loop from 0 to 9
    >>> weights = _indecise(const([...]), i)
    >>> total = Reduction(weights * some_model, i)  # Sum over i
    """
    _supported_ops = {operator.add: 'add'}  # Only add is supported by compiler

    def __init__(self, model: Model, index_param: _Param, op: Callable = operator.add):
        if not isinstance(index_param, _Param):
            raise TypeError("Reduction index must be a Param")
        if op not in Reduction._supported_ops:
            raise ValueError(
                f"Reduction only supports operator.add, got {op.__name__}.\n"
                f"  For products, use: exp(Reduction(log(model), index))"
            )
        self.model = model
        self.index = index_param
        self.op = op
        super().__init__(op, [model])

    @property
    def params(self):
        return self.model.params

    @property
    def free(self):
        return [p for p in self.model.free if p is not self.index]

    def copy_with_args(self, args):
        return Reduction(args[0], self.index, self.op)

    def __repr__(self):
        op_name = getattr(self.op, '__name__', 'λ')
        if op_name == '<lambda>':
            op_name = 'λ'
        return f"Reduce<{op_name}>({repr(self.model)}, index={self.index})"

    def __str__(self):
        op_name = getattr(self.op, '__name__', 'λ')
        if op_name == '<lambda>':
            op_name = 'λ'
        return f"{op_name}_reduce({str(self.model)}, index={self.index})"


# Useful methods

def const(val: Any) -> Model:
    """Wrap a constant value as a Model node.

    Creates a Model that always returns the given constant value
    when evaluated. Useful for embedding literal values in
    expression trees.

    Parameters
    ----------
    val : Any
        The constant value to wrap (scalar, array, etc.).

    Returns
    -------
    Model
        A Model node that evaluates to the constant value.

    Examples
    --------
    >>> c = const(5.0)
    >>> c()
    5.0
    >>> arr = const(np.array([1, 2, 3]))
    """
    # Convert lists/tuples to arrays for consistency and hashability
    if isinstance(val, (list, tuple)):
        val = np.asarray(val)
    f = lambda val=val: val
    f.__name__ = f"const"
    return Model(f, [])

def _indecise(obj: Any, index: _Param = INDEX) -> Model:
    """Create a Model that selects from an array by index.

    Wraps an array (or constant) so that it can be indexed by
    an INDEX parameter, enabling iteration over array elements
    in reductions.

    Parameters
    ----------
    obj : Any
        The array or constant to index into.
    index : Param, default INDEX
        The INDEX parameter to use for selection.

    Returns
    -------
    Model
        A Model that evaluates to ``obj[index]``.

    Examples
    --------
    >>> weights = _indecise(const([0.3, 0.5, 0.2]), i)
    >>> Reduction(weights * some_model, i)  # Sum weighted models
    """
    if not isinstance(obj, typing.Hashable):
        obj = const(obj)
    fn = lambda arr, i: arr[int(i)]
    fn.__name__ = 'indecise'
    return Model(fn, [obj, index])



# gradient

def _zero(): return 0
def _one(): return 1

def _grad_fn(model: Model | _Param | Any, wrt: _Param) -> Model | int:
    """Compute the symbolic gradient of a model w.r.t. a parameter.

    Recursively applies differentiation rules to build a new Model
    representing the derivative. Supports standard arithmetic operations,
    exp, log, sqrt, sum, where, and _indecise.

    Parameters
    ----------
    model : Model | _Param | Any
        The expression to differentiate.
    wrt : Param
        The parameter to differentiate with respect to.

    Returns
    -------
    Model | int
        The derivative expression (0 or 1 for simple cases).

    Raises
    ------
    ValueError
        If an unsupported operation is encountered.
    """
    if isinstance(model, _Param):
        return _one() if model is wrt else _zero()

    if not isinstance(model, Model):
        return _zero()

    if isinstance(model, Reduction):
        df = _grad_fn(model.model, wrt)
        if isinstance(df, (int, float, np.number)):
            df = const(df)
        return Reduction(df, model.index, model.op)

    if isinstance(model, Model) and not model.args:
        # constant Model (i.e., const(...))
        return _zero()

    fn = getattr(model.fn, '__name__', None)
    args = model.args

    # unary ops
    if fn == 'iden':
        return _grad_fn(args[0], wrt)
    if fn == 'neg':
        return -_grad_fn(args[0], wrt)
    if fn == 'exp':
        return model * _grad_fn(args[0], wrt)
    if fn == 'sqrt':
        return _grad_fn(args[0], wrt) / (2 * model)
    if fn == 'log':
        return _grad_fn(args[0], wrt) / args[0]

    # binary ops
    if fn == 'add' or fn == 'radd':
        return _grad_fn(args[0], wrt) + _grad_fn(args[1], wrt)
    if fn == 'sub':
        return _grad_fn(args[0], wrt) - _grad_fn(args[1], wrt)
    if fn == 'rsub':
        return _grad_fn(args[1], wrt) - _grad_fn(args[0], wrt)
    if fn == 'mul' or fn == 'rmul':
        return args[0] * _grad_fn(args[1], wrt) + args[1] * _grad_fn(args[0], wrt)
    if fn == 'div':
        num = _grad_fn(args[0], wrt) * args[1] - args[0] * _grad_fn(args[1], wrt)
        denom = args[1] ** 2
        return num / denom
    if fn == 'rdiv':
        num = _grad_fn(args[1], wrt) * args[0] - args[1] * _grad_fn(args[0], wrt)
        denom = args[0] ** 2
        return num / denom
    if fn == 'pow':
        base, exp = args
        df = _grad_fn(base, wrt)
        if isinstance(exp, (Model, _Param)) and not base is exp:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        elif isinstance(exp, Model) and exp.params:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        else:
            # constant exponent
            return exp * (base ** (exp - 1)) * df

    if fn == 'rpow':
        exp, base = args
        df = _grad_fn(base, wrt)
        if isinstance(exp, (Model, _Param)) and not base is exp:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        elif isinstance(exp, Model) and exp.params:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        else:
            return exp * (base ** (exp - 1)) * df

    if fn == '_indecise':
        entries, index = model.args
        if isinstance(entries, (list, tuple)):
            # differentiate each entry
            grads = []
            for e in entries:
                if isinstance(e, Model):
                    grads.append(_grad_fn(e, wrt))
                elif isinstance(e, _Param):
                    grads.append(_one() if e is wrt else _zero())
                else:
                    grads.append(_zero())
            return Model(model.fn, [grads, index])
        else:
            d_entries = simplify(_grad_fn(entries, wrt))
            if d_entries == 0:
                return 0
            return Model(model.fn, [d_entries, index])

    if fn == 'sum':
        return Model(np.sum, [_grad_fn(args[0], wrt)])
    
    if fn == 'where':
        cond, x, y = args
        # Differentiate only the "branches"
        dx = _grad_fn(x, wrt)
        dy = _grad_fn(y, wrt)
        return Model(np.where, [cond, dx, dy])
    raise ValueError(
        f"Cannot differentiate operation '{fn}'.\n"
        f"  Supported: add, sub, mul, div, pow, neg, exp, sqrt, log, sum, where, _indecise.\n"
        f"  Consider using grad=False in fit() for numerical gradients."
    )

def _grad(self, wrt=None):
    if wrt is None:
        wrt = self.params
    if isinstance(wrt, _Param):
        return ensure_model(_grad_fn(self, wrt)).simplify()
    if isinstance(wrt, (list, tuple)):
        return vector(*[ensure_model(_grad_fn(self, p)).simplify() for p in wrt])
    raise TypeError(f"grad expects Param, list of Param, or None, got {type(wrt)}")


Model.grad = _grad

def iden(x):
    """Identity function, returns input unchanged."""
    return x


def identity(val: Any) -> Model:
    """Wrap a value in an identity Model node.

    Creates a Model that simply passes through its argument unchanged.
    Useful for marking component boundaries in composite models.

    Parameters
    ----------
    val : Any
        The value or Model to wrap.

    Returns
    -------
    Model
        A Model that evaluates to the input value.
    """
    return Model(iden, [val])

def ensure_model(x):
    return x if isinstance(x, Model) else identity(x)

def vector(*models):
    """Create a Model that evaluates multiple models and stacks results."""
    def stack_fn(*args):
        return np.array(args)
    stack_fn.__name__ = "vector"
    return Model(stack_fn, [ensure_model(m) for m in models])


def formula(fn: Callable) -> Callable[..., Model]:
    """Decorator to create a Model factory from a function.

    The decorated function's first argument becomes INPUT (the independent
    variable), and remaining arguments become Model parameters. Returns a
    factory that takes the parameters and produces a Model.

    Parameters
    ----------
    fn : Callable
        A function where the first argument is the input data and
        subsequent arguments are parameters.

    Returns
    -------
    Callable[..., Model]
        A factory function that takes parameters and returns a Model.

    Examples
    --------
    >>> @formula
    ... def line(x, a, b):
    ...     return a * x + b
    >>> model = line(Param('slope'), Param('intercept'))
    >>> model(np.array([1, 2, 3]))

    >>> @formula
    ... def gaussian(x, mu, sigma):
    ...     return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2*np.pi))
    >>> g = gaussian(Param('mu')(0), Param.positive('sigma')(1))
    """
    @functools.wraps(fn)
    def factory(*args):
        return Model(fn, [INPUT, *args])
    return factory


def _is_scalar_equal(x, val):
    """Check if x equals val, handling floats with tolerance."""
    if isinstance(x, (Model, _Param)):
        return False
    if isinstance(x, (int, float, np.number)):
        return np.isclose(x, val, rtol=1e-14, atol=1e-14)
    return x == val


def simplify(m: Model | Any) -> Model | Any:
    """Simplify a model by applying algebraic identities.

    Performs constant folding and removes redundant operations:
    - ``x * 0 -> 0``, ``x * 1 -> x``
    - ``x + 0 -> x``
    - ``x - 0 -> x``, ``0 - x -> -x``
    - ``x / 1 -> x``, ``0 / x -> 0``
    - ``x ** 0 -> 1``, ``x ** 1 -> x``

    Parameters
    ----------
    m : Model | Any
        The expression to simplify.

    Returns
    -------
    Model | Any
        A simplified expression (may be a constant if fully reduced).
    """
    if not isinstance(m, Model):
        return m

    if isinstance(m, Reduction):
        return Reduction(simplify(m.model), m.index, m.op)

    fn = getattr(m.fn, '__name__', None)
    args = [simplify(arg) for arg in m.args]

    is_zero = lambda x: _is_scalar_equal(x, 0)
    is_one = lambda x: _is_scalar_equal(x, 1)

    if fn in ('mul', 'rmul'):
        if any(is_zero(arg) for arg in args):
            return 0
        non_one = [arg for arg in args if not is_one(arg)]
        if len(non_one) == 0:
            return 1
        if len(non_one) == 1:
            return ensure_model(non_one[0])
        return Model(m.fn, non_one)

    if fn in ('add', 'radd'):
        non_zero = [arg for arg in args if not is_zero(arg)]
        if len(non_zero) == 0:
            return 0
        if len(non_zero) == 1:
            return ensure_model(non_zero[0])
        return Model(m.fn, non_zero)

    if fn == 'sub':
        if is_zero(args[1]):
            return ensure_model(args[0])
        if is_zero(args[0]):
            return ensure_model(-args[1])
        return Model(m.fn, args)

    if fn in ('div', 'truediv'):
        if is_zero(args[0]):
            return 0
        if is_one(args[1]):
            return ensure_model(args[0])
        return Model(m.fn, args)

    if fn == 'pow':
        base, exp = args
        if is_zero(exp):
            return 1
        if is_one(exp):
            return ensure_model(base)
        return Model(m.fn, args)

    if fn == 'neg':
        return ensure_model(-args[0])

    return m.copy_with_args(args)

Model.simplify = simplify


def shape(obj: Any) -> tuple | _Param:
    """Infer the output shape of a model or value.

    Recursively determines what shape the expression will produce
    when evaluated.

    Parameters
    ----------
    obj : Any
        A Model, Param, array, or scalar.

    Returns
    -------
    tuple | _Param
        The shape as a tuple, or INPUT if the shape depends on input data.
    """
    if isinstance(obj, (int, float, np.number)):
        return ()

    if isinstance(obj, np.ndarray):
        return obj.shape

    if obj is INPUT:
        return INPUT                       # e.g. you called shape(X) directly

    if isinstance(obj, _Param):
        return ()                      # θ_k etc. are scalars

    if isinstance(obj, Reduction):
        return shape(obj.model)        # reduction never *adds* new shape

    if isinstance(obj, Model):
        fn_name = getattr(obj.fn, '__name__', '')

        # _indecise always returns a scalar for a single index lookup
        if fn_name == '_indecise':
            return ()

        # 1. if expression explicitly depends on free X → vector
        if INPUT in obj.free:
            return INPUT

        # 2. constant Model  (no args)  → try to evaluate once
        if not obj.args:
            try:
                val = obj.fn()
                return shape(val)
            except Exception:
                return ()

        # 3. otherwise: first non-scalar arg determines shape
        for arg in obj.args:
            sh = shape(arg)
            if sh not in ((), None):
                return sh
        return ()                      # pure-scalar combination

    # ---------------------------------------------------------------- fallback
    return ()

Model.shape = shape
