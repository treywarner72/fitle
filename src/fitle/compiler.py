"""compiler.py
=============
Numba JIT compilation for Model expression trees.

This module converts Model DAGs (directed acyclic graphs) into
Numba-compiled functions for high-performance numerical evaluation.

The compilation process has four stages:

1. **DAG Construction** (``_build_dag``): Walk the Model tree and build
   an intermediate representation (IR) of nodes, detecting common
   subexpressions to avoid redundant computation.

2. **Node Partitioning** (``_partition_nodes``): Separate IR nodes into
   "global" nodes (no loop dependencies) and "loop" buckets (nodes that
   depend on specific INDEX parameters).

3. **Temporary Allocation** (``_allocate_temps``): Assign temporary variable
   names to nodes that are used multiple times, enabling code reuse.

4. **Source Emission** (``_emit_source``): Generate Python source code
   that can be compiled by Numba's ``@njit`` decorator.

The compiled function signature is::

    compiled(x, indices, theta) -> result

Where:
- ``x``: Input data array (or None)
- ``indices``: List of current index values for INDEX params
- ``theta``: Array of parameter values in tree-walk order
"""
from __future__ import annotations

import types, textwrap, operator
import numpy as np
from numba import njit
from collections import defaultdict, deque
from .model import Model, Reduction
from .param import Param, _Param, INPUT

# ---------------------------  Internal IR primitives  ---------------------------

class _IRNode:
    """A single expression node in the intermediate representation.

    Attributes
    ----------
    expr : str
        RHS Python string that can be pasted into generated code.
    deps : frozenset[_Param]
        Set of INDEX parameters this node depends on.
    users : int
        Number of times other IR nodes reference this node.
    name : str | None
        Temporary variable name assigned during scheduling (None = inline).
    is_red : tuple[str, str] | None
        None for plain expressions, or (acc_name, op) for reductions.
    is_leaf : bool
        True if this is a leaf node (param, constant).
    is_acc : bool
        True if this is an accumulator node for a reduction.
    model : Model | None
        Reference to the original Model (for shape inference).
    """
    __slots__ = ("expr", "deps", "users", "name", "is_red", "is_leaf", "is_acc", "model")
    def __init__(self, expr, deps):
        self.expr   = expr
        self.deps   = deps            # frozenset({index_param, …})
        self.users  = 0
        self.name   = None
        self.is_red = False
        self.is_leaf = False
        self.is_acc = False
        self.model  = None


class _LoopIR:
    """Container for IR nodes that share exactly one INDEX parameter.

    Groups all expressions that depend on a specific loop variable,
    enabling generation of a single ``for`` loop in the output code.

    Attributes
    ----------
    idx : Param
        The INDEX parameter this loop iterates over.
    body : list[_IRNode]
        IR nodes executed inside the loop.
    accums : list[tuple]
        Accumulator specifications ``(lhs, node, op)``.
    """
    def __init__(self, index_param: _Param):
        self.idx    = index_param
        self.body   = []              # list[_IRNode]   – executed in loop
        self.accums = []              # list[(lhs, node, op)]
        # each acc gets initialised to 0.0 before the loop


# ---------------------------  Compiler class  ---------------------------

class Compiler:
    """Compiles Model expression trees to Numba JIT functions.

    Converts a Model DAG into optimized Python source code that is
    then compiled by Numba for fast numerical evaluation.

    Parameters
    ----------
    root : Model
        The root Model to compile.

    Attributes
    ----------
    root : Model
        The Model being compiled.
    code : str
        Generated Python source code (available after ``compile()``).

    Examples
    --------
    >>> c = Compiler(model)
    >>> compiled_fn = c.compile()  # Returns numba.njit callable
    >>> print(c.code)  # Inspect generated source
    """
    # --------------------------------------------------------------------- #
    def __init__(self, root: Model):
        self.root      = root
        self._node_map = {}           # Model instance   -> _IRNode
        self._loops    = {}           # _Param.index      -> _LoopIR
        self._globals  = []           # list[_IRNode]    – deps == ∅
        self._idx_vars = {}
        self._final    = None         # _IRNode that yields return value
        self._hoisted_consts: dict = {}
        self._hoisted_fns   : dict = {}
        self._param_to_idx = {p: i for i, p in enumerate(root.params)}

    # ------------------------------------------------------------------ #
    #  PUBLIC ENTRY
    # ------------------------------------------------------------------ #
    def compile(self):
        """Compile the Model and return a Numba JIT function.

        Executes the four compilation stages: DAG construction, node
        partitioning, temporary allocation, and source emission.

        Returns
        -------
        callable
            A Numba ``@njit`` compiled function with signature
            ``compiled(x, indices, theta) -> result``.

        Raises
        ------
        TypeError
            If the model contains numpy functions called with keyword
            arguments, which Numba cannot compile.
        """
        self._check_no_kwargs(self.root)
        self._build_dag(self.root)
        self._partition_nodes()
        self._allocate_temps()
        source, ns = self._emit_source()
        self.code = source
        exec(source, ns)
        return njit(ns["compiled"])

    def _check_no_kwargs(self, model) -> None:
        """Check that no functions in the tree use kwargs.

        Raises TypeError if any function was created with keyword arguments
        from numpy's dispatch protocol, since Numba cannot compile these.
        """
        if not isinstance(model, Model):
            return

        fn = model.fn
        if getattr(fn, '_has_kwargs', False):
            kwargs_str = ', '.join(getattr(fn, '_kwargs_info', ()))
            raise TypeError(
                f"Cannot compile model: numpy function '{fn.__name__}' was called "
                f"with keyword arguments ({kwargs_str}).\n"
                f"  Numba does not support keyword arguments in compiled functions.\n"
                f"  Either:\n"
                f"    - Use positional arguments instead\n"
                f"    - Use an equivalent expression (e.g., np.where instead of where=)\n"
                f"    - Skip compilation and use Python evaluation"
            )

        for arg in model.args:
            if isinstance(arg, Model):
                self._check_no_kwargs(arg)

    def _idx_var(self, idx_param: _Param) -> str:
        """Return the unique loop-variable name for an INDEX Param.

        Parameters
        ----------
        idx_param : Param
            An INDEX parameter.

        Returns
        -------
        str
            A unique variable name like ``_i_12345678``.
        """
        return self._idx_vars.setdefault(idx_param,
                                         f"_i_{id(idx_param)}")

    def _try_reuse_compiled(self, model: Model):
        """Try to reuse a pre-compiled sub-model.

        If the model is already in the compiled cache, generate a call
        to the pre-compiled function instead of inlining its expression.
        This can significantly reduce LLVM compilation time.

        Parameters
        ----------
        model : Model
            A sub-model to potentially reuse.

        Returns
        -------
        _IRNode | None
            An IR node representing the call to the pre-compiled function,
            or None if the model cannot be reused.
        """
        # Check if model is in compiled cache
        try:
            model_hash = hash(model)
        except Exception:
            return None

        if model_hash not in Model._compiled_cache:
            return None

        # Get sub-model's params and check they're all in parent's params
        sub_params = model.params
        if not sub_params:
            # No params - just evaluate and hoist as constant
            return None

        # Check all sub-model params exist in parent
        for p in sub_params:
            if p not in self._param_to_idx:
                return None  # Sub-model has params not in parent

        # Check if sub-model has INDEX dependencies (complex case, skip for now)
        if any(p.kind == _Param.index for p in model.free):
            return None  # Skip models with INDEX for now

        # Build theta indexing expression
        indices = [self._param_to_idx[p] for p in sub_params]

        # Check if indices are contiguous (can use slice)
        if indices == list(range(indices[0], indices[0] + len(indices))):
            theta_expr = f"theta[{indices[0]}:{indices[-1]+1}]"
        else:
            # Need explicit array construction
            theta_expr = f"np.array([{', '.join(f'theta[{i}]' for i in indices)}])"

        # Hoist the compiled function
        compiled_fn = Model._compiled_cache[model_hash]
        fn_name = f"_compiled_{id(model)}"
        self._hoisted_fns[fn_name] = compiled_fn

        # Generate the call expression
        expr = f"{fn_name}(x, None, {theta_expr})"

        node = _IRNode(expr, frozenset())  # No INDEX deps
        node.model = model
        node.is_leaf = False
        return self._cache(model, node)

    # ------------------------------------------------------------------ #
    #  1.  DAG construction  (common-sub-expression detection)
    # ------------------------------------------------------------------ #
    def _build_dag(self, model) -> _IRNode:
        """Build the IR DAG by walking the Model tree.

        Performs depth-first traversal, creating IR nodes and detecting
        common subexpressions. Each unique Model gets a single IR node,
        with ``users`` tracking reference counts.

        Parameters
        ----------
        model : Model | Param | Any
            The expression node to process.

        Returns
        -------
        _IRNode
            The IR node representing this expression.
        """
        if model in self._node_map:
            node = self._node_map[model]
            return node

        # ----- LEAF CASES ------------------------------------------------
        if isinstance(model, _Param):
            if model.kind == _Param.theta:
                idx = self._param_to_idx.get(model, -1)
                node = _IRNode(f"theta[{idx}]", frozenset())
            elif model is INPUT:
                node = _IRNode("x", frozenset())
            elif model.kind == _Param.index:
                # Leaf that will be replaced by the loop variable.
                var = self._idx_var(model)
                node = _IRNode(var, frozenset({model}))   # depends on its own index
                node.is_leaf = True
                return self._cache(model, node)
            node.is_leaf = True
            return self._cache(model, node)

        if not isinstance(model, Model):
            # constants / numpy arrays → hoist as const_<id>
            if isinstance(model, np.ndarray):
                cname = f"const_{id(model)}"
                node = _IRNode(cname, frozenset())
                self._hoisted_consts[cname] = model
            else:
                node = _IRNode(repr(model), frozenset())
            node.is_leaf = True
            return self._cache(model, node)

        # ----- REUSE PRE-COMPILED SUB-MODEL ------------------------------
        # Check if this sub-model is already compiled and can be reused
        if isinstance(model, Model) and model is not self.root:
            reuse_node = self._try_reuse_compiled(model)
            if reuse_node is not None:
                return reuse_node

        # ----- REDUCTION -------------------------------------------------
        if isinstance(model, Reduction):
            if model.op is not operator.add:
                raise ValueError("Only operator.add supported in Reduction")
            sub_node = self._build_dag(model.model)
            sub_node.users += 1     # reduction body uses it once per iter
            idx      = model.index
            acc_name = f"_acc_{id(model)}"
            red_node = _IRNode(sub_node.expr, frozenset({idx}))
            red_node.is_red = (acc_name, "+")
            red_node.model  = model
            # Accumulator itself is a zero-dep node (value after the loop)
            acc_node = _IRNode(acc_name, frozenset())
            acc_node.is_acc = True
            return self._cache(model, acc_node, extra=[red_node])

        # ----- GENERIC MODEL NODE ---------------------------------------
        fn     = model.fn

        # recursively compile children first
        args = [self._build_dag(a) for a in model.args]
        # every time we reference a child, bump its user-count
        for child in args:
            child.users += 1

        deps   = frozenset().union(*(a.deps for a in args))
        # build RHS string
        fn_name = getattr(fn, "__name__", None)
        if fn_name in {"add","radd","sub","rsub","mul","rmul",
                       "div","rdiv","truediv",
                       "pow","rpow"} and len(args) == 2:
            a, b = args
            if fn_name.startswith("r"):  # right-assoc helpers
                a, b = b, a
            op = {"add":"+","sub":"-","mul":"*","div":"/","truediv":"/",
                  "pow":"**"}[fn_name.replace("r","")]
            expr = f"({a.expr} {op} {b.expr})"
        elif fn_name == "neg" and len(args) == 1:
            expr = f"-({args[0].expr})"
        elif fn in {np.exp, np.sqrt, np.sum, np.dot}:
            expr = f"np.{fn.__name__}(" + ", ".join(a.expr for a in args) + ")"
        else:
            # fallback: hoist callable into a helper
            helper = self._hoist_fn(fn)
            expr   = f"{helper}(" + ", ".join(a.expr for a in args) + ")"

        node = _IRNode(expr, deps)
        node.model = model
        node.is_leaf = False
        return self._cache(model, node)

    # ------------------------------------------------------------------ #
    def _cache(self, model, node: _IRNode, extra=None):
        """Insert node into map, bump users of children."""
        self._node_map[model] = node
        if extra:                       # reductions produce extra IR nodes
            for e in extra:
                self._globals.append(e) # added as globals for now
        return node


    # ------------------------------------------------------------------ #
    #  2.  Partition nodes into global vs loop buckets
    # ------------------------------------------------------------------ #
    def _partition_nodes(self) -> None:
        """Partition IR nodes into global and loop-local buckets.

        Nodes with no INDEX dependencies go to ``_globals``. Nodes that
        depend on exactly one INDEX go to that loop's ``_LoopIR`` bucket.

        Raises
        ------
        NotImplementedError
            If a node depends on multiple INDEX parameters (nested loops).
        """
        # start fresh in case compile() is called twice
        prev_globals = list(self._globals)   # ← keep reduction bodies
        self._globals = []
        self._loops.clear()

        # Collect *all* nodes (the DAG proper + reduction-body extras)
        all_nodes = set(self._node_map.values()).union(prev_globals)

        for node in all_nodes:
            if not node.deps:
                self._globals.append(node)
            else:
                if len(node.deps) > 1:
                    raise NotImplementedError("nested index loops not yet supported")
                idx = next(iter(node.deps))
                bucket = self._loops.setdefault(idx, _LoopIR(idx))
                bucket.body.append(node)

        # remember the IR node that represents the whole model
        self._final = self._node_map[self.root]


    # ------------------------------------------------------------------ #
    #  3.  Allocate temporaries (multi-use nodes get names)
    # ------------------------------------------------------------------ #
    def _allocate_temps(self) -> None:
        """Assign temporary variable names to multi-use nodes.

        Nodes referenced more than once get a ``_tN`` name to avoid
        redundant computation. Leaf nodes and accumulator holders
        are skipped.
        """
        t_counter = 0
        for node in self._globals + sum((b.body for b in self._loops.values()), []):
            if (node.users >= 2 and
                not getattr(node, "is_leaf", False) and
                not getattr(node, "is_acc", False)) or node.is_red:
                node.name = f"_t{t_counter}"
                t_counter += 1


    def _alias_rhs(self, rhs, alias_map):
        """Replace any known sub-expression with its temp name."""
        for original, temp in sorted(alias_map.items(), key=lambda kv: -len(kv[0])):
            rhs = rhs.replace(original, temp)
        return rhs

    def _init_for_shape(self, obj, alias: dict) -> str:
        """
        Return the source-code string for an accumulator initialiser
        corresponding to `shape_obj`, where
           shape_obj == ()          -> "0.0"
           shape_obj is tuple       -> f"np.zeros({shape_obj})"
           otherwise (Model)        -> f"np.zeros_like({expr})"
        """
        shape_obj = obj.shape()
        if shape_obj == ():
            return "0.0"
        if isinstance(shape_obj, tuple):
            return f"np.zeros({shape_obj})"
        # shape_obj is a Model (symbolic). Use its aliased expression
        if shape_obj is INPUT:
            return f"np.zeros_like(x, dtype=np.float64)"
        raise ValueError(f"Cannot determine shape for {obj}")

    # ------------------------------------------------------------------ #
    #  4.  Emit python source string
    # ------------------------------------------------------------------ #
    def _emit_source(self) -> tuple[str, dict]:
        """Generate Python source code for the compiled function.

        Produces a function definition string and a namespace dict
        containing hoisted constants and helper functions.

        Returns
        -------
        tuple[str, dict]
            (source_code, namespace) ready for ``exec()`` and ``@njit``.
        """
        lines = ["def compiled(x, indices, theta):"]
        ns    = {"np": np}
        ns.update(self._hoisted_fns)
        ns.update(self._hoisted_consts)

        # ---- globals ---------------------------------------------------
        alias        = {}
        post_loop    = []          # IR nodes that use any _acc_*
        acc_names    = {acc for b in self._loops.values()
                             for n in b.body if n.is_red
                             for acc, _ in (n.is_red,)}

        for node in self._globals:
            if getattr(node, "is_acc", False):
                continue                         # accumulator holder
            if any(a in node.expr for a in acc_names):
                post_loop.append(node)           # delay until after loop(s)
                continue
            if node.name:
                rhs = self._alias_rhs(node.expr, alias)
                lines.append(f"    {node.name} = {rhs}")
                alias[node.expr] = node.name

        # ---- loops -----------------------------------------------------
        for loop in self._loops.values():
            idx = self._idx_var(loop.idx)
            r   = loop.idx.range
            lines.append(f"    # --- reduction over {idx} ---")

            # initialisers
            for n in loop.body:
                if n.is_red:
                    acc, _ = n.is_red
                    # pick any rhs that gives the correct output shape
                    shape_obj = n.model          # tuple, (), or Model
                    init      = self._init_for_shape(shape_obj, alias)
                    lines.append(f"    {acc} = {init}")

            # -------- main loop ----------------------------------------
            lines.append(f"    for {idx} in range({r.start}, {r.stop}, {r.step}):")
            for n in loop.body:
                rhs = self._alias_rhs(n.expr, alias)
                if n.name:
                    lines.append(f"        {n.name} = {rhs}")
                    alias[n.expr] = n.name
                    src = n.name
                else:
                    src = rhs
                if n.is_red:
                    acc, op = n.is_red
                    lines.append(f"        {acc} {op}= {src}")
            lines.append("")

        # ---- post-loop -------------------------------------------------
        for node in post_loop:
            rhs = self._alias_rhs(node.expr, alias)
            if node.name:
                lines.append(f"    {node.name} = {rhs}")
                alias[node.expr] = node.name

        # ---- return ----------------------------------------------------
        ret_expr = self._final.name or self._alias_rhs(self._final.expr, alias)
        lines.append(f"    return {ret_expr}")
        lines.append("")
        return "\n".join(lines), ns

    # ------------------------------------------------------------------ #
    #  Helpers: hoist callables & numpy arrays
    # ------------------------------------------------------------------ #
    def _hoist_fn(self, fn) -> str:
        """Hoist a callable into the generated namespace.

        Wraps the function with ``@njit`` and stores it in the namespace
        under a unique name for use in generated code.

        Parameters
        ----------
        fn : callable
            The function to hoist.

        Returns
        -------
        str
            The namespace key (e.g., ``_fn_0``) to reference this function.
        """
        for name, f in self._hoisted_fns.items():
            if f is fn:
                return name
        name = f"_fn_{len(self._hoisted_fns)}"
        self._hoisted_fns[name] = njit(fn)
        return name

def _compile(self, eager: bool = True) -> Model:
    """Compile this Model for fast evaluation using Numba.

    Checks the global cache for an existing compiled function with
    the same structure. If not found, compiles and caches the result.
    Uses LRU eviction when the cache is full.

    Parameters
    ----------
    eager : bool, default True
        If True, trigger LLVM compilation immediately by calling the
        function once. This makes the first real call fast but adds
        upfront cost. Set to False for lazy JIT behavior.

    Returns
    -------
    Model
        Returns self (for method chaining).

    Notes
    -----
    After compilation, ``model.code`` contains the generated source
    and ``model.compiled`` returns True.
    """
    current_hash = hash(self)  # Hash depends on symbolic structure
    if current_hash not in Model._compiled_cache:
        # Evict oldest entry if cache is full (LRU)
        while len(Model._compiled_cache) >= Model._cache_limit:
            Model._compiled_cache.popitem(last=False)
        c = Compiler(self)
        compiled_fn = c.compile()
        Model._compiled_cache[current_hash] = compiled_fn
        self.code = c.code

        # Trigger LLVM compilation by calling once with dummy values
        if eager:
            params = self.params
            theta = np.array([p.value for p in params]) if params else None

            # Check if model needs INPUT
            free = self.free
            needs_input = any(p.kind == _Param.input for p in free)
            x = np.array([0.0]) if needs_input else None

            # Check for INDEX params
            index_params = [p for p in free if p.kind == _Param.index]
            indices = [0] * len(index_params) if index_params else None

            try:
                compiled_fn(x, indices, theta)
            except Exception:
                pass  # Ignore errors during warmup (e.g., domain errors)

    return self


Model.compile = _compile
