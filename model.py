import numpy as np
from numba import njit, types
from numba.typed import Dict
from .param import Param, P, Q, U, X, I
from collections.abc import Iterable
from collections import defaultdict
import operator
import functools

class Model:
    _compiled_cache = {} # Class-level cache for compiled functions
    
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        self.memory = {}

    @property
    def params(self):
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
            elif isinstance(m, Param) and  m.kind not in (Param.input, Param.index):
                if m not in seen:
                    seen.add(m)
                    flat.append(m)

        walk(self)
        return flat

    @property
    def free(self):
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
            elif isinstance(m, Param) and m.kind in (Param.input, Param.index):
                if m not in seen:
                    seen.add(m)
                    flat.append(m)

        walk(self)
        return flat



    def __hash__(self):
        def walk(m):
            if isinstance(m, Param):
                #if m.kind == Param.input:
                #    return ("INPUT")
                #else:
                #    return (m.kind, id(m))
                return ("param", m.kind)
            elif isinstance(m, Reduction):
                return ("reduce", walk(m.model), hash(m.index), m.op.__name__)
            elif isinstance(m, Model):
                fn_id = getattr(m.fn, "__name__", repr(m.fn))
                return (fn_id, tuple(walk(arg) for arg in m.args))
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
        return hash(walk(self))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Model):
            return False

        def walk(a, b):
            if isinstance(a, Param) and isinstance(b, Param):
                if a.kind != b.kind:
                    return False
                if a.kind != Param.input:
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

    def eval_py(self, x, index_map):
        def _eval(m):
            if isinstance(m, Reduction):
                index = m.index
                bound = index.range
                return functools.reduce(
                    m.op,
                    [_eval(m.model % {index: i}) for i in bound]
                )

            if isinstance(m, Model):
                return m.fn(*[_eval(arg) for arg in m.args])
            if isinstance(m, Param):
                if m.kind == Param.theta:
                    return m.value
                if m is X:
                    return x
                if m.kind == Param.index:
                    return index_map[m]
                raise Exception(f"unknown param kind: {m}")
            return m

        return _eval(self)


    def eval(self, x, index_map):
        if not self.params:
            theta = None
        else:
            theta = np.array([p.value for p in self.params])

        index_params = [p for p in self.free if p.kind == Param.index]
        index_list = [index_map[p] for p in index_params] if index_params else None

        current_hash = hash(self) # Hash depends on symbolic structure
        if current_hash in Model._compiled_cache:
            return Model._compiled_cache[current_hash](x, index_list, theta)
        else:
            return self.eval_py(x, index_list)

    def __call__(self, x=None):
        if X in self.free and x is None: raise Exception("Cannot __call__() with free X")
        if X not in self.free and x is not None: raise Exception("Cannot __call__(x) without free X")
        if not self.free:
            return self.eval(None, {})
        if self.free == [X]:
            return self.eval(x, {})
        else:
            return self % {X: x}
# ops

    def __getitem__(self, map):
        if X in self.free:
            if not isinstance(map, dict):
                return self % {I: map}
            return self % map
        return self.eval(None, map)

    def copy_with_args(self, args):
        return type(self)(self.fn, args)

    def __mod__(self, subs):
        if not isinstance(subs, dict):
            subs = {X : subs}
        def walk(m):
            if isinstance(m, Model):
                return m.copy_with_args([walk(a) for a in m.args])
            elif isinstance(m, Iterable) and not isinstance(m, (str, bytes, np.ndarray)):
                return type(m)(walk(x) for x in m)
            elif isinstance(m, Param) and m in subs:
                return subs[m]
            return m

        return walk(self)



#REPRESENTATION

    def __repr__(self):
        fn_name = getattr(self.fn, '__name__', 'λ')
        if fn_name == '<lambda>':
            fn_name = 'λ'
        arg_strs = [repr(arg) for arg in self.args]
        return f"Model<{fn_name}>({', '.join(arg_strs)})"

    def __str__(self):
        return self._format()

    def _format(self, parent_prec=0, side=None):
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
                l_op, l_prec, _ = op_info.get(getattr(left.fn, '__name__', ''), ('', 0, ''))
                if parens_needed(prec, assoc, l_prec, True):
                    lstr = f'({lstr})'
            if isinstance(right, Model):
                r_op, r_prec, _ = op_info.get(getattr(right.fn, '__name__', ''), ('', 0, ''))
                if parens_needed(prec, assoc, r_prec, False):
                    rstr = f'({rstr})'

            return f"{lstr} {op} {rstr}"

        elif fn_name == 'neg' and len(self.args) == 1:
            arg = self.args[0]
            astr = arg._format(4) if isinstance(arg, Model) else str(arg)
            return f"-{astr}"

        arg_strs = [arg._format() if isinstance(arg, Model) else str(arg) for arg in self.args]
        return f"{fn_name}({', '.join(arg_strs)})"

    def freeze(self):
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
            return Model(fn, [self, other]) if 'r' not in name else Model(fn, [other, self])
        setattr(cls, name, op_method)

bind_ops(Model)
bind_ops(Param)


# Reduction

class Reduction(Model):
    def __init__(self, model, index_param, op=operator.add):
        assert isinstance(index_param, Param), "must reduce over a Param"
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

def identity(x): return x

def const(val):
    f = lambda val=val: val
    f.__name__ = f"const_{id(val)}"
    return Model(f, [])

def indecise(obj, index = I):
    if isinstance(obj, (list, tuple)):
        entries = [e if isinstance(e, (Model, Param)) else const(e) for e in obj]
        fn = lambda entries, i: entries[int(i)]
        fn.__name__ = 'indecise'
        return Model(fn, [entries, index])
    else:
        fn = lambda arr, i: arr[int(i)]
        fn.__name__ = 'indecise'
        return Model(fn, [obj, index])



# gradient

def _zero(): return 0
def _one(): return 1

def _grad_fn(model, wrt):
    if isinstance(model, Param):
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
        return 0

    fn = getattr(model.fn, '__name__', None)
    args = model.args

    # unary ops
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
        if isinstance(exp, (Model, Param)) and not base is exp:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        elif isinstance(exp, Model) and exp.params:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        else:
            # constant exponent
            return exp * (base ** (exp - 1)) * df

    if fn == 'rpow':
        exp, base = args
        df = _grad_fn(base, wrt)
        if isinstance(exp, (Model, Param)) and not base is exp:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        elif isinstance(exp, Model) and exp.params:
            return model * (_grad_fn(exp, wrt) * np.log(base) + exp * df / base)
        else:
            return exp * (base ** (exp - 1)) * df

    if fn == 'indecise':
        entries, index = model.args
        if isinstance(entries, (list, tuple)):
            # differentiate each entry
            grads = []
            for e in entries:
                if isinstance(e, Model):
                    grads.append(_grad_fn(e, wrt))
                elif isinstance(e, Param):
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


    # fallback: assume zero
    raise Exception(f"Cannot differentiate {fn}")

def _grad(self, wrt=None):
    if wrt is None:
        wrt = self.params
    if isinstance(wrt, Param):
        return _grad_fn(self, wrt).simplify()
    if isinstance(wrt, (list, tuple)):
        return Model(lambda *args: np.array(args), [_grad_fn(self, p).simplify() for p in wrt])
    raise TypeError(f"grad expects Param, list of Param, or None, got {type(wrt)}")


Model.grad = _grad

def identity(x): return x

def ensure_model(x):
    return x if isinstance(x, Model) else Model(identity, [x])

def simplify(m):
    if not isinstance(m, Model):
        return m

    if isinstance(m, Reduction):
        return Reduction(simplify(m.model), m.index, m.op)

    fn = getattr(m.fn, '__name__', None)
    args = [simplify(arg) for arg in m.args]

    def is_zero(x): return x == 0
    def is_one(x): return x == 1

    if fn in ('mul', 'rmul'):
        if any(is_zero(arg) for arg in args):
            return 0
        args = [arg for arg in args if not is_one(arg)]
        if len(args) == 0:
            return 0
        if len(args) == 1:
            return ensure_model(args[0])
        return Model(m.fn, args)

    if fn in ('add', 'radd'):
        args = [arg for arg in args if not is_zero(arg)]
        if len(args) == 0:
            return 0
        if len(args) == 1:
            return ensure_model(args[0])
        return Model(m.fn, args)

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


def shape(obj):
    if isinstance(obj, (int, float, np.number)):
        return ()

    if isinstance(obj, np.ndarray):
        return obj.shape

    if obj is X:
        return X                       # e.g. you called shape(X) directly

    if isinstance(obj, Param):
        return ()                      # θ_k etc. are scalars

    if isinstance(obj, Reduction):
        return shape(obj.model)        # reduction never *adds* new shape

    if isinstance(obj, Model):
        fn_name = getattr(obj.fn, '__name__', '')

        # indecise always returns a scalar for a single index lookup
        if fn_name == 'indecise':
            return ()

        # 1. if expression explicitly depends on free X → vector
        if X in obj.free:
            return X

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
