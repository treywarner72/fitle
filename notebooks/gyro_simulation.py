"""
Boris Pusher as a Model
=======================

State: [x0, x1, x2, v0, v1, v2, t]
step: state -> state
"""

import numpy as np
from fitle import INPUT, const, vector, concat, Model


def iterate(model, n):
    """Apply a state->state Model n times without expanding the tree."""
    def loop(state):
        s = state
        for _ in range(n):
            s = (model % s)()
        return s
    loop.__name__ = f'iterate[{n}]'
    return Model(loop, [INPUT])

# State - select elements from INPUT
x0, x1, x2 = INPUT[0], INPUT[1], INPUT[2]
v0, v1, v2 = INPUT[3], INPUT[4], INPUT[5]
t = INPUT[6]
x, v = vector(x0, x1, x2), vector(v0, v1, v2)

# Fields
B = const(np.array([0., 0., 1.]))
E = const(np.array([0., 0., 0.]))

# Boris
qom, dt, h = 1.0, 0.1, 0.05

v_m = v + h * E
t_vec = h * B
s_vec = 2 * t_vec / (1 + np.sum(t_vec * t_vec))
v_p = v_m + np.cross(v_m, t_vec)
v_plus = v_m + np.cross(v_p, s_vec)
v_new = v_plus + h * E
x_new = x + dt * v_new
t_new = t + dt

step = concat(x_new, v_new, vector(t_new))

# step % state -> next state
# iterate(step, n) applies step n times without expanding tree
run = iterate(step, 314)  # ~5 orbits
print(run.compile().code)
