import numpy as np
from scipy.integrate import solve_ivp

from ..core.plotting import animate_airframe, plot_state
from ..core.state import Control, State
from ..plant.airframe import Airframe


def get_params():
    gravity = 9.8

    mass = 13.5
    Jx = 0.8244
    Jy = 1.135
    Jz = 1.759
    Jxz = 0  # 0.1204

    gamma = Jx * Jz - Jxz ** 2
    gamma_1 = (Jxz * (Jx - Jy + Jz)) / gamma
    gamma_2 = (Jz * (Jz - Jy) + Jxz ** 2) / gamma
    gamma_3 = Jz / gamma
    gamma_4 = Jxz / gamma
    gamma_5 = (Jz - Jx) / Jy
    gamma_6 = Jxz / Jy
    gamma_7 = ((Jx - Jy) * Jx + Jxz ** 2) / gamma
    gamma_8 = Jx / gamma

    p = type("A", tuple(), {})()

    for var in dir():
        if isinstance(locals()[var], (int, float)):
            setattr(p, var, locals()[var])

    return p


params = get_params()

body_vertices = [
    [10, 0, 0],
    [-10, 5, 0],
    [-10, -5, 0],
]

initial_position = np.array([0, 0, 0])
initial_velocity = np.array([0, 0, 0])

initial_angle = np.array([0, 0, 0])
initial_angle_rate = np.array([0, 0, 0])

initial_state = State(
    0, initial_position, initial_velocity, initial_angle, initial_angle_rate
)

airframe = Airframe(params, body_vertices)


def wrapper(t, y):
    state = State(t, y[0:3], y[3:6], y[6:9], y[9:12])
    forces = np.array([0, 0, 0])
    moments = np.array([1, 0, 0])
    out = airframe.derivative(state, forces, moments)

    return out


if __name__ == "__main__":

    t = np.arange(0, 10, 0.001)

    a = solve_ivp(
        wrapper,
        t_span=(t.min(), t.max()),
        y0=np.hstack(
            (initial_position, initial_velocity, initial_angle, initial_angle_rate)
        ),
        t_eval=t,
    )

    plot_state(a.t, a.y).show()

    body_animation = animate_airframe(a.t, a.y, airframe.body_vertices)

    body_animation.show()
