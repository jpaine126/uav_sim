import numpy as np
from scipy.integrate import solve_ivp

from ..core.plotting import animate_airframe, plot_state
from ..core.state import Control, State
from . import airframe
from .find_trim import find_trim


def run_sim_open_loop(initial_state: State, control: Control):
    t = np.arange(0, 10, 0.01)

    def wrapper(t, y):
        state = State(t, y[0:3], y[3:6], y[6:9], y[9:12])
        forces, moments, *_ = airframe.forces_moments(
            state, control, wind=initial_wind
        )
        out = airframe.derivative(state, forces, moments)
        return out

    a = solve_ivp(
        wrapper,
        t_span=(t.min(), t.max()),
        y0=np.hstack(
            (initial_state.position, initial_state.velocity, initial_state.angle, initial_state.angle_rate)
        ),
        t_eval=t,
    )

    return a


if __name__ == "__main__":
    initial_wind = np.array([0, 0, 0, 0, 0, 0])

    trim_state, trim_control = find_trim(
        desired_airspeed=10.0,
        desired_fpa=np.deg2rad(0),
        desired_radius=20,
    )
    a = run_sim_open_loop(trim_state, trim_control)

    plot_state(a.t, a.y.T).show()

    body_animation = animate_airframe(a.t, a.y, airframe.body_vertices)

    body_animation.show()
