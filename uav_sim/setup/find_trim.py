"""Compute trim conditions for steady state flight given a desired air speed and
flight path angle."""


import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ..core.state import Control, State
from ..plant.airframe import Airframe
from .test_sim import body_vertices, params

initial_wind = np.array([0, 0, 0, 0, 0, 0])

airframe = Airframe(params, body_vertices)


def get_desired_derivtives(airspeed: float, flight_path_angle: float, radius: float):
    return np.array(
        [
            0,
            0,
            airspeed * np.sin(flight_path_angle),
            0,
            0,
            0,
            0,
            0,
            (airspeed / radius) * np.cos(flight_path_angle),
            0,
            0,
            0,
        ]
    )


def compute_cost_function(x, airspeed, fpa, radius):
    alpha, beta, phi = x
    # breakpoint()

    trim_state, trim_control = airframe.trimmed_output(
        airspeed, fpa, radius, alpha, beta, phi
    )
    forces, moments, *_ = airframe.forces_moments(
        trim_state, trim_control, airspeed=airspeed, alpha=alpha, beta=beta
    )

    derivatives = airframe.derivative(trim_state, forces, moments)

    # get cost function value
    desired_derivatives = get_desired_derivtives(
        desired_airspeed, desired_fpa, desired_radius
    )
    J = np.linalg.norm(desired_derivatives[2:] - derivatives[2:]) ** 2
    if np.isnan(J):
        breakpoint()
    return J


if __name__ == "__main__":
    desired_airspeed = 10
    desired_fpa = np.deg2rad(0)
    desired_radius = np.inf

    # alpha, beta, phi
    x0 = np.array([0, 0, 0])

    a = minimize(
        compute_cost_function,
        x0,
        method="CG",
        args=(desired_airspeed, desired_fpa, desired_radius),
        options=dict(
            # gtol=1e-15,
            eps=-1e-3,
        ),
    )

    # recalculate controls using optimized params
    alpha, beta, phi = a.x
    trim_state, trim_control = airframe.trimmed_output(
        desired_airspeed, desired_fpa, desired_radius, alpha, beta, phi
    )
    forces, moments, *_ = airframe.forces_moments(
        trim_state, trim_control, airspeed=desired_airspeed, alpha=alpha, beta=beta
    )

    derivatives = airframe.derivative(trim_state, forces, moments)

    print(f"{alpha=} {beta=} {phi=}")
    print(f"{trim_state=}")
    print(f"{trim_control}")
    print(f"Final cost function value = {a.fun}")
