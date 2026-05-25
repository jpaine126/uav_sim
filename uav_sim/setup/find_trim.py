"""Compute trim conditions for steady state flight given a desired air speed and
flight path angle using a numerical optimizer over the full trim state."""


import numpy as np
from scipy.optimize import minimize

from ..core.state import Control, State
from ..plant.airframe import Airframe
from . import body_vertices, params
from ..core.utilities import rotate_body_to_inertial

airframe = Airframe(params, body_vertices)


def _compute_initial_guess(Va: float, gamma: float, R: float):
    """Compute a reasonable initial guess for the trim optimizer
    based on approximate aerodynamic equilibrium."""
    P = params

    # Desired CL for lift = weight * cos(gamma)
    CL_eq = P.mass * P.gravity * np.cos(gamma) / (
        0.5 * P.rho * Va ** 2 * P.S_wing
    )

    # Solve CL(alpha) = CL_eq for small alpha where sigma ≈ 0
    # CL ≈ C_L_0 + C_L_alpha * alpha_deg  (alpha in degrees)
    alpha_deg_guess = (CL_eq - P.C_L_0) / P.C_L_alpha
    alpha_guess = np.deg2rad(alpha_deg_guess)

    # Bank angle for coordinated turn
    if np.isinf(R):
        phi_guess = 0.0
    else:
        phi_guess = np.arctan(Va ** 2 / (P.gravity * R))

    # Approximate pitching moment balance -> elevator
    theta_guess = alpha_guess - gamma
    delta_e_guess = -(P.C_m_0 + P.C_m_alpha * alpha_guess) / P.C_m_delta_e

    # Approximate drag polar for thrust balance
    CD_approx = P.C_D_p + CL_eq ** 2 / (np.pi * P.e * P.AR)
    drag = 0.5 * P.rho * Va ** 2 * P.S_wing * CD_approx
    # Thrust must also counter the along-track gravity component
    thrust_needed = drag + P.mass * P.gravity * np.sin(gamma)
    if thrust_needed > 0:
        # T = 0.5 * rho * S_prop * C_prop * (k_motor * delta_t)^2
        # => delta_t = sqrt(2*T / (rho * S_prop * C_prop * k_motor^2))
        denom = 0.5 * P.rho * P.S_prop * P.C_prop * (P.k_motor ** 2)
        delta_t_guess = np.sqrt(thrust_needed / denom)
    else:
        delta_t_guess = 0.0

    # Rudder, aileron start at zero
    return np.array([
        alpha_guess, 0.0, phi_guess, delta_e_guess, 0.0, 0.0, delta_t_guess,
    ])


def find_trim(
    desired_airspeed: float,
    desired_fpa: float = 0.0,
    desired_radius: float = np.inf,
    x0: np.ndarray = None,
):
    """Find a steady-state trim condition for the aircraft.

    The optimizer searches over 7 variables simultaneously:
        [alpha, beta, phi, delta_e, delta_a, delta_r, delta_t]

    A trim condition satisfies:
        * Constant airspeed  ->  ||velocity|| = desired_airspeed
        * Steady flight path ->  pd_dot = Va * sin(gamma)
        * Coordinated turn   ->  body rates match the kinematic turn rates
        * Force / moment balance ->  u_dot = v_dot = w_dot = p_dot = q_dot = r_dot = 0

    Args:
        desired_airspeed: True airspeed in m/s.
        desired_fpa: Desired flight-path angle in radians.
                     Positive = below the horizontal plane (descending in NED).
        desired_radius: Turn radius in metres. np.inf for straight flight.
        x0: Optional initial guess for the 7 trim variables.
            Defaults to a reasonable straight-and-level guess.

    Returns:
        trim_state (State): Trim state with concrete yaw = 0 and position = [0,0,0].
        trim_control (Control): Corresponding control surface deflections.
    """
    P = params
    Va = desired_airspeed
    gamma = desired_fpa
    R = desired_radius

    # ------------------------------------------------------------------
    # Initial guess
    # ------------------------------------------------------------------
    if x0 is None:
        x0 = _compute_initial_guess(Va, gamma, R)

    def cost(x):
        alpha, beta, phi, delta_e, delta_a, delta_r, delta_t = x
        theta = alpha - gamma

        # Body-frame velocity components from Va, alpha, beta
        u = Va * np.cos(alpha) * np.cos(beta)
        v = Va * np.sin(beta)
        w = Va * np.sin(alpha) * np.cos(beta)

        # Kinematic turn rates for the desired radius
        yaw_dot = Va / R if not np.isinf(R) else 0.0
        p_des = -yaw_dot * np.sin(theta)
        q_des = yaw_dot * np.sin(phi) * np.cos(theta)
        r_des = yaw_dot * np.cos(phi) * np.cos(theta)

        state = State(
            time=0.0,
            position=np.zeros(3),
            velocity=np.array([u, v, w]),
            angle=np.array([phi, theta, 0.0]),
            angle_rate=np.array([p_des, q_des, r_des]),
        )
        control = Control(delta_e, delta_r, delta_a, delta_t)

        forces, moments, Va_calc, alpha_calc, beta_calc = airframe.forces_moments(
            state, control, airspeed=Va, alpha=alpha, beta=beta
        )
        derivatives = airframe.derivative(state, forces, moments)

        # In trim we want:
        #   u_dot = v_dot = w_dot = 0   (no translational acceleration)
        #   p_dot = q_dot = r_dot = 0   (no rotational acceleration)
        # The position / angle kinematics will naturally follow the flight path
        accel_sq = np.sum(derivatives[3:6] ** 2)  # u_dot, v_dot, w_dot
        ang_accel_sq = np.sum(derivatives[9:12] ** 2)  # p_dot, q_dot, r_dot

        # Also penalize deviation from the kinematic angle rates
        # For a straight flight these should be zero; for a turn they are
        # already baked in via p_des/q_des/r_des, so their derivatives
        # should still be driven to zero by the ang_accel_sq term above
        return float(accel_sq + ang_accel_sq)

    # Variable bounds
    bounds = [
        (-np.deg2rad(15), np.deg2rad(20)),  # alpha
        (-np.deg2rad(10), np.deg2rad(10)),  # beta
        (-np.deg2rad(45), np.deg2rad(45)),  # phi
        (-1.0, 1.0),                        # delta_e
        (-1.0, 1.0),                        # delta_a
        (-1.0, 1.0),                        # delta_r
        (0.0, 1.5),                         # delta_t
    ]

    result = minimize(
        cost,
        x0,
        method="SLSQP",
        bounds=bounds,
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Trim optimisation failed: {result.message}")

    alpha, beta, phi, delta_e, delta_a, delta_r, delta_t = result.x
    theta = alpha - gamma

    # Final state
    u = Va * np.cos(alpha) * np.cos(beta)
    v = Va * np.sin(beta)
    w = Va * np.sin(alpha) * np.cos(beta)

    yaw_dot = Va / R if not np.isinf(R) else 0.0
    p_des = -yaw_dot * np.sin(theta)
    q_des = yaw_dot * np.sin(phi) * np.cos(theta)
    r_des = yaw_dot * np.cos(phi) * np.cos(theta)

    trim_state = State(
        time=0.0,
        position=np.zeros(3),
        velocity=np.array([u, v, w]),
        angle=np.array([phi, theta, 0.0]),
        angle_rate=np.array([p_des, q_des, r_des]),
    )
    trim_control = Control(delta_e, delta_r, delta_a, delta_t)

    return trim_state, trim_control


if __name__ == "__main__":
    # Example: straight-and-level trim at 10 m/s
    trim_state, trim_control = find_trim(
        desired_airspeed=10.0,
        desired_fpa=np.deg2rad(0),
        desired_radius=np.inf,
    )

    u, v, w = trim_state.velocity
    Va_calc = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    alpha_calc = np.arctan2(w, u)
    beta_calc = np.arcsin(v / Va_calc)

    forces, moments, *_ = airframe.forces_moments(
        trim_state, trim_control, airspeed=Va_calc, alpha=alpha_calc, beta=beta_calc
    )
    derivatives = airframe.derivative(trim_state, forces, moments)

    print("Trim state:")
    print(f"  velocity: {trim_state.velocity}")
    print(f"  angle:    {trim_state.angle}")
    print(f"  angle_rate: {trim_state.angle_rate}")
    print()
    print(f"Trim control: {trim_control}")
    print()
    print(f"Derivatives: {derivatives}")
    print(f"Derivative norm: {np.linalg.norm(derivatives):.6e}")
