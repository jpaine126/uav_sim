import numpy as np
from scipy.integrate import solve_ivp

from ..core.plotting import animate_airframe, plot_state
from ..core.state import Control, State
from ..plant.airframe import Airframe


def get_params():
    gravity = 9.8

    mass = 1.56
    Jx = 0.1147
    Jy = 0.0576
    Jz = 0.1712
    Jxz = 0.0015
    # aerodynamic coefficients
    S_wing = 0.2589
    b = 1.4224
    c = 0.3302
    S_prop = 0.0314
    rho = 1.2682
    k_motor = 20
    k_T_P = 0
    k_Omega = 0
    e = 0.9
    AR = b ** 2 / S_wing

    C_L_0 = 0.28
    C_L_alpha = 3.45
    C_L_q = 0.0
    C_L_delta_e = -0.36
    C_D_0 = 0.03
    C_D_alpha = 0.30
    C_D_p = 0.0437
    C_D_q = 0.0
    C_D_delta_e = 0.0
    C_m_0 = -0.02338
    C_m_alpha = -0.38
    C_m_q = -3.6
    C_m_delta_e = -0.5
    C_Y_0 = 0.0
    C_Y_beta = -0.98
    C_Y_p = 0.0
    C_Y_r = 0.0
    C_Y_delta_a = 0.0
    C_Y_delta_r = -0.17
    C_ell_0 = 0.0
    C_ell_beta = -0.12
    C_ell_p = -0.26
    C_ell_r = 0.14
    C_ell_delta_a = 0.08
    C_ell_delta_r = 0.105
    C_n_0 = 0.0
    C_n_beta = 0.25
    C_n_p = 0.022
    C_n_r = -0.35
    C_n_delta_a = 0.06
    C_n_delta_r = -0.032
    C_prop = 1.0
    M = 50
    epsilon = 0.1592
    alpha0 = 0.4712

    gamma = Jx * Jz - Jxz ** 2
    gamma_1 = (Jxz * (Jx - Jy + Jz)) / gamma
    gamma_2 = (Jz * (Jz - Jy) + Jxz ** 2) / gamma
    gamma_3 = Jz / gamma
    gamma_4 = Jxz / gamma
    gamma_5 = (Jz - Jx) / Jy
    gamma_6 = Jxz / Jy
    gamma_7 = ((Jx - Jy) * Jx + Jxz ** 2) / gamma
    gamma_8 = Jx / gamma

    C_p_0 = gamma_3 * C_ell_0 + gamma_4 * C_n_0
    C_p_beta = gamma_3 * C_ell_beta + gamma_4 * C_n_beta
    C_p_p = gamma_3 * C_ell_p + gamma_4 * C_n_p
    C_p_r = gamma_3 * C_ell_r + gamma_4 * C_n_r
    C_p_delta_a = gamma_3 * C_ell_delta_a + gamma_4 * C_n_delta_a
    C_p_delta_r = gamma_3 * C_ell_delta_r + gamma_4 * C_n_delta_r
    C_r_0 = gamma_4 * C_ell_0 + gamma_8 * C_n_0
    C_r_beta = gamma_4 * C_ell_beta + gamma_8 * C_n_beta
    C_r_p = gamma_4 * C_ell_p + gamma_8 * C_n_p
    C_r_r = gamma_4 * C_ell_r + gamma_8 * C_n_r
    C_r_delta_a = gamma_4 * C_ell_delta_a + gamma_8 * C_n_delta_a
    C_r_delta_r = gamma_4 * C_ell_delta_r + gamma_8 * C_n_delta_r

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

initial_position = np.array([0, 0, -100])
initial_velocity = np.array([10, 0, 0])

initial_angle = np.array([0, 0, 0])
initial_angle_rate = np.array([0, 0, 0])

initial_state = State(
    time=0,
    position=initial_position,
    velocity=np.array([9.86075615e00, 1.37564863e-03, 1.66297511e00]),
    angle=np.array([4.18114984e-05, 1.67073701e-01, 0]),
    angle_rate=np.array([-0.0, -0.0, 0.0]),
)

initial_control = Control(
    delta_e=-0.17373601241098888,
    delta_r=0.00042232102644198274,
    delta_a=-0.00034794905156488345,
    delta_t=0.6407080257401786,
)

initial_wind = np.array([0, 0, 0, 0, 0, 0])

airframe = Airframe(params, body_vertices)


def wrapper(t, y):
    state = State(t, y[0:3], y[3:6], y[6:9], y[9:12])
    forces, moments, *_ = airframe.forces_moments(
        state, initial_control, wind=initial_wind
    )
    out = airframe.derivative(state, forces, moments)

    return out


if __name__ == "__main__":

    t = np.arange(0, 20, 0.01)

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
