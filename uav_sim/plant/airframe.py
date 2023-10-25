from typing import Optional

import numpy as np

from ..core.abc import Dynamic
from ..core.state import Control, State


def CD(alpha, P):
    return P.C_D_p + (P.C_L_0 + P.C_L_alpha * alpha) ** 2 / (np.pi * P.e * P.AR)


def CL(alpha, P):
    return (1 - sigma(alpha, P)) * (P.C_L_0 + P.C_L_alpha * alpha) + sigma(alpha, P) * (
        2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
    )


def sigma(alpha, P):
    return (
        1 + np.exp(-P.M * (alpha - P.alpha0)) + np.exp(P.M * (alpha + P.alpha0))
    ) / (
        (1 + np.exp(-P.M * (alpha - P.alpha0))) * (1 + np.exp(P.M * (alpha + P.alpha0)))
    )


def CX(alpha, P):
    return -CD(alpha, P) * np.cos(alpha) + CL(alpha, P) * np.sin(alpha)


def CXq(alpha, P):
    return -P.C_D_q * np.cos(alpha) + P.C_L_q * np.sin(alpha)


def CXde(alpha, P):
    return -P.C_D_delta_e * np.cos(alpha) + P.C_L_delta_e * np.sin(alpha)


def CZ(alpha, P):
    return -CD(alpha, P) * np.sin(alpha) - CL(alpha, P) * np.cos(alpha)


def CZq(alpha, P):
    return -P.C_D_q * np.sin(alpha) - P.C_L_q * np.cos(alpha)


def CZde(alpha, P):
    return -P.C_D_delta_e * np.sin(alpha) - P.C_L_delta_e * np.cos(alpha)


class Airframe(Dynamic):
    """Class to contain the constants and equations representing an airframe."""

    def __init__(self, P, body_vertices):
        self.P = P
        self.body_vertices = body_vertices

    def update(self):
        ...

    def derivative(self, x: State, forces, moments):
        """Calculate the derivatives of the airframe equations of motion."""
        P = self.P
        x_dot = np.zeros((12,))

        # relabel the inputs
        pn = x.position[0]
        pe = x.position[1]
        pd = x.position[2]
        u = x.velocity[0]
        v = x.velocity[1]
        w = x.velocity[2]
        phi = x.angle[0]
        theta = x.angle[1]
        psi = x.angle[2]
        p = x.angle_rate[0]
        q = x.angle_rate[1]
        r = x.angle_rate[2]

        fx = forces[0]
        fy = forces[1]
        fz = forces[2]
        ell = moments[0]
        m = moments[1]
        n = moments[2]

        ct = np.cos(theta)
        cu = np.cos(psi)
        cp = np.cos(phi)
        st = np.sin(theta)
        su = np.sin(psi)
        sp = np.sin(phi)
        tt = np.tan(theta)

        # p_n_dot, p_e_dot, p_d_dot
        R1 = np.array(
            [
                [ct * cu, sp * st * cu - cp * su, cp * st * cu + sp * su],
                [ct * su, sp * st * su + cp * cu, cp * st * su - sp * cu],
                [-st, sp * ct, cp * ct],
            ]
        )
        x_dot[0:3] = R1 @ x.velocity

        # u_dot, v_dot, w_dot
        x_dot[3:6] = np.array([r * v - q * w, p * w - r * u, q * u - p * v]) + (
            np.array([fx, fy, fz]) / P.mass
        )

        # phi_dot, theta_dot, upsilon_dot
        R2 = np.array([[1, sp * tt, cp * tt], [0, cp, -sp], [0, sp / ct, cp / ct]])
        x_dot[6:9] = R2 @ x.angle_rate

        # p_dot, q_dot, r_dot
        x_dot[9] = (
            P.gamma_1 * p * q - P.gamma_2 * q * r + P.gamma_3 * ell + P.gamma_4 * n
        )
        x_dot[10] = P.gamma_5 * p * r - P.gamma_6 * (p ** 2 - r ** 2) + m / P.Jy
        x_dot[11] = (
            P.gamma_7 * p * q - P.gamma_1 * q * r + P.gamma_4 * ell + P.gamma_8 * n
        )

        return x_dot

    def trimmed_output(
        self, airspeed, flight_path_angle, radius, attack, sideslip, roll
    ):
        """Compute trimmmed output for given conditions.
        
        Returns a State so that the output can be used in other airframe functions,
        but is not a complete state. Yaw and position are not actually calculated
        becuase they are not needed for trim calculations.
        """
        P = self.P
        u_star = airspeed * np.cos(attack) * np.cos(sideslip)
        v_star = airspeed * np.sin(sideslip)
        w_star = airspeed * np.sin(attack) * np.cos(sideslip)

        pitch = attack + flight_path_angle
        yaw = np.nan

        yaw_dot = airspeed / radius

        p_star = -yaw_dot * np.sin(pitch)
        q_star = yaw_dot * np.sin(roll) * np.cos(pitch)
        r_star = yaw_dot * np.cos(roll) * np.cos(pitch)

        state_star = State(
            time=np.nan,
            position=np.array([np.nan, np.nan, np.nan]),
            velocity=np.array([u_star, v_star, w_star]),
            angle=np.array([roll, pitch, yaw]),
            angle_rate=np.array([p_star, q_star, r_star]),
        )

        delta_e_star = (
            (
                (P.Jxz * (p_star ** 2 - r_star ** 2) + (P.Jx - P.Jz) * p_star * r_star)
                / (1 / 2)
                * P.rho
                * airspeed ** 2
                * P.c
                * P.S_wing
            )
            - P.C_m_0
            - P.C_m_alpha * attack
            - P.C_m_q * ((P.c * q_star) / 2 * airspeed)
        ) / P.C_m_delta_e

        delta_t_star = np.sqrt(
            np.abs(
                (
                    (
                        2
                        * P.mass
                        * (
                            -r_star * v_star
                            + q_star * w_star
                            + P.gravity * np.sin(pitch)
                        )
                        - (
                            P.rho
                            * (airspeed ** 2)
                            * P.S_wing
                            * (
                                CX(attack, P)
                                + CXq(attack, P) * ((P.c * q_star) / (2 * airspeed))
                                + CXde(attack, P) * delta_e_star
                            )
                        )
                    )
                    / (P.rho * P.S_prop * P.C_prop * (P.k_motor ** 2))
                )
                + ((airspeed ** 2) / (P.k_motor ** 2))
            )
        )
        if np.isnan(delta_t_star):
            ...  # delta_t_star = 0
            breakpoint()

        delta_a_star, delta_r_star = np.linalg.inv(
            np.array([[P.C_p_delta_a, P.C_p_delta_r], [P.C_r_delta_a, P.C_r_delta_r],])
        ) @ np.array(
            [
                (
                    (
                        (-P.gamma_1 * p_star * q_star + P.gamma_2 * q_star * r_star)
                        / (1 / 2)
                        * P.rho
                        * airspeed ** 2
                        * P.S_wing
                        * P.b
                    )
                    - P.C_p_0
                    - P.C_p_beta * sideslip
                    - P.C_p_p * (P.b * p_star / 2 * airspeed)
                    - P.C_p_r * (P.b * r_star / 2 * airspeed)
                ),
                (
                    (
                        (-P.gamma_7 * p_star * q_star + P.gamma_1 * q_star * r_star)
                        / (1 / 2)
                        * P.rho
                        * airspeed ** 2
                        * P.S_wing
                        * P.b
                    )
                    - P.C_r_0
                    - P.C_r_beta * sideslip
                    - P.C_r_p * (P.b * p_star / 2 * airspeed)
                    - P.C_r_r * (P.b * r_star / 2 * airspeed)
                ),
            ]
        )

        control_star = Control(delta_e_star, delta_r_star, delta_a_star, delta_t_star)

        return state_star, control_star

    def get_airspeed_alpha_beta(self, x: State, control: Control, wind: np.ndarray):
        P = self.P

        # relabel the inputs
        pn = x.position[0]
        pe = x.position[1]
        pd = x.position[2]
        u = x.velocity[0]
        v = x.velocity[1]
        w = x.velocity[2]
        phi = x.angle[0]
        theta = x.angle[1]
        psi = x.angle[2]
        p = x.angle_rate[0]
        q = x.angle_rate[1]
        r = x.angle_rate[2]
        delta_e = control.delta_e
        delta_a = control.delta_a
        delta_r = control.delta_r
        delta_t = control.delta_t
        w_ns = wind[0]  # steady wind - North
        w_es = wind[1]  # steady wind - East
        w_ds = wind[2]  # steady wind - Down
        u_wg = wind[3]  # gust along body x-axis
        v_wg = wind[4]  # gust along body y-axis
        w_wg = wind[5]  # gust along body z-axis

        # compute wind data in NED
        ct = np.cos(theta)
        cu = np.cos(psi)
        cp = np.cos(phi)
        st = np.sin(theta)
        su = np.sin(psi)
        sp = np.sin(phi)
        #     t = tan(theta)
        R1 = np.array(
            [
                [ct * cu, sp * st * cu - cp * su, cp * st * cu + sp * su],
                [ct * su, sp * st * su + cp * cu, cp * st * su - sp * cu],
                [-st, sp * ct, cp * ct],
            ]
        )

        # body to vehicle
        R2 = R1.T
        # vehicle to body

        # compute wind data in NED
        w_NED = (R1 @ np.array([u_wg, v_wg, w_wg])) + np.array([w_ns, w_es, w_ds])

        w_n = w_NED[0]
        w_e = w_NED[1]
        w_d = w_NED[2]

        # compute air data
        V_bw = (R2 @ np.array([w_ns, w_es, w_ds])) + np.array([u_wg, v_wg, w_wg])

        u_w = V_bw[0]
        v_w = V_bw[1]
        w_w = V_bw[2]

        V_ba = np.array([u - u_w, v - v_w, w - w_w])

        u_r = V_ba[0]
        v_r = V_ba[1]
        w_r = V_ba[2]

        Va = np.sqrt(u_r ** 2 + v_r ** 2 + w_r ** 2)
        alpha = np.arctan((w_r) / (u_r))
        beta = np.arcsin(v_r / Va)

        return Va, alpha, beta

    def forces_moments(
        self,
        x: State,
        control: Control,
        wind: Optional[np.ndarray] = None,
        airspeed: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        """Calculate the forces and moments on the airframe."""
        if wind is not None and all(i is None for i in [airspeed, alpha, beta]):
            # calculate from full state and wind
            airspeed, alpha, beta = self.get_airspeed_alpha_beta(x, control, wind)
        elif all(i is not None for i in [airspeed, alpha, beta]) and wind is None:
            # calculate directly from given airspeed, attack, sideslip
            w_n = None
            w_e = None
            w_d = None
        else:
            raise ValueError(
                "Must provide only 'wind' or all of 'airspeed', 'alpha', and"
                " 'beta', not a combination or none."
            )
        P = self.P

        # relabel the inputs
        pn = x.position[0]
        pe = x.position[1]
        pd = x.position[2]
        u = x.velocity[0]
        v = x.velocity[1]
        w = x.velocity[2]
        phi = x.angle[0]
        theta = x.angle[1]
        psi = x.angle[2]
        p = x.angle_rate[0]
        q = x.angle_rate[1]
        r = x.angle_rate[2]
        delta_e = control.delta_e
        delta_a = control.delta_a
        delta_r = control.delta_r
        delta_t = control.delta_t

        # compute wind data in NED
        ct = np.cos(theta)
        cu = np.cos(psi)
        cp = np.cos(phi)
        st = np.sin(theta)
        su = np.sin(psi)
        sp = np.sin(phi)

        # compute external forces and torques on aircraft
        f1 = (
            -P.mass * P.gravity * st
            + (
                ((1 / 2) * P.rho * (airspeed ** 2) * P.S_wing)
                * (
                    CX(alpha, P)
                    + CXq(alpha, P) * (P.c * q) / (2 * airspeed)
                    + CXde(alpha, P) * delta_e
                )
            )
            + ((1 / 2) * P.rho * P.S_prop * P.C_prop * ((P.k_motor * delta_t) ** 2))
        )
        # if np.isnan(f1):
        #     breakpoint()
        f2 = (P.mass * P.gravity * ct * sp) + (
            ((1 / 2) * P.rho * (airspeed ** 2) * P.S_wing)
            * (
                P.C_Y_0
                + P.C_Y_beta * beta
                + P.C_Y_p * (P.b * p) / (2 * airspeed)
                + P.C_Y_r * (P.b * r) / (2 * airspeed)
                + P.C_Y_delta_a * delta_a
                + P.C_Y_delta_r * delta_r
            )
        )
        f3 = (P.mass * P.gravity * ct * cp) + (
            ((1 / 2) * P.rho * airspeed ** 2 * P.S_wing)
            * (
                CZ(alpha, P)
                + CZq(alpha, P) * (P.c * q) / (2 * airspeed)
                + CZde(alpha, P) * delta_e
            )
        )

        Force = np.array([f1, f2, f3])

        t1 = ((1 / 2) * P.rho * airspeed ** 2 * P.S_wing) * (
            P.b
            * (
                P.C_ell_0
                + P.C_ell_beta * beta
                + P.C_ell_p * (P.b * p) / (2 * airspeed)
                + P.C_ell_r * (P.b * r) / (2 * airspeed)
                + P.C_ell_delta_a * delta_a
                + P.C_ell_delta_r * delta_r
            )
        ) - (P.k_T_P * (P.k_Omega * delta_t) ** 2)
        t2 = ((1 / 2) * P.rho * airspeed ** 2 * P.S_wing) * (
            P.c
            * (
                P.C_m_0
                + P.C_m_alpha * alpha
                + P.C_m_q * P.c / (2 * airspeed) * q
                + P.C_m_delta_e * delta_e
            )
        )
        t3 = ((1 / 2) * P.rho * airspeed ** 2 * P.S_wing) * (
            P.b
            * (
                P.C_n_0
                + P.C_n_beta * beta
                + P.C_n_p * P.b / (2 * airspeed) * p
                + P.C_n_r * P.b / (2 * airspeed) * r
                + P.C_n_delta_a * delta_a
                + P.C_n_delta_r * delta_r
            )
        )

        Torque = np.array([t1, t2, t3])

        return Force, Torque, airspeed, alpha, beta  # , w_n, w_e, w_d
