import numpy as np

from ..core.abc import Dynamic
from ..core.state import Control, State


class Airframe(Dynamic):
    """Class to contain the constants and equations representing an airframe."""

    def __init__(self, P, body_vertices):
        self.P = P
        self.body_vertices = body_vertices

    def update(self):
        ...

    def model_derivative(self, x: State, forces, moments):
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
        R2 = np.array([[1, sp * tt, cp * tt], [0, cp, -sp], [0, sp / ct, cp / ct],])
        x_dot[6:9] = R2 @ x.angle_rate

        # p_dot, q_dot, r_dot
        g = P.Jx * P.Jz - P.Jxz ** 2
        g1 = (P.Jxz * (P.Jx - P.Jy + P.Jz)) / g
        g2 = (P.Jz * (P.Jz - P.Jy) + P.Jxz ** 2) / g
        g3 = P.Jz / g
        g4 = P.Jxz / g
        g5 = (P.Jz - P.Jx) / P.Jy
        g6 = P.Jxz / P.Jy
        g7 = ((P.Jx - P.Jy) * P.Jx + P.Jxz ** 2) / g
        g8 = P.Jx / g

        x_dot[9] = g1 * p * q - g2 * q * r + g3 * ell + g4 * n
        x_dot[10] = g5 * p * r - g6 * (p ** 2 - r ** 2) + m / P.Jy
        x_dot[11] = g7 * p * q - g1 * q * r + g4 * ell + g8 * n

        return x_dot

    def forces_moments(self, x: State, control: Control, wind):
        """Calculate the forces and moments on the airframe."""
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

        # compute external forces and torques on aircraft
        def CD(alpha):
            return P.C_D_p + (P.C_L_0 + P.C_L_alpha * alpha) ** 2 / (np.pi * P.e * P.AR)

        def CL(alpha):
            return (1 - sigma(alpha)) * (P.C_L_0 + P.C_L_alpha * alpha) + sigma(
                alpha
            ) * (2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha))

        def sigma(alpha):
            return (
                1 + np.exp(-P.M * (alpha - P.alpha0)) + np.exp(P.M * (alpha + P.alpha0))
            ) / (
                (1 + np.exp(-P.M * (alpha - P.alpha0)))
                * (1 + np.exp(P.M * (alpha + P.alpha0)))
            )

        def CX(alpha):
            return -CD(alpha) * np.cos(alpha) + CL(alpha) * np.sin(alpha)

        def CXq(alpha):
            return -P.C_D_q * np.cos(alpha) + P.C_L_q * np.sin(alpha)

        def CXde(alpha):
            return -P.C_D_delta_e * np.cos(alpha) + P.C_L_delta_e * np.sin(alpha)

        def CZ(alpha):
            return -CD(alpha) * np.sin(alpha) - CL(alpha) * np.cos(alpha)

        def CZq(alpha):
            return -P.C_D_q * np.sin(alpha) - P.C_L_q * np.cos(alpha)

        def CZde(alpha):
            return -P.C_D_delta_e * np.sin(alpha) - P.C_L_delta_e * np.cos(alpha)

        f1 = (
            -P.mass * P.gravity * st
            + 1
            / 2
            * P.rho
            * Va ** 2
            * P.S_wing
            * (CX(alpha) + CXq(alpha) * P.c / (2 * Va) * q + CXde(alpha) * delta_e)
            + 1 / 2 * P.rho * P.S_prop * P.C_prop * (P.k_motor * delta_t) ** 2
        )
        f2 = P.mass * P.gravity * ct * sp + 1 / 2 * P.rho * Va ** 2 * P.S_wing * (
            P.C_Y_0
            + P.C_Y_beta * beta
            + P.C_Y_p * P.b / (2 * Va) * p
            + P.C_Y_r * P.b / (2 * Va) * r
            + P.C_Y_delta_a * delta_a
            + P.C_Y_delta_r * delta_r
        )
        f3 = P.mass * P.gravity * ct * cp + 1 / 2 * P.rho * Va ** 2 * P.S_wing * (
            CZ(alpha) + CZq(alpha) * P.c / (2 * Va) * q + CZde(alpha) * delta_e
        )

        Force = np.array([f1, f2, f3])

        t1 = (
            1
            / 2
            * P.rho
            * Va ** 2
            * P.S_wing
            * (
                P.b
                * (
                    P.C_ell_0
                    + P.C_ell_beta * beta
                    + P.C_ell_p * P.b / (2 * Va) * p
                    + P.C_ell_r * P.b / (2 * Va) * r
                    + P.C_ell_delta_a * delta_a
                    + P.C_ell_delta_r * delta_r
                )
            )
            - P.k_T_P * (P.k_Omega * delta_t) ** 2
        )
        t2 = (
            1
            / 2
            * P.rho
            * Va ** 2
            * P.S_wing
            * (
                P.c
                * (
                    P.C_m_0
                    + P.C_m_alpha * alpha
                    + P.C_m_q * P.c / (2 * Va) * q
                    + P.C_m_delta_e * delta_e
                )
            )
        )
        t3 = (
            1
            / 2
            * P.rho
            * Va ** 2
            * P.S_wing
            * (
                P.b
                * (
                    P.C_n_0
                    + P.C_n_beta * beta
                    + P.C_n_p * P.b / (2 * Va) * p
                    + P.C_n_r * P.b / (2 * Va) * r
                    + P.C_n_delta_a * delta_a
                    + P.C_n_delta_r * delta_r
                )
            )
        )

        Torque = np.array([t1, t2, t3])

        return Force, Torque, Va, alpha, beta, w_n, w_e, w_d
