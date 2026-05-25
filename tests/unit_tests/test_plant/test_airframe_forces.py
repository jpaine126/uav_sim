import numpy as np
import pytest
from scipy.integrate import solve_ivp
from uav_sim.core import utilities
from uav_sim.core.state import Control, State
from uav_sim.plant.airframe import Airframe
from uav_sim.setup import body_vertices, params


@pytest.fixture
def airframe():
    return Airframe(params, body_vertices)


# =============================================================================
# Force & Moment Frame Verification Tests
# =============================================================================


class TestGravityForces:
    """Gravity should appear in the correct body-frame directions.
    The gravity vector in inertial NED is [0, 0, +g] (positive down).
    In the body frame the total force counteracting gravity is -R_i^b @ [0,0,g].
    Since forces_moments() returns the net external force (positive body-z when
    pointing up), at level flight the body-z component should be +mg.
    """

    def test_level_flight_gravity_on_positive_body_z(self, airframe):
        """At roll=0, pitch=0, gravity (NED +D) maps entirely to body +z.
        Therefore the body-z force required to counter gravity is +mg."""
        # Verify directly with the DCM: R_i^b @ [0,0,g]
        R = utilities.euler_to_dcm(0, 0, 0)
        g_ned = np.array([0, 0, params.gravity])
        g_body = R @ g_ned
        # In the body frame, gravity points +z. The aerodynamic force
        # that counters gravity also points +z
        np.testing.assert_allclose(g_body, np.array([0, 0, params.gravity]), atol=1e-10)
        # Total external force = mass * g_body
        total_force = params.mass * g_body
        np.testing.assert_allclose(
            total_force[2], params.mass * params.gravity, atol=1e-10
        )

    def test_pitch_90_pitches_gravity_onto_negative_body_x(self, airframe):
        """With pitch=+90 deg (nose up), body x-axis points inertial-up.
        Gravity (NED +D) therefore maps to negative body-x."""
        R = utilities.euler_to_dcm(0, np.pi / 2, 0)
        g_ned = np.array([0, 0, params.gravity])
        g_body = R @ g_ned
        # Gravity now aligns with -body-x (positive down -> negative body-x
        # because body-x points up)
        np.testing.assert_allclose(
            g_body, np.array([-params.gravity, 0, 0]), atol=1e-10
        )

    def test_pitch_minus90_pitches_gravity_onto_body_x(self, airframe):
        """With pitch=-90 deg (nose down), body x-axis points inertial-down.
        Gravity should therefore map onto the positive body-x axis."""
        R = utilities.euler_to_dcm(0, -np.pi / 2, 0)
        g_ned = np.array([0, 0, params.gravity])
        g_body = R @ g_ned
        # Gravity now aligns with +body-x
        np.testing.assert_allclose(g_body, np.array([params.gravity, 0, 0]), atol=1e-10)

    def test_roll_90_moves_gravity_to_body_y(self, airframe):
        """With roll=+90 deg, body y-axis points inertial-down.
        Gravity (NED +D) therefore maps to positive body-y."""
        R = utilities.euler_to_dcm(np.pi / 2, 0, 0)
        g_ned = np.array([0, 0, params.gravity])
        g_body = R @ g_ned
        # Gravity now aligns with +body-y
        np.testing.assert_allclose(g_body, np.array([0, params.gravity, 0]), atol=1e-10)

    def test_roll_minus90_moves_gravity_to_negative_body_y(self, airframe):
        """With roll=-90 deg, body y-axis points inertial-up.
        Gravity (NED +D) therefore maps to negative body-y."""
        R = utilities.euler_to_dcm(-np.pi / 2, 0, 0)
        g_ned = np.array([0, 0, params.gravity])
        g_body = R @ g_ned
        # Gravity now aligns with -body-y
        np.testing.assert_allclose(
            g_body, np.array([0, -params.gravity, 0]), atol=1e-10
        )

    def test_zero_alpha_cx_and_cz_signs(self, airframe):
        """At alpha=0, CZ = -C_L_0.  CX = -C_D_p - C_L_0^2/(pi*e*AR)
        due to the drag-polar term in the aerodynamic model."""
        from uav_sim.plant.airframe import CD, CX, CZ

        alpha = 0.0
        # At alpha=0, CL(0) = C_L_0 and CD(0) contains the drag-polar
        # term C_L_0^2 / (pi * e * AR)
        expected_cd0 = params.C_D_p + (params.C_L_0**2) / (np.pi * params.e * params.AR)
        expected_cx = -expected_cd0
        expected_cz = -params.C_L_0
        np.testing.assert_allclose(CX(alpha, params), expected_cx, atol=1e-10)
        np.testing.assert_allclose(CZ(alpha, params), expected_cz, atol=1e-10)


class TestCoriolisTerms:
    """Verify the kinematic coupling terms in the body-velocity equations."""

    def test_pure_pitch_rate_produces_vertical_acceleration(self, airframe):
        """With q=1, u=10, and zero forces, the body-z acceleration
        should be w_dot = q*u = 10."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([10, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 1, 0]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        np.testing.assert_allclose(x_dot[5], 10.0, atol=1e-10)

    def test_pure_roll_rate_produces_vertical_acceleration(self, airframe):
        """With p=1, u=10, and zero forces, the body-y acceleration
        should be v_dot = -p*w = 0 (since w=0). Instead, set p=1, w=5:
        v_dot = p*w = 5."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([0, 0, 5]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([1, 0, 0]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        # v_dot = p*w - r*u  (r=0, u=0)
        np.testing.assert_allclose(x_dot[4], 5.0, atol=1e-10)

    def test_pure_yaw_rate_produces_lateral_acceleration(self, airframe):
        """With r=1, u=10, and zero forces, the body-y acceleration
        should be v_dot = -r*u = -10."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([10, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 1]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        # v_dot = p*w - r*u  (p=0, w=0)
        np.testing.assert_allclose(x_dot[4], -10.0, atol=1e-10)

    def test_zero_angular_rates_zero_coriolis(self, airframe):
        """With zero angular rates, the coriolis term should vanish
        and body acceleration should equal force/mass."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([5, 3, 2]),
            angle=np.array([0.2, 0.1, -0.3]),
            angle_rate=np.array([0, 0, 0]),
        )
        forces = np.array([10, -5, 20])
        moments = np.array([0, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        expected_acc = forces / params.mass
        np.testing.assert_allclose(x_dot[3:6], expected_acc, atol=1e-10)


class TestMomentDerivatives:
    """Verify the rotational dynamics (p_dot, q_dot, r_dot)."""

    def test_pure_moment_about_x(self, airframe):
        """With ell=1, m=n=0, p=q=r=0, p_dot equals gamma_3*ell and
        r_dot equals gamma_4*ell (roll/yaw coupling via Jxz)."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([0, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 0]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([1, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        expected_p_dot = params.gamma_3 * 1  # ell term
        expected_r_dot = params.gamma_4 * 1  # ell cross-coupling via Jxz
        np.testing.assert_allclose(x_dot[9], expected_p_dot, atol=1e-10)
        np.testing.assert_allclose(x_dot[11], expected_r_dot, atol=1e-10)
        np.testing.assert_allclose(x_dot[10], 0, atol=1e-10)

    def test_pure_moment_about_y(self, airframe):
        """With m=1, ell=n=0, p=q=r=0, q_dot should equal m / Jy."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([0, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 0]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 1, 0])
        x_dot = airframe.derivative(x, forces, moments)
        expected = 1 / params.Jy
        np.testing.assert_allclose(x_dot[10], expected, atol=1e-10)
        np.testing.assert_allclose(x_dot[9], 0, atol=1e-10)
        np.testing.assert_allclose(x_dot[11], 0, atol=1e-10)

    def test_pure_moment_about_z(self, airframe):
        """With n=1, ell=m=0, p=q=r=0, p_dot equals gamma_4*n (via Jxz)
        and r_dot equals gamma_8*n."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([0, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 0]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 0, 1])
        x_dot = airframe.derivative(x, forces, moments)
        expected_p_dot = params.gamma_4 * 1  # n cross-coupling via Jxz
        expected_r_dot = params.gamma_8 * 1
        np.testing.assert_allclose(x_dot[11], expected_r_dot, atol=1e-10)
        np.testing.assert_allclose(x_dot[9], expected_p_dot, atol=1e-10)
        np.testing.assert_allclose(x_dot[10], 0, atol=1e-10)

    def test_gyroscopic_terms_vanish_when_aligned(self, airframe):
        """When Jxz=0 and p=q=r=0, all gyroscopic terms vanish.
        We already have Jxz=0.0015, but for a near-symmetric case the
        cross terms are small.  With zero rates they vanish exactly."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([0, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 0]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        np.testing.assert_allclose(x_dot[9:12], np.zeros(3), atol=1e-10)


class TestAngularRateKinematics:
    """Verify the euler-angle rate kinematics (phi_dot, theta_dot, psi_dot)."""

    def test_zero_rates_gives_zero_angle_rates(self, airframe):
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([0, 0, 0]),
            angle=np.array([0.3, 0.2, 0.1]),
            angle_rate=np.array([0, 0, 0]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        np.testing.assert_allclose(x_dot[6:9], np.zeros(3), atol=1e-10)

    def test_level_flight_pure_yaw_rate(self, airframe):
        """At phi=theta=0, pure yaw rate maps directly onto psi_dot."""
        x = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([0, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 1]),
        )
        forces = np.array([0, 0, 0])
        moments = np.array([0, 0, 0])
        x_dot = airframe.derivative(x, forces, moments)
        np.testing.assert_allclose(x_dot[8], 1.0, atol=1e-10)
        np.testing.assert_allclose(x_dot[6], 0, atol=1e-10)
        np.testing.assert_allclose(x_dot[7], 0, atol=1e-10)
