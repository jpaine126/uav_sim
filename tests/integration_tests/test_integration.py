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


class TestBallisticEnergyConservation:
    """Verify that total mechanical energy is conserved when forces and
    moments are zero (free-fall / ballistic flight)."""

    def test_energy_conservation_zero_aerodynamics(self, airframe):
        """Integrate with gravity only (zero aerodynamic forces/moments).
        Total mechanical energy (KE + PE) should be conserved."""
        initial_state = State(
            time=0,
            position=np.array([100, 50, -200]),
            velocity=np.array([15, -10, 5]),
            angle=np.array([0.2, -0.1, 0.5]),
            angle_rate=np.array([0.1, 0.05, -0.02]),
        )

        t_span = (0, 5.0)
        t_eval = np.linspace(*t_span, 500)

        def wrapper(t, y):
            state = State(t, y[0:3], y[3:6], y[6:9], y[9:12])
            # Gravity only: body-frame gravity = mass * R_i^b @ [0,0,g]
            g_ned = np.array([0, 0, params.gravity])
            forces = params.mass * utilities.rotate_inertial_to_body(state.angle, g_ned)
            moments = np.zeros(3)
            return airframe.derivative(state, forces, moments)

        sol = solve_ivp(
            wrapper,
            t_span=t_span,
            y0=np.hstack(
                (
                    initial_state.position,
                    initial_state.velocity,
                    initial_state.angle,
                    initial_state.angle_rate,
                )
            ),
            t_eval=t_eval,
            dense_output=True,
        )

        # Compute total mechanical energy at each timestep
        positions = sol.y[0:3, :].T
        velocities = sol.y[3:6, :].T
        # h = -pd  (height above origin)
        heights = -positions[:, 2]
        potential = params.mass * params.gravity * heights
        kinetic = 0.5 * params.mass * np.sum(velocities**2, axis=1)
        total_energy = potential + kinetic

        # Energy should be conserved to within solver tolerance
        max_drift = np.max(np.abs(total_energy - total_energy[0]))
        assert max_drift < 0.5, f"Energy drifted by {max_drift}"

    def test_zero_angular_rates_fixed_attitude(self, airframe):
        """With zero angular rates, the attitude derivatives should vanish
        regardless of forces, so the attitude should stay fixed."""
        initial_state = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([10, 0, 0]),
            angle=np.array([0.2, -0.1, 0.3]),
            angle_rate=np.array([0, 0, 0]),
        )

        def wrapper(t, y):
            state = State(t, y[0:3], y[3:6], y[6:9], y[9:12])
            # Gravity only: body-frame gravity = mass * R_i^b @ [0,0,g]
            g_ned = np.array([0, 0, params.gravity])
            forces = params.mass * utilities.rotate_inertial_to_body(state.angle, g_ned)
            moments = np.zeros(3)
            return airframe.derivative(state, forces, moments)

        sol = solve_ivp(
            wrapper,
            t_span=(0, 2.0),
            y0=np.hstack(
                (
                    initial_state.position,
                    initial_state.velocity,
                    initial_state.angle,
                    initial_state.angle_rate,
                )
            ),
            t_eval=np.linspace(0, 2, 200),
        )

        # Angular rates should remain zero
        np.testing.assert_allclose(sol.y[9:12, :], 0, atol=1e-10)
        # Angles should remain constant (no coupling if rates are zero)
        for i, expected_angle in enumerate(initial_state.angle):
            np.testing.assert_allclose(sol.y[6 + i, :], expected_angle, atol=1e-10)


class TestKinematicVerification:
    """Low-level kinematic checks using manually-constructed states."""

    def test_steady_turn_kinematics(self, airframe):
        """Verify Euler-angle rate kinematics for a pure yaw rate."""
        state = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([10, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 0.5]),
        )
        forces = np.zeros(3)
        moments = np.zeros(3)
        x_dot = airframe.derivative(state, forces, moments)
        np.testing.assert_allclose(x_dot[6], 0, atol=1e-10)  # phi_dot
        np.testing.assert_allclose(x_dot[7], 0, atol=1e-10)  # theta_dot
        np.testing.assert_allclose(x_dot[8], 0.5, atol=1e-10)  # psi_dot

    def test_steady_climb_kinematics(self, airframe):
        """Pitch up -> NED-down rate equals -Va*sin(theta)."""
        theta = np.deg2rad(5)
        Va = 10.0
        state = State(
            time=0,
            position=np.array([0, 0, -100]),
            velocity=np.array([Va, 0, 0]),
            angle=np.array([0, theta, 0]),
            angle_rate=np.array([0, 0, 0]),
        )
        g_ned = np.array([0, 0, params.gravity])
        forces = params.mass * utilities.rotate_inertial_to_body(state.angle, g_ned)
        moments = np.zeros(3)
        x_dot = airframe.derivative(state, forces, moments)
        # NED down rate
        ned_vel = utilities.rotate_body_to_inertial(state.angle, state.velocity)
        np.testing.assert_allclose(x_dot[2], ned_vel[2], atol=1e-10)
        np.testing.assert_allclose(x_dot[2], -Va * np.sin(theta), atol=1e-10)

    def test_level_flight_free_fall(self, airframe):
        """With no aerodynamics, gravity produces [0,0,g] body acceleration."""
        state = State(
            time=0,
            position=np.array([0, 0, 0]),
            velocity=np.array([0, 0, 0]),
            angle=np.array([0, 0, 0]),
            angle_rate=np.array([0, 0, 0]),
        )
        g_body = np.array([0, 0, params.gravity])
        forces = params.mass * g_body
        moments = np.zeros(3)
        x_dot = airframe.derivative(state, forces, moments)
        np.testing.assert_allclose(x_dot[3:6], g_body, atol=1e-10)


class TestTrimSolver:
    """Verify the numerical trim solver finds physically-consistent states."""

    def test_straight_and_level_trim(self, airframe):
        """A straight-and-level trim should produce near-zero derivatives
        in velocity and angular acceleration."""
        from uav_sim.setup.find_trim import find_trim

        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=np.deg2rad(0),
            desired_radius=np.inf,
        )

        u, v, w = trim_state.velocity
        Va_calc = np.sqrt(u**2 + v**2 + w**2)
        alpha_calc = np.arctan2(w, u)
        beta_calc = np.arcsin(v / Va_calc) if Va_calc > 0 else 0.0

        forces, moments, *_ = airframe.forces_moments(
            trim_state,
            trim_control,
            airspeed=Va_calc,
            alpha=alpha_calc,
            beta=beta_calc,
        )
        derivatives = airframe.derivative(trim_state, forces, moments)

        # Accelerations should be essentially zero in a true trim
        np.testing.assert_allclose(derivatives[3:6], np.zeros(3), atol=1e-2)
        np.testing.assert_allclose(derivatives[9:12], np.zeros(3), atol=1e-2)

    def test_climb_trim(self, airframe):
        """A climb trim should give steady climb rate."""
        from uav_sim.setup.find_trim import find_trim

        gamma = np.deg2rad(5)
        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=gamma,
            desired_radius=np.inf,
        )

        u, v, w = trim_state.velocity
        Va_calc = np.sqrt(u**2 + v**2 + w**2)
        alpha_calc = np.arctan2(w, u)
        beta_calc = np.arcsin(v / Va_calc) if Va_calc > 1e-6 else 0.0

        forces, moments, *_ = airframe.forces_moments(
            trim_state,
            trim_control,
            airspeed=Va_calc,
            alpha=alpha_calc,
            beta=beta_calc,
        )
        derivatives = airframe.derivative(trim_state, forces, moments)

        # Translational and rotational accelerations should vanish
        np.testing.assert_allclose(derivatives[3:6], np.zeros(3), atol=1e-1)
        np.testing.assert_allclose(derivatives[9:12], np.zeros(3), atol=1e-1)

        # NED-down rate should reflect the climb angle
        expected_pd_dot = Va_calc * np.sin(gamma)
        np.testing.assert_allclose(derivatives[2], expected_pd_dot, atol=1e-2)

    def test_coordinated_turn_trim(self, airframe):
        """A coordinated turn should produce steady yaw rate."""
        from uav_sim.setup.find_trim import find_trim

        R_turn = 100.0
        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=np.deg2rad(0),
            desired_radius=R_turn,
        )

        u, v, w = trim_state.velocity
        Va_calc = np.sqrt(u**2 + v**2 + w**2)
        alpha_calc = np.arctan2(w, u)
        beta_calc = np.arcsin(v / Va_calc) if Va_calc > 1e-6 else 0.0

        forces, moments, *_ = airframe.forces_moments(
            trim_state,
            trim_control,
            airspeed=Va_calc,
            alpha=alpha_calc,
            beta=beta_calc,
        )
        derivatives = airframe.derivative(trim_state, forces, moments)

        # Translational and rotational accelerations should vanish
        np.testing.assert_allclose(derivatives[3:6], np.zeros(3), atol=1e-1)
        np.testing.assert_allclose(derivatives[9:12], np.zeros(3), atol=1e-1)

        # Yaw rate should be Va / R
        expected_yaw_rate = Va_calc / R_turn
        np.testing.assert_allclose(derivatives[8], expected_yaw_rate, atol=1e-2)


class TestOpenLoopTrimStability:
    """Propagate from the numerical trim and verify the state stays close."""

    def test_open_loop_straight_and_level(self, airframe):
        """Open-loop propagation from straight-and-level trim."""
        from uav_sim.setup.find_trim import find_trim

        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=np.deg2rad(0),
            desired_radius=np.inf,
        )

        def wrapper(t, y):
            state = State(t, y[0:3], y[3:6], y[6:9], y[9:12])
            forces, moments, *_ = airframe.forces_moments(
                state, trim_control, wind=np.zeros(6)
            )
            return airframe.derivative(state, forces, moments)

        sol = solve_ivp(
            wrapper,
            t_span=(0, 5.0),
            y0=np.hstack(
                (
                    trim_state.position,
                    trim_state.velocity,
                    trim_state.angle,
                    trim_state.angle_rate,
                )
            ),
            t_eval=np.linspace(0, 5, 500),
        )

        # Airspeed should remain roughly constant
        velocities = sol.y[3:6, :].T
        airspeeds = np.linalg.norm(velocities, axis=1)
        airspeed_drift = np.max(np.abs(airspeeds - airspeeds[0]))
        assert airspeed_drift < 1.0, f"Airspeed drifted by {airspeed_drift}"

        # Altitude should stay roughly constant (straight and level)
        p_d = sol.y[2, :]
        altitude_drift = np.max(np.abs(p_d - p_d[0]))
        assert altitude_drift < 10.0, f"Altitude drifted by {altitude_drift}"
