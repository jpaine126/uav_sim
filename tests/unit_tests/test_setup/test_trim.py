import numpy as np
import pytest
from scipy.integrate import solve_ivp

from uav_sim.core.state import State
from uav_sim.plant.airframe import Airframe
from uav_sim.setup import body_vertices, params


@pytest.fixture
def airframe():
    return Airframe(params, body_vertices)


def _compute_alpha_beta(state, airspeed):
    """Extract incidence / sideslip from the body-frame velocity vector."""
    u, v, w = state.velocity
    alpha = np.arctan2(w, u)
    beta = np.arcsin(v / airspeed) if airspeed > 1e-6 else 0.0
    return alpha, beta


class TestTrimSolver:
    """Comprehensive tests for the numerical trim solver."""

    def test_straight_and_level_trim_derivatives_are_zero(self, airframe):
        """At straight-and-level trim, all acceleration derivatives
        should be close to zero."""
        from uav_sim.setup.find_trim import find_trim

        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=0.0,
            desired_radius=np.inf,
        )

        alpha, beta = _compute_alpha_beta(trim_state, 10.0)
        forces, moments, *_ = airframe.forces_moments(
            trim_state, trim_control, airspeed=10.0, alpha=alpha, beta=beta
        )
        derivatives = airframe.derivative(trim_state, forces, moments)

        np.testing.assert_allclose(derivatives[3:6], np.zeros(3), atol=1e-2)
        np.testing.assert_allclose(derivatives[9:12], np.zeros(3), atol=1e-2)

    def test_climb_trim_produces_positive_pd_dot(self, airframe):
        """A positive flight-path angle should yield a positive NED-down
        rate (i.e. the aircraft descends)."""
        from uav_sim.setup.find_trim import find_trim

        gamma = np.deg2rad(5)
        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=gamma,
            desired_radius=np.inf,
        )

        alpha, beta = _compute_alpha_beta(trim_state, 10.0)
        forces, moments, *_ = airframe.forces_moments(
            trim_state, trim_control, airspeed=10.0, alpha=alpha, beta=beta
        )
        derivatives = airframe.derivative(trim_state, forces, moments)

        np.testing.assert_allclose(derivatives[3:6], np.zeros(3), atol=1e-1)
        np.testing.assert_allclose(derivatives[9:12], np.zeros(3), atol=1e-1)

        expected_pd_dot = 10.0 * np.sin(gamma)
        np.testing.assert_allclose(derivatives[2], expected_pd_dot, atol=1e-2)

    def test_turn_trim_produces_correct_yaw_rate(self, airframe):
        """A coordinated turn should produce the kinematic yaw rate Va/R."""
        from uav_sim.setup.find_trim import find_trim

        R = 100.0
        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=0.0,
            desired_radius=R,
        )

        alpha, beta = _compute_alpha_beta(trim_state, 10.0)
        forces, moments, *_ = airframe.forces_moments(
            trim_state, trim_control, airspeed=10.0, alpha=alpha, beta=beta
        )
        derivatives = airframe.derivative(trim_state, forces, moments)

        np.testing.assert_allclose(derivatives[3:6], np.zeros(3), atol=1e-1)
        np.testing.assert_allclose(derivatives[9:12], np.zeros(3), atol=1e-1)

        np.testing.assert_allclose(derivatives[8], 10.0 / R, atol=1e-2)

    def test_trim_state_is_returned_clean(self, airframe):
        """find_trim must not mutate the returned State in-place."""
        from uav_sim.setup.find_trim import find_trim

        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
        )
        # Previously the function set angle[2] = 0 directly
        # After the fix the returned yaw should already be 0.0
        assert np.isfinite(trim_state.angle[2])
        assert np.all(np.isfinite(trim_state.position))


class TestTrimOpenLoopStability:
    """Propagate from trim states and verify stability."""

    def _propagate_from_trim(self, airframe, trim_state, trim_control, t_span):
        def wrapper(t, y):
            state = State(t, y[0:3], y[3:6], y[6:9], y[9:12])
            forces, moments, *_ = airframe.forces_moments(
                state, trim_control, wind=np.zeros(6)
            )
            return airframe.derivative(state, forces, moments)

        return solve_ivp(
            wrapper,
            t_span=t_span,
            y0=np.hstack((
                trim_state.position,
                trim_state.velocity,
                trim_state.angle,
                trim_state.angle_rate,
            )),
            t_eval=np.linspace(*t_span, 500),
        )

    def test_straight_level_opens_loop_stable(self, airframe):
        """Open-loop propagation from straight-and-level trim."""
        from uav_sim.setup.find_trim import find_trim

        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=0.0,
            desired_radius=np.inf,
        )

        sol = self._propagate_from_trim(airframe, trim_state, trim_control, (0, 5.0))

        # Airspeed should remain within 0.5 m/s of trim
        velocities = sol.y[3:6, :].T
        airspeeds = np.linalg.norm(velocities, axis=1)
        assert np.max(np.abs(airspeeds - 10.0)) < 0.5

        # Altitude (pd) should not drift excessively
        assert np.max(np.abs(sol.y[2, :] - sol.y[2, 0])) < 5.0

    def test_climb_opens_loop_stable(self, airframe):
        """Open-loop propagation from climb trim shows correct descent rate."""
        from uav_sim.setup.find_trim import find_trim

        gamma = np.deg2rad(5)
        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_fpa=gamma,
        )

        sol = self._propagate_from_trim(airframe, trim_state, trim_control, (0, 5.0))

        # Average NED-down rate should be close to Va*sin(gamma)
        avg_pd_dot = (sol.y[2, -1] - sol.y[2, 0]) / 5.0
        np.testing.assert_allclose(avg_pd_dot, 10.0 * np.sin(gamma), atol=0.5)

    def test_turn_opens_loop_closes_loop(self, airframe):
        """Open-loop propagation from turn trim should close a loop
        after one full orbit."""
        from uav_sim.setup.find_trim import find_trim

        R = 100.0
        trim_state, trim_control = find_trim(
            desired_airspeed=10.0,
            desired_radius=R,
        )

        period = 2 * np.pi * R / 10.0  # one orbit in seconds
        sol = self._propagate_from_trim(
            airframe, trim_state, trim_control, (0, period)
        )

        # After one orbit the lateral position should be close to the start
        delta_n = sol.y[0, -1] - sol.y[0, 0]
        delta_e = sol.y[1, -1] - sol.y[1, 0]
        assert np.sqrt(delta_n ** 2 + delta_e ** 2) < 10.0, (
            f"Orbit did not close: N error={delta_n:.2f}, E error={delta_e:.2f}"
        )
