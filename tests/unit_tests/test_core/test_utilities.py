import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from uav_sim.core import utilities

# =============================================================================
# Coordinate Transform Verification Tests
# =============================================================================


class TestEulerToDcm:
    """Verify that euler_to_dcm returns the correct Direction Cosine Matrix."""

    def test_returns_3x3_for_scalar_angles(self):
        R = utilities.euler_to_dcm(0.1, 0.2, 0.3)
        assert R.shape == (3, 3)

    def test_returns_3x3xN_for_array_angles(self):
        n = 5
        R = utilities.euler_to_dcm(np.zeros(n), np.zeros(n), np.zeros(n))
        assert R.shape == (3, 3, n)

    def test_zero_angles_returns_identity(self):
        R = utilities.euler_to_dcm(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3))

    @pytest.mark.parametrize(
        "roll,pitch,yaw",
        [
            (np.pi / 4, 0, 0),
            (0, np.pi / 6, 0),
            (0, 0, np.pi / 3),
            (np.pi / 4, np.pi / 6, np.pi / 3),
            (0.1, 0.2, 0.3),
            (-0.5, 1.2, -0.3),
        ],
    )
    def test_consistency_with_scipy(self, roll, pitch, yaw):
        """Our R_i^b should be the transpose of scipy's R_b^i."""
        expected = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix().T
        actual = utilities.euler_to_dcm(roll, pitch, yaw)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_array_consistency_with_scipy(self):
        n = 10
        angles = np.random.uniform(-np.pi, np.pi, (n, 3))
        # Our euler_to_dcm returns R_i^b (inertial->body), which is the
        # transpose of scipy's R_b^i (body->inertial)
        # We use transpose(2,1,0) to swap within each (3x3) slice while
        # bringing the batch dimension to the back
        expected = (
            R.from_euler("xyz", angles).as_matrix().transpose(2, 1, 0)
        )
        actual = utilities.euler_to_dcm(
            angles[:, 0], angles[:, 1], angles[:, 2]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-10)


class TestRotateInertialToBody:
    """Verify that vectors are correctly rotated from inertial to body frame."""

    def test_zero_rotation_is_identity(self):
        unit_rot = np.array([0, 0, 0])
        test_vec = np.array([10, 0, 0])
        result = utilities.rotate_inertial_to_body(unit_rot, test_vec)
        np.testing.assert_allclose(result, test_vec)

    def test_pure_yaw_rotates_north_to_body_x(self):
        """Pure yaw (90 deg CCW when looking down) should map
        inertial-North to body-East if using standard convention."""
        # yaw = 90 deg: the body x-axis points inertial-East
        # A purely-North inertial vector should therefore appear along
        # the negative body-y axis
        angle = np.array([0, 0, np.pi / 2])
        vec_i = np.array([1, 0, 0])  # North
        vec_b = utilities.rotate_inertial_to_body(angle, vec_i)
        np.testing.assert_allclose(vec_b, np.array([0, -1, 0]), atol=1e-10)

    def test_pure_pitch_maps_downward_to_body_x(self):
        """Pitch down 90 deg: body x-axis points inertial-down."""
        angle = np.array([0, np.pi / 2, 0])
        vec_i = np.array([0, 0, 1])  # Down
        vec_b = utilities.rotate_inertial_to_body(angle, vec_i)
        np.testing.assert_allclose(vec_b, np.array([-1, 0, 0]), atol=1e-10)

    def test_pure_roll_maps_east_to_body_z(self):
        """Roll right 90 deg: body z-axis points inertial-East."""
        angle = np.array([np.pi / 2, 0, 0])
        vec_i = np.array([0, 1, 0])  # East
        vec_b = utilities.rotate_inertial_to_body(angle, vec_i)
        np.testing.assert_allclose(vec_b, np.array([0, 0, -1]), atol=1e-10)

    def test_batch_rotation(self):
        """Function should work with batch (N x 3) angles and vectors."""
        angles = np.array([[0, 0, 0], [0, 0, np.pi / 2]])
        vecs = np.array([[1, 0, 0], [1, 0, 0]])
        out = utilities.rotate_inertial_to_body(angles, vecs)
        expected = np.array([[1, 0, 0], [0, -1, 0]])
        np.testing.assert_allclose(out, expected, atol=1e-10)


class TestRotateBodyToInertial:
    """Verify that vectors are correctly rotated from body to inertial frame."""

    def test_zero_rotation_is_identity(self):
        unit_rot = np.array([0, 0, 0])
        test_vec = np.array([10, 0, 0])
        result = utilities.rotate_body_to_inertial(unit_rot, test_vec)
        np.testing.assert_allclose(result, test_vec)

    def test_pure_yaw_rotates_body_x_to_east(self):
        """Pure yaw (90 deg) should map body-x to inertial-East."""
        angle = np.array([0, 0, np.pi / 2])
        vec_b = np.array([1, 0, 0])  # body x
        vec_i = utilities.rotate_body_to_inertial(angle, vec_b)
        np.testing.assert_allclose(vec_i, np.array([0, 1, 0]), atol=1e-10)

    def test_pure_pitch_rotates_body_x_to_down(self):
        """Pitch down 90 deg: body x-axis points inertial-down."""
        angle = np.array([0, np.pi / 2, 0])
        vec_b = np.array([1, 0, 0])  # body x
        vec_i = utilities.rotate_body_to_inertial(angle, vec_b)
        np.testing.assert_allclose(vec_i, np.array([0, 0, -1]), atol=1e-10)

    def test_pure_roll_rotates_body_z_to_east(self):
        """Roll right 90 deg: body z-axis points inertial-East."""
        angle = np.array([np.pi / 2, 0, 0])
        vec_b = np.array([0, 0, 1])  # body z
        vec_i = utilities.rotate_body_to_inertial(angle, vec_b)
        np.testing.assert_allclose(vec_i, np.array([0, -1, 0]), atol=1e-10)

    def test_batch_rotation(self):
        """Function should work with batch (N x 3) angles and vectors."""
        angles = np.array([[0, 0, 0], [0, 0, np.pi / 2]])
        vecs = np.array([[1, 0, 0], [1, 0, 0]])
        out = utilities.rotate_body_to_inertial(angles, vecs)
        expected = np.array([[1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(out, expected, atol=1e-10)


class TestRoundTrip:
    """Inertial->Body->Inertial and Body->Inertial->Body must return identity."""

    @pytest.mark.parametrize("angle", [
        np.array([0, 0, 0]),
        np.array([np.pi / 4, 0, 0]),
        np.array([0, np.pi / 6, 0]),
        np.array([0, 0, np.pi / 3]),
        np.array([0.3, 0.5, -0.7]),
        np.array([1.2, -0.8, 0.4]),
    ])
    def test_inertial_to_body_to_inertial(self, angle):
        vec_i = np.array([1.0, 2.0, 3.0])
        vec_b = utilities.rotate_inertial_to_body(angle, vec_i)
        vec_i_back = utilities.rotate_body_to_inertial(angle, vec_b)
        np.testing.assert_allclose(vec_i_back, vec_i, atol=1e-10)

    @pytest.mark.parametrize("angle", [
        np.array([0, 0, 0]),
        np.array([np.pi / 4, 0, 0]),
        np.array([0, np.pi / 6, 0]),
        np.array([0, 0, np.pi / 3]),
        np.array([0.3, 0.5, -0.7]),
    ])
    def test_body_to_inertial_to_body(self, angle):
        vec_b = np.array([-2.0, 5.0, 1.0])
        vec_i = utilities.rotate_body_to_inertial(angle, vec_b)
        vec_b_back = utilities.rotate_inertial_to_body(angle, vec_i)
        np.testing.assert_allclose(vec_b_back, vec_b, atol=1e-10)
