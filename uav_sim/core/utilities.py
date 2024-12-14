import numpy as np


def euler_to_dcm(roll, pitch, yaw):
    """Return the Direction Cosine Matrix R_i^b (inertial to body) for given
    euler angles.

    Args:
        roll, pitch, yaw: Euler angles in radians; may be scalars or arrays.

    Returns:
        np.ndarray: If inputs are scalar, returns a (3, 3) matrix.
            If inputs are 1D arrays of length N, returns a (3, 3, N) stack
            of matrices.
    """
    roll = np.asarray(roll)
    pitch = np.asarray(pitch)
    yaw = np.asarray(yaw)

    c_roll = np.cos(roll)
    s_roll = np.sin(roll)
    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    if roll.ndim == 0:
        # Scalar euler angles -- return a simple (3, 3) matrix.
        return np.array(
            [
                [c_pitch * c_yaw, c_pitch * s_yaw, -s_pitch],
                [s_roll * s_pitch * c_yaw - c_roll * s_yaw, s_roll * s_pitch * s_yaw + c_roll * c_yaw, s_roll * c_pitch],
                [c_roll * s_pitch * c_yaw + s_roll * s_yaw, c_roll * s_pitch * s_yaw - s_roll * c_yaw, c_roll * c_pitch],
            ]
        )
    else:
        # Array euler angles -- return (3, 3, N) stack.
        R = np.empty((3, 3, roll.size))
        R[0, 0, :] = c_pitch * c_yaw
        R[0, 1, :] = c_pitch * s_yaw
        R[0, 2, :] = -s_pitch
        R[1, 0, :] = s_roll * s_pitch * c_yaw - c_roll * s_yaw
        R[1, 1, :] = s_roll * s_pitch * s_yaw + c_roll * c_yaw
        R[1, 2, :] = s_roll * c_pitch
        R[2, 0, :] = c_roll * s_pitch * c_yaw + s_roll * s_yaw
        R[2, 1, :] = c_roll * s_pitch * s_yaw - s_roll * c_yaw
        R[2, 2, :] = c_roll * c_pitch
        return R


def rotate_inertial_to_body(angle: np.ndarray, vec: np.ndarray):
    """Rotate a vector from the inertial frame into the body frame.

    Args:
        angle: length-3 vector of euler angles [roll, pitch, yaw].
        vec: length-3 vector expressed in the inertial frame.

    Returns:
        np.ndarray: The input vector rotated into the body frame.
    """
    angle = np.atleast_2d(np.asarray(angle))
    vec = np.atleast_2d(np.asarray(vec))

    if angle.shape[0] == 1 and vec.shape[0] == 1:
        # Single vector case: extract scalars and return a 1D result
        R = euler_to_dcm(
            float(angle[0, 0]),
            float(angle[0, 1]),
            float(angle[0, 2]),
        )
        return (R @ vec.T).T.ravel()

    # Batch case: pass arrays through to einsum
    R = euler_to_dcm(angle[:, 0], angle[:, 1], angle[:, 2])
    return np.einsum('ijk,kj->ki', R, vec)


def rotate_body_to_inertial(angle: np.ndarray, vec: np.ndarray):
    """Rotate a vector from the body frame into the inertial frame.

    Args:
        angle: length-3 vector of euler angles [roll, pitch, yaw].
        vec: length-3 vector expressed in the body frame.

    Returns:
        np.ndarray: The input vector rotated into the inertial frame.
    """
    angle = np.atleast_2d(np.asarray(angle))
    vec = np.atleast_2d(np.asarray(vec))

    if angle.shape[0] == 1 and vec.shape[0] == 1:
        # Single vector case: extract scalars and return a 1D result
        R = euler_to_dcm(
            float(angle[0, 0]),
            float(angle[0, 1]),
            float(angle[0, 2]),
        )
        return (R.T @ vec.T).T.ravel()

    # Batch case: pass arrays through to einsum
    R = euler_to_dcm(angle[:, 0], angle[:, 1], angle[:, 2])
    return np.einsum('jik,kj->ki', R, vec)
