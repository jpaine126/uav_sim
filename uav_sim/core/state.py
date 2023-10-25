from dataclasses import dataclass

import numpy as np

rng = np.random.default_rng(42)


@dataclass
class State:
    """Class to pass state values for UAV."""

    time: float

    position: np.ndarray
    velocity: np.ndarray

    angle: np.ndarray
    angle_rate: np.ndarray

    @classmethod
    def from_vector(cls, x: np.ndarray):
        """Instantiate state given an ordered vector of the state.
        
        Args:
            x (np.ndarray): 13 x 0 array of states time, x, y, z, x dot, y dot,
                z dot, phi, theta, psi, phi dot, theta dot, psi dot.
        """
        return cls(x[0], x[1:4], x[4:7], x[7:10], x[10:13])


@dataclass
class Control:
    """Class to pass control vector values for UAV."""

    delta_e: np.ndarray
    delta_r: np.ndarray
    delta_a: np.ndarray
    delta_t: np.ndarray

    @classmethod
    def from_vector(cls, u: np.ndarray):
        """Instantiate control given an ordered vector of the control surface deflections.
        
        Args:
            u (np.ndarray): 13 x 0 array of controls elevator, rudder, aileron,
                and thrust.
        """
        return cls(u[0], u[1], u[2], u[3])
