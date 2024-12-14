from dataclasses import dataclass, field

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

    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_vector(cls, x: np.ndarray):
        """Instantiate state given an ordered vector of the state.
        
        Args:
            x (np.ndarray): 12 x 1 array of states time, pn, pe, pd, u, v, w,
                phi, theta, psi, p, q, r.
        """
        return cls(x[0], x[1:4], x[4:7], x[7:10], x[10:13], np.zeros(3))


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
            u (np.ndarray): 4 x 1 array of controls elevator, rudder, aileron,
                and thrust.
        """
        return cls(u[0], u[1], u[2], u[3])
    
    @property
    def vector(self):
        return np.array([self.delta_e, self.delta_r, self.delta_a, self.delta_t])
