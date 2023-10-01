from dataclasses import dataclass

import numpy as np

rng = np.random.default_rng(42)


@dataclass
class State:
    """Class to pass state values for UAV.
    
    Holds all data needed for every part of the sim, including UAV physical
    states and misc parameters.
    """

    time: float

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    euler_angles: np.ndarray
    angle_rate: np.ndarray
