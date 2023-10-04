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


@dataclass
class Control:
    """Class to pass control vector values for UAV."""

    delta_e: np.ndarray
    delta_r: np.ndarray
    delta_a: np.ndarray
    delta_t: np.ndarray