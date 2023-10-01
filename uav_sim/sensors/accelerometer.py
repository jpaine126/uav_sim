import numpy as np

from ..core.abc import Sensor
from ..core.state import State, rng


class Accelerometer(Sensor):
    """Provide sensor readings for an accelerometer.

    Noise is provided as noise density to allow for varying sample rate while
    keeping the same normal distribution sampling.
    
    Attributes:
        sample_rate: Putput smaple rate of sensor in Hz. Used to convert noise
            density to standard deviation.
        noise_density: Noise density in degrees/second/sqrt(Hz). Converted to 
            standard deviation internally.

    """

    def __init__(
        self, sample_rate: float, noise_density: float = 0.0,
    ):
        self.sample_rate = sample_rate
        self.noise_density = noise_density
        self.noise_std = noise_density * sample_rate ** 2

    def read(self, state: State):
        """Make a reading."""
        noise = rng.normal(0, self.noise_std, (3,))

        measurement = noise + state.acceleration
        return measurement
