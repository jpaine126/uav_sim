import numpy as np

from ..core.abc import Sensor
from ..core.state import State, rng


class RateGyro(Sensor):
    """Provide sensor readings for a rate gyro.

    Noise is provided as noise density to allow for varying sample rate while
    keeping the same normal distribution sampling.
    
    Attributes:
        sample_rate: Putput smaple rate of sensor in Hz. Used to convert noise
            density to standard deviation.
        noise_density: Noise density in degrees/second/sqrt(Hz). Converted to 
            standard deviation internally.
        bias: Constant offset of the measurment.
        turn_on_bias: The amount of random error in the bias at turn on. Turn off
            for tuning observers, and turn on to simulate anccounted for bias.
        in_run_bias_rate: The rate at which bias drifts over time while sampling.
    
    """

    def __init__(
        self,
        sample_rate: float,
        noise_density: float = 0.0,
        bias: float = 0.0,
        turn_on_bias: float = 0.0,
        in_run_bias_rate: float = 0.0,
    ):
        self.sample_rate = sample_rate
        self.noise_density = noise_density
        self.noise_std = noise_density * sample_rate ** 2
        self.bias = bias
        self.turn_on_bias = turn_on_bias
        self.in_run_bias_rate = in_run_bias_rate

    def read(self, state: State):
        """Make a reading."""
        noise = rng.normal(0, self.noise_std, (3,))
        in_run_bias_term = state.time * self.in_run_bias
        total_bias = self.bias + self.turn_on_bias + in_run_bias_term
        total_noise = noise + total_bias
        measurement = total_noise + state.angle_rate
        return measurement
