"""Abstract base classes."""

from abc import ABC, abstractmethod

from .state import State


class Sensor(ABC):
    """Sensor abstract base class."""

    @abstractmethod
    def read(self, state: State):
        """Take a reading."""


class Dynamic(ABC):
    """Abstract base class for all objects with dynamics."""

    @abstractmethod
    def update(self, state: State):
        """Update internal state."""


class Estimator(Dynamic, ABC):
    """Estimator abstract base class."""


class Controller(Dynamic, ABC):
    """Controller abstract base class."""
