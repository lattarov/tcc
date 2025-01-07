"""Implements a base class for all controllers."""


class ControllerABC:
    """A controller abstract base class (ABC)."""

    def __init__(self):
        """Initialize parameters shared by allc controllers."""
        self.max_action = 3
