"""Dependency-free public interfaces."""

import abc


class SolverParameters(metaclass=abc.ABCMeta):
    """Interface for backend-specific ODE solver configuration."""
