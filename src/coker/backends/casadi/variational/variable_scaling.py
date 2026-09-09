"""Internal affine scaling for variational-solver decision vectors.

The solver works in ``y`` coordinates while model expressions and returned
solutions remain in physical ``v`` coordinates::

    v = offset + scale * y

This module deliberately has no public package exports; it is an implementation
building block for the CasADi variational backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np


def _as_vector(value: Any, name: str) -> np.ndarray:
    """Convert a numeric vector to a one-dimensional float array."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.ndim == 0:
        raise ValueError(f"{name} must be a vector")
    if array.ndim > 1 and 1 not in array.shape:
        raise ValueError(f"{name} must be one-dimensional")
    return np.ascontiguousarray(array.reshape(-1), dtype=float)


@dataclass(frozen=True)
class _VariableScaling:
    """Positive affine vector transform used by the internal transcription."""

    offset: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        offset = _as_vector(self.offset, "offset")
        scale = _as_vector(self.scale, "scale")
        if offset.size != scale.size:
            raise ValueError("offset and scale must have equal lengths")
        if np.any(~np.isfinite(offset)):
            raise ValueError("offset must be finite")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)

    def _casadi_constants(self) -> tuple[ca.DM, ca.DM]:
        return ca.DM(self.offset), ca.DM(self.scale)

    def encode(self, values: Any) -> Any:
        """Map physical points to normalized coordinates."""
        if isinstance(values, (ca.DM, ca.SX, ca.MX)):
            offset, scale = self._casadi_constants()
            return (values - offset) / scale
        array = _as_vector(values, "values")
        if array.size != self.offset.size:
            raise ValueError("values has the wrong length")
        return (array - self.offset) / self.scale

    def decode(self, values: Any) -> Any:
        """Map normalized points to physical coordinates."""
        if isinstance(values, (ca.DM, ca.SX, ca.MX)):
            offset, scale = self._casadi_constants()
            return offset + scale * values
        array = _as_vector(values, "values")
        if array.size != self.offset.size:
            raise ValueError("values has the wrong length")
        return self.offset + self.scale * array

    def encode_bounds(self, lower: Any, upper: Any) -> tuple[Any, Any]:
        """Map physical bounds, retaining either sign of infinity."""
        return self.encode(
            _bounds_vector(lower, self.offset.size, "lower")
        ), self.encode(_bounds_vector(upper, self.offset.size, "upper"))

    def decode_bounds(self, lower: Any, upper: Any) -> tuple[Any, Any]:
        """Map normalized bounds back to physical coordinates."""
        return self.decode(
            _bounds_vector(lower, self.offset.size, "lower")
        ), self.decode(_bounds_vector(upper, self.offset.size, "upper"))


def _bounds_vector(value: Any, size: int, name: str) -> Any:
    if isinstance(value, (ca.DM, ca.SX, ca.MX)):
        if value.numel() != size:
            raise ValueError(f"{name} has the wrong length")
        return value
    array = _as_vector(value, name)
    if array.size != size:
        raise ValueError(f"{name} has the wrong length")
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} contains NaN")
    return array


def _safe_distance(first: float, second: float) -> float:
    """Return a finite distance, saturating an overflowing difference."""
    distance = abs(first * 0.5 - second * 0.5) * 2.0
    return min(distance, float(np.finfo(float).max))


def _derive_variable_scaling(
    lower: Any, guess: Any, upper: Any
) -> _VariableScaling:
    """Derive a stable positive affine transform from bounds and a guess.

    Finite intervals use their midpoint and half-span. One-sided intervals
    are centered at the finite bound (or guess when both bounds are infinite)
    and use a scale at least one. Fixed coordinates use identity scaling
    around the fixed physical value.
    """
    lo = _as_vector(lower, "lower")
    x0 = _as_vector(guess, "guess")
    hi = _as_vector(upper, "upper")
    if not (lo.size == x0.size == hi.size):
        raise ValueError("lower, guess, and upper must have equal lengths")
    if np.any(~np.isfinite(x0)):
        raise ValueError("guess must contain only finite values")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("bounds must not contain NaN")
    if np.any(lo > hi):
        raise ValueError("lower bounds must not exceed upper bounds")

    offset = np.empty_like(x0)
    scale = np.empty_like(x0)
    for i, (lower_i, guess_i, upper_i) in enumerate(zip(lo, x0, hi)):
        finite_lower = np.isfinite(lower_i)
        finite_upper = np.isfinite(upper_i)
        if finite_lower and finite_upper:
            midpoint = lower_i * 0.5 + upper_i * 0.5
            half_span = abs(upper_i * 0.5 - lower_i * 0.5)
            if half_span == 0.0:
                offset[i], scale[i] = lower_i, 1.0
            else:
                offset[i], scale[i] = midpoint, half_span
        elif finite_lower:
            offset[i], scale[i] = lower_i, max(
                1.0, _safe_distance(guess_i, lower_i)
            )
        elif finite_upper:
            offset[i], scale[i] = upper_i, max(
                1.0, _safe_distance(guess_i, upper_i)
            )
        else:
            offset[i], scale[i] = guess_i, max(1.0, abs(guess_i))

    if np.any(~np.isfinite(offset)) or np.any(~np.isfinite(scale)):
        raise ValueError("could not derive finite scaling")
    return _VariableScaling(offset, scale)
