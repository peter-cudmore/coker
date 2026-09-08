from dataclasses import dataclass

import casadi as ca


@dataclass(frozen=True)
class DecisionLayout:
    """Contiguous decision-vector blocks and their associated values."""

    horizon_size: int
    path_size: int
    control_size: int
    parameter_size: int
    horizon_lower: float = -ca.inf
    horizon_guess: float = 1.0
    horizon_upper: float = ca.inf

    @property
    def horizon_slice(self):
        return slice(0, self.horizon_size)

    @property
    def parameter_slice(self):
        start = self.horizon_size + self.path_size + self.control_size
        return slice(start, start + self.parameter_size)

    def vector(self, horizon, path, control, parameters):
        blocks = [path, control, parameters]
        if self.horizon_size:
            blocks.insert(0, horizon)
        return ca.vertcat(*blocks)

    def bounds(self, path_lower, control_lower, parameter_lower):
        horizon = (
            ca.DM(
                [
                    self.horizon_lower,
                ]
            )
            if self.horizon_size
            else ca.DM.zeros(0, 1)
        )
        return self.vector(
            horizon,
            path_lower,
            ca.vertcat(*control_lower),
            parameter_lower,
        )

    def guess(self, path, control, parameters):
        horizon = (
            ca.DM([self.horizon_guess])
            if self.horizon_size
            else ca.DM.zeros(0, 1)
        )
        return self.vector(horizon, path, control, parameters)

    def upper_bounds(self, path_upper, control_upper, parameter_upper):
        horizon = (
            ca.DM([self.horizon_upper])
            if self.horizon_size
            else ca.DM.zeros(0, 1)
        )
        return self.vector(
            horizon,
            path_upper,
            ca.vertcat(*control_upper),
            parameter_upper,
        )
