from typing import List, Optional, Tuple

from coker.dynamics.controls import (
    ControlVariable,
    PiecewiseConstantVariable,
    SpikeVariable,
)
from coker.dynamics.variational.problem import TranscriptionOptions


def split_at_non_differentiable_points(
    control_variables: List[ControlVariable],
    t_final: float,
    transcription_options: TranscriptionOptions,
    additional_points: Optional[List[float]] = None,
) -> List[Tuple[float, float]]:
    """
    Splits the time domain into intervals taking into account
    control variables, additional integration points, and
    transcription requirements.

    The function determines interval boundaries based on the
    characteristics and constraints of the provided control
    variables. It also considers any additional points passed,
    ensuring subdivision complies with the transcription options.
    The intervals are adjusted to guarantee a minimum required
    number and their lengths are subdivided iteratively when
    necessary.

    Args:
        control_variables (List[ControlVariable]): A list of
            control variables, which define parameters affecting
            the differentiation process. This may include variables
            such as piecewise constants or spike events.
        t_final (float): The total duration of the time domain or
            integration window.
        transcription_options (TranscriptionOptions): Configuration
            options specifying transcription constraints, such as
            the minimum number of intervals.
        additional_points (Optional[List[float]]): Additional time
            points that should be included as interval boundaries,
            if provided.

    Returns:
        List[Tuple[float, float]]: A list of tuples representing
            the sorted interval boundaries. Each tuple contains the
            start and end of an interval.

    """
    interval_boundaries = (
        set(additional_points) if additional_points else set()
    )
    for d in control_variables:
        if isinstance(d, PiecewiseConstantVariable):
            assert d.sample_rate > 0, "Sample rate must be positive"
            steps = t_final * d.sample_rate
            interval_boundaries |= {
                i * t_final / steps for i in range(int(steps))
            }

        if isinstance(d, SpikeVariable):
            assert (
                0 <= d.time < t_final
            ), "Spike time must be within integration window"

            interval_boundaries.add(d.time)

    if 0 not in interval_boundaries:
        interval_boundaries.add(0)

    if t_final not in interval_boundaries:
        interval_boundaries.add(t_final)

    sorted_boundaries = list(interval_boundaries)

    sorted_boundaries.sort()

    while len(sorted_boundaries) < transcription_options.minimum_n_intervals:
        intervals = [
            (stop - start, (stop + start) / 2)
            for start, stop in zip(
                sorted_boundaries[:-1], sorted_boundaries[1:]
            )
        ]
        max_length = max(length for length, _ in intervals)

        if all(length == max_length for length, _ in intervals):
            sorted_boundaries += [mid for _, mid in intervals]
        else:
            sorted_boundaries += [
                point for length, point in intervals if point == max_length
            ]
        sorted_boundaries.sort()

    return [
        (start, stop)
        for start, stop in zip(sorted_boundaries[:-1], sorted_boundaries[1:])
    ]
