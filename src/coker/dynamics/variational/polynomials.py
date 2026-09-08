from typing import Callable, Iterator, List, Tuple

import numpy as np

from coker.dynamics.transcription.collocation import InterpolatingPoly


class InterpolatingPolyCollection:
    def __init__(self, polys: List[InterpolatingPoly]):
        self.polys = polys
        self._size = sum(p.size() for p in polys)
        self.intervals = [p.interval for p in polys]

    def size(self):
        return self._size

    def __call__(self, t):
        for i, (start, end) in enumerate(self.intervals):
            if start <= t <= end:
                return self.polys[i](t)
        raise ValueError(f"Value {t} is not in any interval")

    def interval_starts(self):
        for p in self.polys:
            yield p.start_point()

    def interval_ends(self):
        for p in self.polys:
            yield p.end_point()

    def knot_points(self) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
        for p in self.polys:
            for point in p.knot_points():
                yield point

    def map(self, func: Callable[[float, np.ndarray], np.ndarray]):
        new_polys = [p.map(func) for p in self.polys]
        return InterpolatingPolyCollection(new_polys)
