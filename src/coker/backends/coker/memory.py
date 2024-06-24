import dataclasses


@dataclasses.dataclass
class MemorySpec:
    location: int
    count: int

    def __hash__(self):
        return hash((self.location, self.count))
