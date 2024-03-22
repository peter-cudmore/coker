
class Dimension:
    def __init__(self, tuple_or_none):
        if isinstance(tuple_or_none, int):
            tuple_or_none = (tuple_or_none, )

        assert tuple_or_none is None or isinstance(tuple_or_none, tuple)
        self.dim = tuple_or_none

    def __eq__(self, other):
        return self.dim == other.dim

    def is_scalar(self):
        return self.dim is None

    def is_vector(self):
        return self.dim is not None and len(self.dim) == 1

    def is_covector(self):
        return not self.is_scalar() and len(self.dim) == 2 and self.dim[0] == 1

    def is_matrix(self):
        return not self.is_scalar() and len(self.dim) == 2 and self.dim[0] > 1

    def is_multilinear_map(self):
        return isinstance(self.dim, tuple) and len(self.dim) > 2

    def __iter__(self):
        return iter(self.dim)

    def __repr__(self):
        if self.dim is None:
            return "R"

        return repr(self.dim)

