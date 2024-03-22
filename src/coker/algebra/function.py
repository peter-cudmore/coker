from typing import Callable


class FunctionSignature:
    def __init__(self, input_shape, output_shape, continuity_class=0):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.continuity_class = continuity_class


class Function:
    def __init__(self,
                 signature: FunctionSignature,
                 implementation: Callable):
        self.signature = signature
        self.implementation = implementation

    @property
    def shape(self):
        pass

    def __call__(self, *args):
        pass