from abc import ABCMeta


class System(metaclass=ABCMeta):
    pass


def controller_main(build):
    def controller_main_decorator(cls):
        assert issubclass(cls, System)

        return cls
