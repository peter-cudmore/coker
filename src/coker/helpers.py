from typing import Type


def get_all_subclasses(cls: Type):
    return {
        item.__name__: inspect.getsourcefile(item)
        for item in cls.__subclasses__()
    }
