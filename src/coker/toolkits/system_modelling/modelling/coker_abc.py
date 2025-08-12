import abc


class CokerListableSubclasses:

    @classmethod
    def list_subclasses(cls):
        return list(cls.__subclasses__())
