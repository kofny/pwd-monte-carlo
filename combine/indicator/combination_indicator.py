import abc


class CombinationIndicator(metaclass=abc.ABCMeta):
    def __init__(self, threshold: float):
        self.__threshold: float = threshold

    @property
    def threshold(self):
        return self.__threshold

    @threshold.setter
    def threshold(self, new_threshold: float):
        self.__threshold = new_threshold
        pass

    @abc.abstractmethod
    def can_combine(self, master_set: set, servant_set: set) -> bool:
        pass
