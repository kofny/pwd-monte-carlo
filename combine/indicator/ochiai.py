from combine.indicator.combination_indicator import CombinationIndicator
import math


class Ochiai(CombinationIndicator):

    def similarity(self, master_set: set, servant_set: set) -> float:
        intersection = set.intersection(master_set, servant_set)
        sqrt = math.sqrt(len(master_set)) * math.sqrt(len(servant_set))
        return len(intersection) / sqrt
        pass
