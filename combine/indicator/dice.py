from combine.indicator.combination_indicator import CombinationIndicator


class Dice(CombinationIndicator):

    def similarity(self, master_set: set, servant_set: set) -> float:
        intersection = set.intersection(master_set, servant_set)
        return 2 * len(intersection) / (len(master_set) + len(servant_set))
