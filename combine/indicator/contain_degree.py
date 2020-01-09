from combine.indicator.combination_indicator import CombinationIndicator


class ContainDegree(CombinationIndicator):
    def similarity(self, master_set: set, servant_set: set) -> float:
        """

        :param master_set: main set
        :param servant_set:
        :return: whether can be combined or not
        """
        intersection = set.intersection(master_set, servant_set)
        return len(intersection) / len(master_set)
