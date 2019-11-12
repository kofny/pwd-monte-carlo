from combine.indicator.combination_indicator import CombinationIndicator


class ContainDegree(CombinationIndicator):
    def can_combine(self, master_set: set, servant_set: set) -> bool:
        """

        :param master_set: main set
        :param servant_set:
        :return: whether can be combined or not
        """
        intersection = set.intersection(master_set, servant_set)
        return (len(intersection) / len(master_set)) >= self.threshold

