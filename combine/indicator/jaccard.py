from combine.indicator.combination_indicator import CombinationIndicator


class Jaccard(CombinationIndicator):

    def similarity(self, master_set: set, servant_set: set) -> float:
        intersection = set.intersection(master_set, servant_set)
        union = set.union(master_set, servant_set)
        return len(intersection) / len(union)
