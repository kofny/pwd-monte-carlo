from enum import Enum, unique
from combine.indicator.combination_indicator import CombinationIndicator
from combine.indicator.dice import Dice
from combine.indicator.contain_degree import ContainDegree
from combine.indicator.ochiai import Ochiai
from combine.indicator.simpson import Simpson
from combine.indicator.jaccard import Jaccard


@unique
class Indicator(Enum):
    Jaccard = 1
    Ochiai = 2
    Simpson = 3
    Dice = 4
    ContainDegree = 5


indicator_map = {
    Indicator.Jaccard: Jaccard,
    Indicator.Ochiai: Ochiai,
    Indicator.Simpson: Simpson,
    Indicator.Dice: Dice,
    Indicator.ContainDegree: ContainDegree
}


class IndicatorFactory:
    def __init__(self):
        pass

    @staticmethod
    def build(indicator: Indicator, threshold: float) -> CombinationIndicator:
        return indicator_map.get(indicator)(threshold)
        pass
