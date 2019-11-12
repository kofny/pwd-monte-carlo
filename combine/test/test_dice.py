from unittest import TestCase
from combine.indicator.dice import Dice


class TestDice(TestCase):

    def test_can_combine(self):
        dice = Dice()
        ok = dice.can_combine({"hh", "ww"}, {"ww"})
        self.assertEqual(ok, True, "can be combined")

    def test_can_not_be_combined(self):
        dice = Dice()
        not_ok = dice.can_combine({"hh", "h", "hhh", "ww"}, {"h"})
        self.assertEqual(not_ok, False, "can not be combined")
