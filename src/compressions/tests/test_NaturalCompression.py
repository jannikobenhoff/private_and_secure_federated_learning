import unittest
import numpy as np
from src.compressions.NaturalCompression import *


class TestNaturalCompression(unittest.TestCase):
    def test_compression(self):
        nc = NaturalCompression()
        c_nat = nc.compress(t=-2.75)
        self.assertTrue(c_nat == -2, "Not equal.")


if __name__ == '__main__':
    unittest.main()
