import unittest
import numpy as np
from src.compressions.TernGrad import *


class TestTernGrad(unittest.TestCase):
    def test_ternarize(self):
        tern = TernGrad()
        self.assertTrue(np.array_equal(tern.ternarize(np.array([0.3, -1.2, 0.9])),
                                       np.array([0, -1.2, 1.2])), "Not equal.")


if __name__ == '__main__':
    unittest.main()
