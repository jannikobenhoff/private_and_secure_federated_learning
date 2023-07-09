import numpy as np


class NaturalCompression:
    @staticmethod
    def compress(t):
        """
        Performing a randomized logarithmic rounding of input t
        """
        if t == 0:
            return 0
        else:
            a = np.log2(abs(t))
            a_up = np.ceil(a)
            a_down = np.floor(a)

            # Probability of whether to take floor or ceiling of a
            p_down = (np.power(2, a_up)-abs(t))/(np.power(2, a_down))
            if p_down >= 0.5:
                return np.sign(t)*np.power(2, a_down)
            else:
                return np.sign(t)*np.power(2, a_up)

