import numpy as np

class SignSGD:
    def compress(self, gradient):
        return np.sign(gradient)
