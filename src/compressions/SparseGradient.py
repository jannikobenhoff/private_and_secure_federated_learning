import numpy as np


class SparseGradient:
    def step(self):
        """
        """
        pass

    def gradDrop(self, gradient, drop_rate):
        """
        Updates by removing drop_rate % of the smallest gradients by absolute value
        TODO: Remember residuals (dropped values) locally to add to next gradient
              before dropping again
        """
        threshold = np.percentile(abs(gradient), drop_rate)
        gradient[abs(gradient) <= threshold] = 0
        return gradient

