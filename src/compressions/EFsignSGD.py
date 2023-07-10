import numpy as np


class EFsignSGD:
    def step(self, learn_rate, g_t, e_t, x_t):
        p_t = learn_rate*g_t+e_t
        delta_t = np.sign(p_t) * ()
        x_t_next = x_t - delta_t
        e_t_next = p_t - delta_t

