import numpy as np


class TernGrad:
    def step(self, z_t_i):
        """
        Input: z_t_i, a part of a mini-batch of training samples z_t

        Compute gradients g_t_i under z_t_i
        Ternarize gradients to g_t_i_tern = ternarize(g_t_i)
        Push ternary g_t_i_tern to the server
        Pull averaged gradients g_t_average from the server
        Update parameters w_t_next = w_t - step * g_t_average
        """
        g_t_i = np.gradient(z_t_i)
        g_t_i_tern = self.ternarize(g_t_i=g_t_i)
        return g_t_i_tern

    @staticmethod
    def ternarize(g_t_i):
        """
        g_t_i_tern = s_t * sign(g_t_i) o b_t
        s_t = max(abs(g_t_i)) = ||g_t_i||∞ (max norm)
        o : Hadamard product
        """
        s_t = max(abs(g_t_i))
        b_t = [1 if abs(g)/s_t >= 0.5 else 0 for g in g_t_i]
        print("b_t:", b_t)
        print("sign:", np.sign(g_t_i)*b_t)
        g_t_i_tern = s_t * np.sign(g_t_i) * b_t
        print("g_t_i_tern:", g_t_i_tern)
        return g_t_i_tern

