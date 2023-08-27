import numpy as np
import tensorflow as tf
import torch
from scipy.optimize import nnls
from tensorflow import Tensor

from .Compression import Compression


class vqSGD(Compression):
    def __init__(self, repetition: int = 1, name="vqSGD"):
        super().__init__(name=name)
        self.s = repetition
        self.compression_rates = []

    def build(self, var_list):
        """Initialize optimizer variables.

        vqSGD optimizer has no variables.

        Args:
          var_list: list of model variables to build vqSGD variables on.
        """
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

    def compress(self, gradient: Tensor, variable) -> Tensor:
        """
        Quantise gradient: Q(v) = c_i with probability a_i
        c_i of 2*d point set {+- sqrt(d) e_i | i e [d]}
        cp = tf.concat([ d_sqrt * tf.eye(d), - d_sqrt * tf.eye(d)], axis=1)

        probability algorithm:
        for i in range(2 * d):
            if gradient[i % d] > 0 and i <= d - 1:
                a[i] = gradient[i % d] / d_sqrt + gamma / (2 * d)
            elif gradient[i % d] <= 0 and i > d - 1:
                a[i] = -gradient[i % d] / d_sqrt + gamma / (2 * d)
            else:
                a[i] = gamma / (2 * d)
        """
        input_shape = gradient.shape
        gradient = tf.where(tf.math.is_nan(gradient), 0., gradient)

        l2 = tf.norm(gradient, ord=2)
        # if l2 != 0:
        if l2 > 1:
            gradient = tf.reshape(gradient, [-1]) / l2
        else:
            gradient = tf.reshape(gradient, [-1])

        d = gradient.shape[0]
        d_sqrt = np.sqrt(d)

        a = np.zeros(2 * d)

        gamma = 1 - tf.norm(gradient, ord=1) / d_sqrt
        gamma_by_2d = gamma / (2 * d)

        a[:d] = tf.cast(gradient > 0, tf.float32) * ((gradient / d_sqrt) + gamma_by_2d)
        a[d:] = tf.cast(gradient <= 0, tf.float32) * ((-gradient / d_sqrt) + gamma_by_2d)

        a = tf.where(a == 0, gamma_by_2d, a)

        a = a.numpy()
        a[np.isnan(a)] = 0
        np.divide(a, a.sum(), out=a)

        indices = np.random.choice(np.arange(2 * d), self.s, p=a)
        compressed_gradient = np.zeros(d)

        for index in indices:
            if index >= d:
                compressed_gradient[index - d] -= d_sqrt
            else:
                compressed_gradient[index] += d_sqrt

        compressed_gradient = tf.reshape(compressed_gradient, input_shape) / self.s
        compressed_gradient = tf.cast(compressed_gradient, dtype=variable.dtype)

        if variable.ref() not in self.cr:
            self.cr[variable.ref()] = gradient.dtype.size * 8 * np.prod(
                gradient.shape.as_list()) / self.get_sparse_tensor_size_in_bits(
                compressed_gradient)
            self.compression_rates.append(self.cr[variable.ref()])

        return compressed_gradient * l2


code_books = {}


def get_code_book(args, dim, ks):
    if (dim, ks) not in code_books:
        location = './codebooks/{}/angular_dim_{}_Ks_{}.fvecs'
        location = location.format('kmeans_codebook', dim, ks)
        codewords = fvecs_read(location)
        book = torch.from_numpy(codewords)

        if args.gpus is not None:
            book = book.cuda()
        book /= torch.norm(book, dim=1, keepdim=True)[0]
        code_books[(dim, ks)] = book
        return book
    else:
        return code_books[(dim, ks)]


class VQSGD(torch.optim.SGD):
    def __init__(self, myargs, *args, **kwargs):
        super(VQSGD, self).__init__(*args, **kwargs)
        self.args = myargs
        self.dim = myargs.dim
        self.ks = myargs.ks
        self.rate = myargs.rate
        print('VQSGD, rate = {}'.format(self.rate))
        self.code_books = get_code_book(self.args, self.dim, self.ks)

    def vq_gd(self, p, lr, d_p):
        code_books = self.code_books
        l = 1 / self.rate * lr
        u = self.rate * lr
        x = p.data.reshape(-1, self.dim, 1)
        grad = d_p.reshape(-1, self.dim, 1)
        M = x.size(0)
        W = x.expand(-1, -1, self.ks)
        F = grad.expand(-1, -1, self.ks)
        C = code_books.t().expand(M, self.dim, self.ks)
        Tu = C - W + F.mul(u)
        Tl = C - W + F.mul(l)

        r0 = torch.min(Tu.pow(2), Tl.pow(2))
        r1 = (torch.sign(Tu) + torch.sign(Tl)).pow(2)
        result = 1 / 4 * r0.mul(r1)
        codes = result.sum(dim=1).argmin(dim=1)

        p1 = torch.index_select(
            code_books, 0, codes).reshape(p.data.shape)
        return p1

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                if hasattr(p, 'org'):
                    p.org.copy_(p.data - group['lr'] * d_p)
                if 'name' in group and group['name'] == 'others':
                    p.data.add_(-group['lr'], d_p)
                else:
                    if group['name'] == 'conv2d':
                        tensor = p.data.clone().detach().permute(0, 2, 3, 1).contiguous()
                        p.data = self.vq_gd(
                            tensor, group['lr'], d_p).permute(0, 3, 1, 2)
                    else:
                        p.data = self.vq_gd(p, group['lr'], d_p)

        return loss


if __name__ == "__main__":
    vq = VQSGD()
