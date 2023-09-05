import torch
import tensorflow as tf


def scaled_sign(x):
    """
    :param x: torch Tensor
    :return: The sign tensor scaled by it's L1 norm and divided by the number of elements
    """
    print(tf.norm(tf.convert_to_tensor(x), ord=1))
    print(tf.size(tf.convert_to_tensor(x)))
    print(x.norm(p=1))
    return x.norm(p=1) / x.nelement() * torch.sign(x)


print(scaled_sign(torch.Tensor([1, -2, 3])))
