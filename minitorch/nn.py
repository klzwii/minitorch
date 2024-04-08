from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import (
    Tensor
)
from .tensor_functions import Function, rand, tensor
import numpy as np

from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    k_tensor = input.zeros((2,))
    k_tensor[0], k_tensor[1] = kh, kw
    return Tile.apply(input, k_tensor), height//kh, height//kw


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    mid_tensor, nh, nw = tile(input=input, kernel=kernel)
    return AvgPool2d.apply(mid_tensor)

class Tile(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tensor) -> Tensor:
        batch, channel, height, width = input.shape
        kh, kw = int(kernel[0]), int(kernel[1])
        shape = [batch, channel, height // kh, width // kw, kh, kw]
        out_tensor = input.zeros(tuple(shape))
        for ba in range(batch):
            for ch in range(channel):
                for h in range(shape[2]):
                    for w in range(shape[3]):
                        for th in range(kh):
                            for tw in range(kw):
                                out_tensor[ba, ch, h, w, th, tw] = input[ba, ch, h*kh+th, w*kw+tw]
        ctx.save_for_backward(input, kernel)
        return out_tensor 

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, kernel = ctx.saved_values
        batch, channel, height, width = input.shape
        kh, kw = int(kernel[0]), int(kernel[1])
        shape = [batch, channel, height // kh, width // kw, kh, kw]
        grad_tensor = input.zeros()
        for ba in range(batch):
            for ch in range(channel):
                for h in range(shape[2]):
                    for w in range(shape[3]):
                        for th in range(kh):
                            for tw in range(kw):
                                grad_tensor[ba, ch, h * kh + th, w * kw + tw] = grad_output[ba, ch, h, w, th, tw]
        return grad_tensor, grad_output.zeros((2,))


class AvgPool2d(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor) -> Tensor:
        batch, channel, height, width, kh, kw = input.shape
        k_size = kh*kw
        shape = [batch, channel, height, width]
        out_tensor = input.zeros(tuple(shape))
        for ba in range(batch):
            for ch in range(channel):
                for h in range(height):
                    for w in range(width):
                        temp = 0.
                        for th in range(kh):
                            for tw in range(kw):
                                temp += input[ba, ch, h, w, th, tw]
                        out_tensor[ba, ch, h, w] = temp/k_size
        ctx.save_for_backward(input, out_tensor)
        return out_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        input, out_tensor = ctx.saved_values
        batch, channel, height, width, kh, kw = input.shape
        k_size = kh*kw
        grad_tensor = input.zeros()
        for ba in range(batch):
            for ch in range(channel):
                for h in range(height):
                    for w in range(width):
                        grad = grad_output[ba, ch, h, w] / k_size
                        for th in range(kh):
                            for tw in range(kw):
                                grad_tensor[ba, ch, h, w, th, tw] = grad
        return grad_tensor

class MaxPool2D(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor) -> Tensor:
        batch, channel, height, width, kh, kw = input.shape
        k_size = kh*kw
        shape = [batch, channel, height, width]
        out_tensor = input.zeros(tuple(shape))
        for ba in range(batch):
            for ch in range(channel):
                for h in range(height):
                    for w in range(width):
                        temp = input[ba, ch, h, w, 0, 0]
                        for th in range(kh):
                            for tw in range(kw):
                                p = input[ba, ch, h, w, th, tw]
                                temp = p if p > temp else temp
                        out_tensor[ba, ch, h, w] = temp
        ctx.save_for_backward(input, out_tensor)
        return out_tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        input, out_tensor = ctx.saved_values
        batch, channel, height, width, kh, kw = input.shape
        k_size = kh*kw
        grad_tensor = input.zeros()
        for ba in range(batch):
            for ch in range(channel):
                for h in range(height):
                    for w in range(width):
                        grad = grad_output[ba, ch, h, w] / k_size
                        for th in range(kh):
                            for tw in range(kw):
                                grad_tensor[ba, ch, h, w, th, tw] = 1.0 if out_tensor[ba, ch, h, w] == input[ba, ch, h, w, th, tw] else 0.
        return grad_tensor


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        dim=int(dim[0])
        in_shape = input.shape
        out_shape = list(in_shape)
        out_shape[dim] = 1
        out_tensor = input.zeros(shape=tuple(out_shape))
        in_size = 1
        for _, d in enumerate(in_shape):
            in_size *= d
        for i in range(in_size):
            in_idx = np.zeros(len(in_shape), dtype=np.int32)
            to_index(i, input.shape, in_idx)
            in_idx = tuple(in_idx)
            if in_idx[dim] == 0:
                out_tensor[in_idx] = input[in_idx]
            else:
                out_idx = list(in_idx)
                out_idx[dim] = 0
                out_idx=tuple(out_idx)
                out_tensor[out_idx] = out_tensor[out_idx] if out_tensor[out_idx] > input[in_idx] else input[in_idx]
        ctx.save_for_backward(input, out_tensor, dim, in_size)
        return out_tensor
            

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        input, out_tensor, dim, in_size = ctx.saved_values
        out_tensor = input.zeros()
        for i in range(in_size):
            in_idx = np.zeros(len(input.shape), dtype=np.int32)
            to_index(i, input.shape, in_idx)
            out_idx = in_idx
            out_idx[dim] = 0
            in_idx = tuple(in_idx)
            out_idx = tuple(out_idx)
            input[in_idx] = grad_output[out_idx] if out_tensor[out_idx] == input[in_idx] else 0.0
        return out_tensor, 0.


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))

def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    exp_tensor = input.exp()
    sum_tensor = exp_tensor.sum(dim)
    return exp_tensor / sum_tensor



def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    return input - input.exp().sum(dim).log()



def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    mid_tensor, nh, nw = tile(input=input, kernel=kernel)
    return MaxPool2D.apply(mid_tensor)


class Dropout(Function):
    @staticmethod
    def forward(ctx: Context, input:Tensor, rate: Tensor):
        rate = rate[0]
        mask = [input._tensor._storage[i] if not np.random.rand() < rate else 0. for i in range(input.size)]
        out_tensor = input.make(mask, input.shape, input._tensor.strides, input.backend)
        ctx.save_for_backward(mask)
        return out_tensor
    
    @staticmethod
    def backward(ctx: Context, grad_out:Tensor) -> Tuple[Tensor, float]:
        (mask,) = ctx.saved_values
        out_tensor = grad_out.make([0.0 for i in range(grad_out.size)], grad_out.shape, grad_out._tensor.strides, grad_out.backend)
        for i in range(out_tensor.size):
            if mask[i]:
                out_tensor._tensor._storage[i] = 0.
        return out_tensor, 0.

def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore:
        return input
    return Dropout.apply(input, input._ensure_tensor(rate))
