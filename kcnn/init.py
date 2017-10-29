import math
import torch
from torch.autograd import Variable

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`U(-a, a)` where
    :math:`a = gain \\times \sqrt{2 / (fan\_in + fan\_out)} \\times \sqrt{3}`.
    Also known as Glorot initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`N(0, std)` where
    :math:`std = gain \\times \sqrt{2 / (fan\_in + fan\_out)}`.
    Also known as Glorot initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_normal(w)
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


def orthogonal(tensor, gain=1):
    """Fills the input Tensor or Variable with a (semi) orthogonal matrix, as
    described in "Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks" - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable, where n >= 2
        gain: optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.orthogonal(w)
    """
    if isinstance(tensor, Variable):
        orthogonal(tensor.data, gain=gain)
        return tensor

    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor