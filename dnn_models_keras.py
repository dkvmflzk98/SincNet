import numpy as np


def flip(x, dim):
    x_size = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *x_size[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(np.arange(x.size(1)-1, -1, -1),
                                                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(x_size)


def sinc(band, t_right):
    y_right = np.sin(2 * np.pi * band * t_right)/(2 * np.pi * band * t_right)
    y_left = flip(y_right, 0)

    y = np.concatenate([y_left, np.ones(1).data.cuda(), y_right])

    return y
