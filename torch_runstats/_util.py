import torch


def _pad_dim0(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return
    elif n < 0:
        raise ValueError
    return torch.nn.functional.pad(x, (0,) * ((x.ndim - 1) * 2) + (0, n))
