import pytest
import random

import torch

from torch_runstats._util import _pad_dim0


@pytest.mark.parametrize("ndim", [1, 2, 4])
def test_pad_dim0(ndim):
    orig_shape = tuple(random.randint(1, 5) for _ in range(ndim))
    x = torch.ones(orig_shape)
    to_add = 3
    padded = _pad_dim0(x, to_add)
    assert padded.shape[1:] == orig_shape[1:]
    assert padded.shape[0] == orig_shape[0] + to_add
    assert torch.equal(x, padded[:-to_add])
    assert padded[-to_add:].abs().max() == 0
