from typing import Optional
import functools

import pytest

import random

import torch
from torch_scatter import scatter

from torch_runstats import RunningStats, Reduction


@pytest.fixture(scope="module")
def allclose(float_tolerance):
    return functools.partial(torch.allclose, atol=float_tolerance)


class StatsTruth:
    """Inefficient ground truth for RunningStats."""

    def __init__(
        self, dim=1, reduction: Reduction = Reduction.MEAN, reduce_dims=tuple()
    ):
        if isinstance(dim, int):
            self._dim = (dim,)
        else:
            self._dim = tuple(dim)
        self._reduction = reduction
        self._reduce_dims = tuple(reduce_dims)
        self._n_bins = 0
        self.reset()

    def accumulate_batch(
        self, batch: torch.Tensor, accumulate_by: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if accumulate_by is None:
            accumulate_by = torch.zeros(len(batch), dtype=torch.long)
        if hasattr(self, "_state"):
            self._state = torch.cat((self._state, batch), dim=0)
            self._acc = torch.cat((self._acc, accumulate_by), dim=0)
        else:
            self._state = batch.clone()
            self._acc = accumulate_by.clone()

        if self._acc.max() + 1 > self._n_bins:
            self._n_bins = int(self._acc.max() + 1)

        if accumulate_by is None:
            reduce_in_dims = tuple(i + 1 for i in self._reduce_dims)
            if self._reduction == Reduction.MEAN:
                return batch.mean(dim=(0,) + reduce_in_dims)
            elif self._reduction == Reduction.RMS:
                return batch.square().mean(dim=(0,) + reduce_in_dims).sqrt()
        else:
            N = torch.bincount(accumulate_by)
            for d in self._reduce_dims:
                N *= self._dim[d]
            N = 1 / N
            torch.nan_to_num_(N, nan=0.0, posinf=0.0, neginf=0.0)

            if self._reduction == Reduction.RMS:
                batch = batch.square()

            if len(self._reduce_dims) > 0:
                batch = batch.sum(dim=tuple(i + 1 for i in self._reduce_dims))

            N = N.reshape((len(N),) + (1,) * (batch.ndim - 1))

            if self._reduction == Reduction.MEAN:
                return N * scatter(
                    batch,
                    accumulate_by,
                    reduce="sum",
                    dim=0,
                )
            elif self._reduction == Reduction.RMS:
                return (
                    N
                    * scatter(
                        batch,
                        accumulate_by,
                        reduce="sum",
                        dim=0,
                    )
                ).sqrt()

    def reset(self) -> None:
        if hasattr(self, "_state"):
            delattr(self, "_state")
            delattr(self, "_acc")

    def current_result(self):
        if not hasattr(self, "_state"):
            return torch.zeros(self._dim)
        N = torch.bincount(self._acc, minlength=self._n_bins)
        for d in self._reduce_dims:
            N *= self._dim[d]
        N = 1 / N
        torch.nan_to_num_(N, nan=0.0, posinf=0.0, neginf=0.0)

        if self._reduction == Reduction.RMS:
            state = self._state.square()
        else:
            state = self._state

        if len(self._reduce_dims) > 0:
            state = state.sum(dim=tuple(i + 1 for i in self._reduce_dims))

        N = N.reshape((len(N),) + (1,) * (state.ndim - 1))

        out = N * scatter(state, self._acc, dim=0, reduce="sum", dim_size=self._n_bins)

        if self._reduction == Reduction.RMS:
            out.sqrt_()

        return out


@pytest.mark.parametrize(
    "dim,reduce_dims",
    [
        (1, tuple()),
        (3, tuple()),
        ((2, 3), tuple()),
        (torch.Size((1, 2, 1)), tuple()),
        (torch.Size((1, 2, 1)), (1,)),
        (torch.Size((3, 2, 4)), (0, 2)),
    ],
)
@pytest.mark.parametrize("reduction", [Reduction.MEAN, Reduction.RMS])
@pytest.mark.parametrize("do_accumulate_by", [True, False])
def test_runstats(dim, reduce_dims, reduction, do_accumulate_by, allclose):
    n_batchs = (random.randint(1, 4), random.randint(1, 4))
    truth_obj = StatsTruth(dim=dim, reduction=reduction, reduce_dims=reduce_dims)
    runstats = RunningStats(dim=dim, reduction=reduction, reduce_dims=reduce_dims)

    for n_batch in n_batchs:
        for _ in range(n_batch):
            batch = torch.randn((random.randint(1, 10),) + runstats.dim)
            if do_accumulate_by and random.choice((True, False)):
                accumulate_by = torch.randint(
                    0, random.randint(1, 5), size=(batch.shape[0],)
                )
                truth = truth_obj.accumulate_batch(batch, accumulate_by=accumulate_by)
                res = runstats.accumulate_batch(batch, accumulate_by=accumulate_by)
            else:
                truth = truth_obj.accumulate_batch(batch)
                res = runstats.accumulate_batch(batch)
            assert allclose(truth, res)
        assert allclose(truth_obj.current_result(), runstats.current_result())
        truth_obj.reset()
        runstats.reset()


@pytest.mark.parametrize("reduction", [Reduction.MEAN, Reduction.RMS])
def test_zeros(reduction, allclose):
    dim = (4,)
    runstats = RunningStats(dim=dim, reduction=reduction)
    assert allclose(runstats.current_result(), torch.zeros(dim))
    runstats.accumulate_batch(torch.randn((3,) + dim))
    runstats.reset()
    assert allclose(runstats.current_result(), torch.zeros(dim))


def test_raises():
    runstats = RunningStats(dim=4, reduction=Reduction.MEAN)
    with pytest.raises(AssertionError):
        runstats.accumulate_batch(torch.zeros(10, 2))


@pytest.mark.parametrize("dim", [1, 3, (2, 3), torch.Size((1, 2, 1))])
@pytest.mark.parametrize("reduction", [Reduction.MEAN, Reduction.RMS])
@pytest.mark.parametrize("do_accumulate_by", [True, False])
def test_one_acc(dim, reduction, do_accumulate_by, allclose):
    runstats = RunningStats(dim=dim, reduction=reduction)
    batch = torch.randn((random.randint(3, 10),) + runstats.dim)
    if do_accumulate_by:
        accumulate_by = torch.randint(0, random.randint(1, 5), size=(batch.shape[0],))
        if reduction == Reduction.MEAN:
            truth = scatter(batch, accumulate_by, dim=0, reduce="mean")
        elif reduction == Reduction.RMS:
            truth = scatter(batch.square(), accumulate_by, dim=0, reduce="mean").sqrt()
        res = runstats.accumulate_batch(batch, accumulate_by=accumulate_by)
    else:
        if reduction == Reduction.MEAN:
            truth = batch.mean(dim=0)
        elif reduction == Reduction.RMS:
            truth = batch.square().mean(dim=0).sqrt()
        res = runstats.accumulate_batch(batch)
    assert allclose(truth, res)
    assert allclose(truth, runstats.current_result())
