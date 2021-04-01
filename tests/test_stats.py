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
        self,
        dim=1,
        reduction: Reduction = Reduction.MEAN,
        reduce_dims=tuple(),
        weighted=False,
    ):
        if isinstance(dim, int):
            self._dim = (dim,)
        else:
            self._dim = tuple(dim)
        self._reduction = reduction
        if isinstance(reduce_dims, int):
            self._reduce_dims = (reduce_dims,)
        else:
            self._reduce_dims = tuple(reduce_dims)
        self._n_bins = 0
        self.weighted = weighted
        self.reset()

    def accumulate_batch(
        self,
        batch: torch.Tensor,
        accumulate_by: Optional[torch.Tensor] = None,
        weight=None,
    ) -> torch.Tensor:
        if accumulate_by is None:
            accumulate_by = torch.zeros(len(batch), dtype=torch.long)

        if self.weighted:
            assert weight is not None

        if hasattr(self, "_state"):
            self._state = torch.cat((self._state, batch), dim=0)
            self._acc = torch.cat((self._acc, accumulate_by), dim=0)
            if self.weighted:
                self._weights = torch.cat((self._weights, weight), dim=0)
        else:
            self._state = batch.clone()
            self._acc = accumulate_by.clone()
            if self.weighted:
                self._weights = weight.clone()

        if self._acc.max() + 1 > self._n_bins:
            self._n_bins = int(self._acc.max() + 1)

        if self.weighted:
            mul_by = scatter(weight, accumulate_by, dim=0)
            mul_by.sum(dim=self._reduce_dims)
        else:
            mul_by = torch.bincount(accumulate_by)
            for d in self._reduce_dims:
                mul_by *= self._dim[d]
        mul_by = 1.0 / mul_by
        torch.nan_to_num_(mul_by, nan=0.0, posinf=0.0, neginf=0.0)

        if self._reduction == Reduction.RMS:
            batch = batch.square()

        if len(self._reduce_dims) > 0:
            batch = batch.sum(dim=tuple(i + 1 for i in self._reduce_dims))

        mul_by = mul_by.reshape((len(mul_by),) + (1,) * (batch.ndim - 1))

        if self._reduction == Reduction.MEAN:
            return mul_by * scatter(
                batch,
                accumulate_by,
                reduce="sum",
                dim=0,
            )
        elif self._reduction == Reduction.RMS:
            return (
                mul_by
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
            if self.weighted:
                delattr(self, "_weights")

    def current_result(self):
        if not hasattr(self, "_state"):
            return torch.zeros(self._dim)

        if self.weighted:
            mul_by = scatter(self._weights, self._acc, dim=0, dim_size=self._n_bins)
            mul_by.sum(dim=self._reduce_dims)
        else:
            mul_by = torch.bincount(self._acc, minlength=self._n_bins)
            for d in self._reduce_dims:
                mul_by *= self._dim[d]
        mul_by = 1 / mul_by
        torch.nan_to_num_(mul_by, nan=0.0, posinf=0.0, neginf=0.0)

        if self._reduction == Reduction.RMS:
            state = self._state.square()
        else:
            state = self._state

        if len(self._reduce_dims) > 0:
            state = state.sum(dim=tuple(i + 1 for i in self._reduce_dims))

        mul_by = mul_by.reshape((len(mul_by),) + (1,) * (state.ndim - 1))

        out = mul_by * scatter(
            state, self._acc, dim=0, reduce="sum", dim_size=self._n_bins
        )

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
@pytest.mark.parametrize("weighted", [(True, True), (True, False), (False, False)])
def test_runstats(dim, reduce_dims, reduction, do_accumulate_by, weighted, allclose):
    weighted, weight_extra_dims = weighted

    if weighted and reduce_dims != tuple(range(len(reduce_dims))):
        pytest.xfail("Cannot weight when reduce dims are not the first dims")

    n_batchs = (random.randint(1, 4), random.randint(1, 4))
    truth_obj = StatsTruth(
        dim=dim, reduction=reduction, reduce_dims=reduce_dims, weighted=weighted
    )
    runstats = RunningStats(
        dim=dim, reduction=reduction, reduce_dims=reduce_dims, weighted=weighted
    )

    for n_batch in n_batchs:
        for _ in range(n_batch):
            batch = torch.randn((random.randint(1, 10),) + runstats.dim)

            if weighted:
                if weight_extra_dims:
                    weight = torch.randn(
                        (len(batch),) + tuple(runstats.dim[i] for i in reduce_dims)
                    )
                else:
                    weight = torch.randn((len(batch),))
            else:
                weight = None

            if do_accumulate_by and random.choice((True, False)):
                accumulate_by = torch.randint(
                    0, random.randint(1, 5), size=(batch.shape[0],)
                )
                truth = truth_obj.accumulate_batch(
                    batch, accumulate_by=accumulate_by, weight=weight
                )
                res = runstats.accumulate_batch(
                    batch, accumulate_by=accumulate_by, weight=weight
                )
            else:
                truth = truth_obj.accumulate_batch(batch, weight=weight)
                res = runstats.accumulate_batch(batch, weight=weight)
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
    with pytest.raises(ValueError):
        runstats.accumulate_batch(torch.zeros(10, 2))


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
def test_one_acc(dim, reduce_dims, reduction, do_accumulate_by, allclose):
    runstats = RunningStats(dim=dim, reduction=reduction, reduce_dims=reduce_dims)
    reduce_in_dims = tuple(i + 1 for i in reduce_dims)
    batch = torch.randn((random.randint(3, 10),) + runstats.dim)
    if do_accumulate_by:
        accumulate_by = torch.randint(0, random.randint(1, 5), size=(batch.shape[0],))
        res = runstats.accumulate_batch(batch, accumulate_by=accumulate_by)

        if reduction == Reduction.RMS:
            batch = batch.square()

        outs = []
        for i in range(max(accumulate_by) + 1):
            tmp = batch[accumulate_by == i].mean(dim=(0,) + reduce_in_dims)
            torch.nan_to_num_(tmp, nan=0.0)
            outs.append(tmp)

        truth = torch.stack(outs, dim=0)
        assert truth.shape[1:] == tuple(
            d for i, d in enumerate(runstats.dim) if i not in reduce_dims
        )

        if reduction == Reduction.RMS:
            truth.sqrt_()
    else:
        res = runstats.accumulate_batch(batch)
        if reduction == Reduction.MEAN:
            truth = batch.mean(dim=(0,) + reduce_in_dims)
        elif reduction == Reduction.RMS:
            truth = batch.square().mean(dim=(0,) + reduce_in_dims).sqrt()

    assert allclose(truth, res)
    assert allclose(truth, runstats.current_result())
