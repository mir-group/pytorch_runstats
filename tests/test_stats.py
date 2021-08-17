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


class StatsTruth(RunningStats):
    """Inefficient ground truth for RunningStats by directly storing all data"""

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
            self._n_bins = 1

        if self._acc.max() + 1 > self._n_bins:
            self._n_bins = int(self._acc.max() + 1)

        average, _, _ = self.batch_result(batch, accumulate_by)
        return average

    def reset(self, reset_n_bins: bool = False) -> None:
        if hasattr(self, "_state"):
            delattr(self, "_state")
            delattr(self, "_acc")
        if reset_n_bins:
            self._n_bins = 1

    def current_result(self):
        if not hasattr(self, "_state"):
            return torch.zeros(self._dim)
        average, _, _ = self.batch_result(self._state, self._acc)

        if len(average) < self._n_bins:
            N_to_add = self._n_bins - len(average)
            average = torch.cat((average, torch.zeros((N_to_add,)+average.shape[1:])))

        return average


@pytest.mark.parametrize(
    "dim,reduce_dims",
    [
        (1, tuple()),
        (1, (0,)),
        (3, tuple()),
        (3, (0,)),
        ((2, 3), tuple()),
        (torch.Size((1, 2, 1)), tuple()),
        (torch.Size((1, 2, 1)), (1,)),
        (torch.Size((3, 2, 4)), (0, 2)),
        (torch.Size((3, 2, 4)), (0, 1, 2)),
    ],
)
@pytest.mark.parametrize("reduction", [Reduction.MEAN, Reduction.RMS])
@pytest.mark.parametrize("do_accumulate_by", [True, False])
@pytest.mark.parametrize("nan_attrs", [True, False])
def test_runstats(dim, reduce_dims, nan_attrs, reduction, do_accumulate_by, allclose):

    n_batchs = (random.randint(1, 4), random.randint(1, 4))
    truth_obj = StatsTruth(dim=dim, reduction=reduction, reduce_dims=reduce_dims, has_nan=nan_attrs)
    runstats = RunningStats(dim=dim, reduction=reduction, reduce_dims=reduce_dims, has_nan=nan_attrs)

    for n_batch in n_batchs:
        for _ in range(n_batch):
            batch = torch.randn((random.randint(1, 10),) + runstats.dim)
            if nan_attrs and random.choice([True, False]):
                batch.view(-1)[0] = float("NaN")
            if do_accumulate_by and random.choice((True, False)):
                accumulate_by = torch.randint(
                    0, random.randint(1, 5), size=(batch.shape[0],)
                )
            else:
                accumulate_by = None

            truth = truth_obj.accumulate_batch(batch, accumulate_by=accumulate_by)
            res = runstats.accumulate_batch(batch, accumulate_by=accumulate_by)
            assert allclose(truth, res)
        truth = truth_obj.current_result()
        res = runstats.current_result()
        assert allclose(truth, res)
        truth_obj.reset(reset_n_bins=True)
        runstats.reset(reset_n_bins=True)
        
@pytest.mark.parametrize("do_accumulate_by", [True, False])
@pytest.mark.parametrize("nan_attrs", [True, False])
def test_batching(do_accumulate_by, nan_attrs, allclose):

    n_samples = 100
    dim = (3,)
    reduction = Reduction.MEAN
    reduce_dims = (0,)

    # generate reference data
    data = torch.randn((n_samples,) + dim)
    accumulate_by = torch.randint(0, 5, size=(data.shape[0],)) if do_accumulate_by else None
    if nan_attrs:
        ids = torch.randperm(n_samples)[:10]
        for idx in ids:
            data.view(-1)[idx] = float("NaN")

    # compute ground truth
    truth_obj = StatsTruth(dim=dim, reduction=reduction, reduce_dims=reduce_dims, has_nan=nan_attrs)
    truth_obj.accumulate_batch(data, accumulate_by=accumulate_by)
    truth = truth_obj.current_result()
    del truth_obj

    runstats = RunningStats(dim=dim, reduction=reduction, reduce_dims=reduce_dims, has_nan=nan_attrs)

    for stride in [1, 3, 5, 7, 13, 100]:
        n_batch = n_samples // stride
        if n_batch*stride < n_samples:
            n_batch += 1
        count = 0
        for idx in range(n_batch):

            loid = count
            hiid = count+stride
            hiid = n_samples if hiid > n_samples else hiid
            count += stride

            batch = data[loid:hiid]
            acc = None if accumulate_by is None else accumulate_by[loid:hiid]
            runstats.accumulate_batch(batch, accumulate_by=acc)

        res = runstats.current_result()
        assert allclose(truth, res)
        runstats.reset(reset_n_bins=True)


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
