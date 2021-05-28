"""Batched running statistics for PyTorch."""

from typing import Union, Tuple, Optional, Sequence
import enum
import numbers

import torch
from torch_scatter import scatter


def _prod(x):
    """Compute the product of an iterable."""
    out = 1
    for a in x:
        out *= a
    return out


class Reduction(enum.Enum):
    r"""Enum indicating a reduction over :math:`N` values :math:`x_i`.

    Currently supported reductions:

     * ``Reduction.MEAN``: :math:`\frac{1}{N}\sum_i^{N}{x_i}`
     * ``Reduction.RMS``: :math:`\sqrt{\frac{1}{N}\sum_i^{N}{x_i^2}}`
    """

    MEAN = "mean"
    MEAN_STD = "mean_std"
    RMS = "rms"
    COUNT = "count"


# TODO: impliment counting
# TODO: impliment stds
class RunningStats:
    r"""Compute running statistics over batches of samples.

    Args:
        dim: the shape of a single sample. If an integer, interpreted as ``(dim,)``.
        reduction: the statistic to compute
        reduce_dims: extra dimensions within each sample to reduce over.
            If an integer, interpreted as ``(reduce_dims,)``.

            This is a tuple of dimension indexes that are interpreted as dimension indexes within each *sample*: ``reduce_dims=(1,)`` implies that in a batch of size ``(N, A, B, C)`` with ``dim = (A, B, C)`` the ``N`` and ``B`` dimensions will be reduced over. (To reduce over ``A`` instead, you would use ``reduce_dims=(0,)`` to reduce over the first non-batch dimension.)

            By default an empty tuple, i.e., reduce only over the batch dimension.
    """

    _in_dim: Tuple[int, ...]
    _dim: Tuple[int, ...]
    _reduce_dims: Tuple[int, ...]
    _reduction_factor: int
    _reduction: Reduction

    def __init__(
        self,
        dim: Union[int, Tuple[int, ...]] = 1,
        reduction: Reduction = Reduction.MEAN,
        reduce_dims: Union[int, Sequence[int]] = tuple(),
        weighted: bool = False,
    ):
        if isinstance(dim, numbers.Integral):
            self._dim = (dim,)
        elif isinstance(dim, tuple):
            self._dim = dim
        else:
            raise TypeError(f"Invalid dim {dim}")
        self._in_dim = self._dim

        if isinstance(reduce_dims, numbers.Integral):
            self._reduce_dims = (reduce_dims,)
        else:
            self._reduce_dims = tuple(set(reduce_dims))
        if len(self._reduce_dims) > 0:
            if min(self._reduce_dims) < 0 or max(self._reduce_dims) >= len(self._dim):
                raise ValueError(
                    f"Invalid dimension indexes in reduce_dims {self._reduce_dims}"
                )
            # we do reduce other dims, make a new _dim
            self._dim = tuple(
                d for i, d in enumerate(self._in_dim) if i not in self._reduce_dims
            )
            self._reduction_factor = _prod(
                d for i, d in enumerate(self._in_dim) if i in self._reduce_dims
            )
        else:
            self._reduction_factor = 1

        self._out_dim = tuple(
            self._dim[i] for i in range(len(self._dim)) if i not in self._reduce_dims
        )

        if reduction not in (Reduction.MEAN, Reduction.RMS):
            raise NotImplementedError(f"Reduction {reduction} not yet implimented")
        self._reduction = reduction

        self.weighted = weighted
        if weighted:
            raise NotImplementedError

        self.reset()

    def accumulate_batch(
        self, batch: torch.Tensor, accumulate_by: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Accumulate a batch of samples into the running statistics.

        Args:
            batch: tensor of shape ``(N_samples,) + self.dim``. The batch of samples to process.
            accumulate_by: tensor of indexes of shape ``(N_samples,)``.
                If provided, the nth sample will be accumulated into the ``accumulate_by[n]``th bin. If ``None`` (the default), all samples will be accumulated into the first (0th) bin.

        Returns:
            tensor of shape ``(N_bins,) + self.output_dim`` giving the aggregated statistics *for this input batch*. Accumulated statistics up to this point can be retreived with ``current_result()``.

            ``N_bins`` is ``accumulate_by.max() + 1`` --- the number of bins in the batch --- and not the overall number of bins ``self.n_bins``.
        """
        if batch.shape == self._in_dim:
            batch = batch.unsqueeze(0)
        elif batch.shape[1:] == self._in_dim:
            pass
        else:
            raise ValueError(
                f"Data shape of batch, {batch.shape}, does not match the input data dimension of this RunningStats, {self._in_dim}"
            )

        if self._reduction == Reduction.COUNT:
            raise NotImplementedError
        else:
            if self._reduction == Reduction.RMS:
                batch = batch.square()

            if accumulate_by is None:
                # accumulate everything into the first bin
                # the number of samples we have is the size of the
                # extra dims times how many samples we have
                N = batch.shape[0] * self._reduction_factor
                new_sum = batch.sum(dim=(0,) + tuple(i + 1 for i in self._reduce_dims))

                # accumulate
                self._state[0] += (new_sum - N * self._state[0]) / (self._n[0, 0] + N)
                self._n[0, 0] += N

                # for the batch
                new_sum /= N

                if self._reduction == Reduction.RMS:
                    new_sum.sqrt_()

                return new_sum.unsqueeze(0)
            else:
                # How many samples did we see in each bin?
                N = torch.bincount(accumulate_by).reshape([-1, 1])

                # Reduce along non-batch dimensions
                if len(self._reduce_dims) > 0:
                    batch = batch.sum(dim=tuple(i + 1 for i in self._reduce_dims))
                    # Each sample is now a reduction over _reduction_factor samples
                    N *= self._reduction_factor

                # reduce along the first (batch) dimension using accumulate_by
                new_sum = scatter(batch, accumulate_by, dim=0)

                # do we need new bins?
                if new_sum.shape[0] > self._n_bins:
                    # time to expand
                    N_to_add = new_sum.shape[0] - self._n_bins
                    self._state = torch.cat(
                        (self._state, self._state.new_zeros((N_to_add,) + self._dim)),
                        dim=0,
                    )
                    self._n = torch.cat(
                        (self._n, self._n.new_zeros((N_to_add, 1))), dim=0
                    )
                    assert self._state.shape == (self._n_bins + N_to_add,) + self._dim
                    self._n_bins += N_to_add

                N_bins_new = new_sum.shape[0]
                bshape = (N_bins_new,) + (1,) * len(self._dim)

                self._state[:N_bins_new] += (
                    new_sum - N.reshape(bshape) * self._state[:N_bins_new]
                ) / (self._n[:N_bins_new] + N).reshape(bshape)
                self._n[:N_bins_new] += N
                # Make div by zero 0
                torch.nan_to_num_(self._state, nan=0.0)

                new_sum /= N.reshape(bshape)
                # Make div by zero 0
                torch.nan_to_num_(new_sum, nan=0.0)

                if self._reduction == Reduction.RMS:
                    new_sum.sqrt_()

                return new_sum

    def reset(self, reset_n_bins: bool = False) -> None:
        """Forget all previously accumulated state.

        This method does *not* forget ``self.n_bins`` unless ``reset_n_bins`` is True.

        Args:
            reset_n_bins: whether to reset this object to one accumulation bin. This defaults to False on the assumption that a reset object will likely be used to process data with a similar or equal number of bins.
        """
        if not reset_n_bins and hasattr(self, "_state"):
            self._state.fill_(0.0)
            self._n.fill_(0)
        else:
            self._n_bins = 1
            self._n = torch.zeros((self._n_bins, 1), dtype=torch.long)
            self._state = torch.zeros((self._n_bins,) + self._dim)

    def to(self, device=None, dtype=None) -> None:
        """Move this ``RunningStats`` to a new dtype and/or device.

        Args:
            dtype: like ``torch.Tensor.to``
            device: like ``torch.Tensor.to``
        """
        self._state = self._state.to(dtype=dtype, device=device)
        self._n = self._n.to(device=device)

    def current_result(self) -> torch.Tensor:
        """Get the current value of the running statistic.

        Returns:
            A tensor of shape ``(self.n_bins,) + self.output_dim``. The nth bin contains the accumulated statistics for all processed samples whose ``accumulate_by`` was n.
        """
        assert self._state.shape == (self._n_bins,) + self._dim
        if self._reduction == Reduction.MEAN:
            return self._state.clone()
        elif self._reduction == Reduction.RMS:
            return torch.sqrt(self._state)

    @property
    def n(self) -> torch.Tensor:
        """The number of samples processed so far in each bin.

        Returns:
            A ``LongTensor`` of shape ``(self.n_bins,)``
        """
        return self._n.squeeze(1)

    @property
    def n_bins(self) -> int:
        """The number of ``accumulate_by`` bins currently maintained by this object."""
        return self._n_bins

    @property
    def dim(self) -> Tuple[int, ...]:
        """The shape of a single input sample for this ``RunningStats``"""
        return self._in_dim

    @property
    def output_dim(self) -> Tuple[int, ...]:
        """The shape of the output statistic for a single bin."""
        return self._out_dim

    @property
    def reduce_dims(self) -> Tuple[int, ...]:
        """Indexes of dimensions in each sample that will be reduced."""
        return self._reduce_dims

    @property
    def reduction(self) -> Reduction:
        """The reduction computed by this object."""
        return self._reduction
