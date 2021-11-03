.. pytorch_runstats documentation master file, created by
   sphinx-quickstart on Fri May 28 13:18:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================
pytorch_runstats
================

 .. toctree::

   self


``torch_runstats`` implements memory-efficient online reductions on tensors. Notable features:

 * Arbitrary sample shapes beyond single scalars
 * Reduction over arbitrary dimensions of each sample
 * "Batched"/"binned" reduction into multiple running tallies using a per-sample bin index. This can be useful, for example, in accumulating statistics over samples by some kind of "type" index or for accumulating statistics per-graph in a ``pytorch_geometric``-like `batching scheme <https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html>`_ . (This feature is similar to `torch_scatter <https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html>`_ .)
 * Option to ignore NaN values with correct sample counting

 .. note::
   The implementations currently heavily uses in-place operations for peformance and memory efficiency. This probably doesn't play nice with the autograd engine — this is currently likely the wrong library for accumulating running statistics you want to backward through. (See `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_ for a possible alternative.)

Examples
--------

Basic
^^^^^

 .. code-block:: python
   
   import torch
   from torch_runstats import Reduction, RunningStats

   # Interspersed ones and zeros with a ratio of 2:1 ones to zeros
   data = torch.cat([torch.ones(5), torch.zeros(3), torch.ones(5), torch.zeros(2)])
   data.unsqueeze_(-1)

   rs = RunningStats(
      dim=(1,),
      reduction=Reduction.MEAN,
   )
   # Accumulate the statistics over the data in batches
   # Note that each call to accumulate_batch also returns the statistic for the current batch:
   print(rs.accumulate_batch(data[:5]))  # => tensor([[1.]])
   rs.accumulate_batch(data[5:7])
   rs.accumulate_batch(data[7:13])
   rs.accumulate_batch(data[13:])
   print(rs.current_result())  # => tensor([[0.6667]])

   # Accumulated data can be cleared
   rs.reset()
   # An empty object returns the identity for the reduction:
   print(rs.current_result())  # => tensor([[0.]])

"Binned"
^^^^^^^^

A main feature of ``torch_runstats`` is accumulating different samples in a batch into different "bins" --- different running statistics --- based on a provided index:

.. code-block:: python
  
   import torch
   from torch_runstats import Reduction, RunningStats

   data = torch.cat([torch.ones(5), torch.zeros(3), torch.ones(5), torch.zeros(2)])
   data.unsqueeze_(-1)
   sample_type = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(7, dtype=torch.long)])

   rs = RunningStats(
      dim=(1,),
      reduction=Reduction.MEAN,
   )
   rs.accumulate_batch(data, accumulate_by=sample_type)
   # The first entry is for "bin" (sample_type) 0, the second for 1:
   print(rs.current_result())  # => tensor([[0.6250], [0.7143]])
   # These values are what we expect:
   print(5/8, 5/7)  # => 0.625 0.714


Reduce over arbitrary dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A reduction can also be taken over a sample dimension:

.. code-block:: python
 
   import torch
   from torch_runstats import Reduction, RunningStats

   data = torch.cat([torch.ones(5, 3, 2), torch.zeros(3, 3, 2)], dim=0)

   rs = RunningStats(
      dim=(3, 2),
      reduction=Reduction.MEAN,
      reduce_dims=0,  # reduce the sample dimension of size 3
   )
   rs.accumulate_batch(data)
   # Note that the reduction has a bin index (len 1),
   # and the sample dimension of shape 2,
   # but that the dimension of size 3 has been reduced out:
   print(rs.current_result())  # => tensor([[0.6250, 0.6250]])


Ignore NaNs
^^^^^^^^^^^

When the ``ignore_nan`` option is enabled, ``RunningStats`` will only count and reduce over non-NaN elements:

.. code-block:: python

   import torch
   from torch_runstats import Reduction, RunningStats

   NaN = float("nan")

   data = torch.Tensor([
      [1.0, NaN, NaN],
      [NaN, NaN, NaN],
      [1.0, NaN, 1.0],
      [1.0, 3.0, 1.0],
      [1.0, NaN, NaN]
   ])
   accumulate_by = torch.LongTensor([0, 0, 1, 1, 1])

   rs = RunningStats(
      dim=(3,),
      reduction=Reduction.MEAN,
      reduce_dims=0,  # reduce the sample dimension of size 3
      ignore_nan=True
   )
   rs.accumulate_batch(data, accumulate_by=accumulate_by)
   # In the first bin, we see that the mean was taken over only one sample-
   # the one non-NaN sample, giving a value of 1.0
   #
   # In the second bin, we see that we got the mean of the non-NaN
   # elements: (1 * 5 + 3) / 6 = 1.33333...
   print(rs.current_result())  # => tensor([1.0000, 1.3333])


Class Reference
---------------

Currently supported :class:`Reduction` s are:

 .. autoclass:: torch_runstats.Reduction
   :members:

Support for bincounting integers and combined one-pass mean/standard deviation are planned.

The core of the library is the :class:`RunningStats` class:

 .. autoclass:: torch_runstats.RunningStats
   :members:   
