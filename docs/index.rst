.. pytorch_runstats documentation master file, created by
   sphinx-quickstart on Fri May 28 13:18:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytorch_runstats
================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


``torch_runstats`` implements memory-efficient online reductions on tensors. 

 .. note::
   The implementations currently heavily use in-place operations for peformance and memory efficiency. This probably doesn't play nice with the autograd engine — this is currently likely the wrong library for accumulating running statistics you want to backward through. (See `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/>`_ for a possible alternative.)

Currently supported :class:`Reduction` s are:

 .. autoclass:: torch_runstats.Reduction
   :members:

Support for bincounting integers and combined one-pass mean/standard deviation are planned.

The core of the library is the :class:`RunningStats` class:

 .. autoclass:: torch_runstats.RunningStats
   :members:   
