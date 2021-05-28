# pytorch_runstats
Running/online statistics for PyTorch.

`torch_runstats` implements memory-efficient online reductions on tensors. 

Notable features:
 - Arbitrary sample shapes beyond single scalars
 - Reduction over arbitrary dimensions of each sample
 - "Batched"/"binned" reduction into multiple running tallies using a per-sample bin index. 
  This can be useful, for example, in accumulating statistics over samples by some kind of "type" index or for accumulating statistics per-graph in a `pytorch_geometric`-like [batching scheme](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html). (This feature uses and is similar to [`torch_scatter`](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html).)

**Note:** the implementations currently heavily use in-place operations for peformance and memory efficiency. This probably doesn't play nice with the autograd engine — this is currently likely the wrong library for accumulating running statistics you want to backward through. (See [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/) for a possible alternative.)

## License

`pytorch_runstats` is distributed under an [MIT license](LICENSE).