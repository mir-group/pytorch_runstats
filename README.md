# pytorch_runstats
Running/online statistics for PyTorch.

`torch_runstats` implements memory-efficient online reductions on tensors.

**Note:** the implementations currently heavily use in-place operations for peformance and memory efficiency. This probably doesn't play nice with the autograd engine — this is currently likely the wrong library for accumulating running statistics you want to backward through. (See [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/) for a possible alternative.)

## License

`pytorch_runstats` is distributed under an [MIT license](LICENSE).