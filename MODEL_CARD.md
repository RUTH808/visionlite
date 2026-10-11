# VisionLite model and evaluation card

Compressed student accuracy **86.50%** on 10,000 held-out images, versus teacher **89.83%**. Checkpoint **61,399 bytes**, **96.37% smaller** than teacher.

## Evaluation protocol and limits

- Official 60,000 training images split into 55,000 train and 5,000 validation, stratified with seed 42. Official 10,000-image test split remains held out; ten classes.
- Five epochs per learned model; checkpoint selection uses validation accuracy. Teacher training seed 42; compact students share initialization seed 99 and data order.
- Report accuracy, macro-F1, a confusion matrix, per-class accuracy and a controlled Gaussian-noise stress test (sigma 0.10 on normalized pixels).
- Accuracy intervals are Wilson binomial 95% intervals. One training seed was run; a small student-vs-distilled difference does not establish a statistically reliable distillation improvement.
- CPU latency: one thread, batch 1, 20 warmups, 200 timed calls including softmax. Throughput: 20 repetitions of batch 256. Model loading/network are excluded. Shared-host timings are observational and hardware dependent.
- INT8 linear layers reduce storage but were slower than the unquantized student in the recorded batch-1 timings. The teacher was more accurate. The report preserves both trade-offs.

## Sources

- [Fashion-MNIST, Zalando Research](https://github.com/zalandoresearch/fashion-mnist) and [dataset paper](https://arxiv.org/abs/1708.07747)
- [PyTorch knowledge distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [PyTorch dynamic quantization API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)

Fashion-MNIST is MIT licensed by its authors; upstream data/dependencies retain their terms. Source files are downloaded rather than redistributed. Quantization APIs are version-pinned because their recommended interfaces may change.

## Intended use

Educational portfolio, reproducible offline evaluation, and local interactive demonstration. No autonomous high-impact decisions. Results apply to the named benchmark under the recorded split and hardware. See README for dependencies and run commands.
