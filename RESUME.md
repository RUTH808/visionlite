# VisionLite | Compact Image Classification

PyTorch, CNNs, Knowledge Distillation, INT8 Quantization

- Reduced checkpoint size by 96.4% through a compact distilled CNN and INT8 linear layers, achieving 86.50% accuracy on 10,000 test images versus 89.83% for the teacher.
- Trained models on 55,000 images with a separate 5,000-image validation set; benchmarked 4 variants for accuracy, noise robustness, model size, and CPU inference latency.

Measured offline benchmark results, not production business impact. Review README and reproduce the experiment before presenting the work in an interview.
