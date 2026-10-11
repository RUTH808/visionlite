# VisionLite

Compact CNN classification with distillation and INT8 linear layers.

## Executed result

Compressed student accuracy **86.50%** on 10,000 held-out images, versus teacher **89.83%**. Checkpoint **61,399 bytes**, **96.37% smaller** than teacher.

## Requirements and installation

- Python **3.12**, Git, and a terminal (Windows PowerShell, macOS, or Linux).
- Internet access for the initial package/data download. No paid API key is required.
- Create and activate a virtual environment in this project folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Problem and implementation

Measure whether a small clothing-image classifier can reduce CPU inference cost and checkpoint size while retaining useful accuracy. Trains a teacher CNN and two identically initialized compact CNNs: supervised and distilled. Distillation combines cross-entropy and temperature-scaled KL divergence (temperature 3, equal weights). A fourth variant dynamically quantizes only the distilled student's Linear layers to INT8; convolution layers remain FP32.

The web/API demo serves that compressed student. This is a reproducible Fashion-MNIST benchmark, not a claim of production catalog-image accuracy.

## Run

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py train --epochs 5
python app.py predict --input results/example_request.json
python app.py serve --port 8013
```

Open http://localhost:8013. Training downloads about 30 MB of Fashion-MNIST files from the official repository; checksums are verified. Use roughly 4 GB RAM and allow several minutes on CPU. GET /health and POST /predict are available. Send a JSON `pixels` array containing 28 rows of 28 grayscale values in [0,255]. The supplied example is a held-out test image. This model does not accept arbitrary photos without suitable preprocessing.

For a quick functional smoke run use `python app.py train --epochs 1 --train-limit 2000 --output results/smoke`. This intentionally gives different quality results and replaces artifacts; rerun full training before reproducing the committed benchmark.

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

## Tests and reproducibility

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`results/metrics.json` contains the executed protocol, environment and measured values; `results/dashboard.html` is the self-contained searchable report. Source code, tests, input schema, sample request, and GitHub Actions are included. CI runs utility tests; it does not train/download the entire benchmark on every push. Hosted CI starts after publication. Python 3.12 and pinned top-level packages were used for the recorded run. CPU thread limits affect timing; exact speed is not portable.

Dockerfile is supplied for packaging, but a Docker build was not executed in this environment. Local Python and the HTTP/browser demo are the validated paths. Bindings are localhost-only; the server has no authentication, TLS, rate limiting or production deployment claim. Load only locally generated trusted joblib/checkpoint artifacts. Training/downloading must finish before serving.
