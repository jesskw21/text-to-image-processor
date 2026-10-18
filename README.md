# Text-to-Image Diffusion from Scratch

A compact text-conditioned diffusion model in PyTorch with three hand-written
reverse-process samplers (DDPM, DDIM, DPM-Solver++), FP16 mixed-precision
training with gradient accumulation, and benchmarks that measure the
speed/fidelity trade-off.

Prompts look like `"a small purple triangle on the left"`: compositional
captions over shape × color × position × size (180 combinations). The data is
generated procedurally, so nothing needs downloading and fidelity can be
measured objectively.

## Layout

| File | What it does |
|---|---|
| `tinydiff/data.py` | Procedural dataset (anti-aliased shapes + captions), tokenizer, image grids |
| `tinydiff/model.py` | Transformer text encoder; UNet with time + text-pooled conditioning and cross-attention to caption tokens (~10M params) |
| `tinydiff/diffusion.py` | Cosine noise schedule, forward process, ε-prediction loss with caption dropout for classifier-free guidance |
| `tinydiff/samplers.py` | DDPM ancestral, DDIM (η), DPM-Solver++(2M), and the classifier-free-guidance wrapper |
| `tinydiff/training.py` | FP16/BF16 autocast + GradScaler, gradient accumulation, grad clipping, EMA, checkpoints |
| `tinydiff/evaluator.py` | Attribute classifier → prompt accuracy and Fréchet distance (FID-style) |
| `train.py` | Train the generator |
| `train_evaluator.py` | Train the evaluator on held-out real data, plus the real-vs-real FD noise floor |
| `bench_sampling.py` | Throughput, FD and prompt accuracy for each sampler/step count |
| `bench_memory.py` | Largest micro-batch under a memory budget, FP32 vs FP16 |
| `sample.py` | Prompt → image grid |
| `tests/` | Samplers checked against an analytic Gaussian denoiser; grad-accumulation == full-batch gradient |

## How it works

**Samplers.** All three use the same network, the same timestep grid, and
exactly `steps` network evaluations (NFE); only the update rule changes.
- *DDPM* takes stochastic ancestral steps through the posterior q(x_s | x_t, x̂₀). It needs many steps, because each step's injected noise must be averaged out.
- *DDIM* is a deterministic (η=0) non-Markovian update that follows the same marginals. It's equivalent to a first-order solver of the probability-flow ODE.
- *DPM-Solver++(2M)* is an exponential integrator in log-SNR time λ = log(α/σ) using the data prediction x̂₀. It's second-order, and it reuses the previous step's x̂₀ so the extra accuracy costs no extra NFE.

**Mixed precision.** Under autocast, convolutions and matmuls run in FP16, while
reductions (norms, softmax, the loss) stay FP32. Master weights and Adam state
stay FP32. A GradScaler multiplies the loss to prevent gradient underflow,
unscales before clipping, and skips steps that overflow.

**Gradient accumulation.** Each micro-batch loss is divided by `accum` before
`backward()`, so the summed `.grad` equals the full-batch gradient.
`tests/test_training.py` checks this exactly. The point is that FP16 raises the
largest micro-batch that fits, and accumulation lets you reach any *effective*
batch beyond that.

**Fidelity measurement.** A CNN trained only on real images (separate seed)
scores generated images:
- *prompt accuracy*: the fraction where shape, color, position and size all match the prompt;
- *Fréchet distance*: FID computed in that CNN's feature space against held-out real images.

`train_evaluator.py` also reports the real-vs-real FD at the same sample size.
That's the noise floor: a sampler whose FD is within it shows no measurable
fidelity loss.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

## Reproducing the benchmarks

Use a CUDA GPU (a free Google Colab T4 works). It's much faster than a laptop,
and `bench_memory.py` only measures exact peak memory on CUDA. On an 8 GB M3
Mac, training runs at about 65 img/s, so a full run takes roughly 10
hours.

In Colab (Runtime → Change runtime type → T4 GPU):

```
!git clone https://github.com/jesskw21/text-to-image-processor.git
%cd text-to-image-processor
!pip install -q -r requirements.txt
!python train.py --steps 20000 --micro-batch 128 --precision fp16
!python train_evaluator.py
!python bench_sampling.py --n 2000
!python bench_memory.py --budget-gb 4
```

Each benchmark writes `results.md` and `results.json` under `runs/`. Look at
the `preview_*.png` grids from training, and the per-sampler grids in
`runs/bench_sampling/`, to check sample quality by eye too.

Try images yourself:

```bash
.venv/bin/python sample.py "a large yellow triangle at the top" --sampler dpmsolver --steps 20
```

## Results

### Verified (deterministic, reproducible with the command shown)

**Sampler accuracy on an analytic Gaussian** (`PYTHONPATH=. python tests/test_samplers.py`).
The data distribution is N(0.3, 0.5²), so the ideal noise prediction is known in
closed form. Each sampler's output std is compared against the true 0.500.
This checks the solvers' math, not image quality.

| Sampler | Steps | Output std | Error vs 0.500 |
|---|---|---|---|
| DDPM | 100 | 0.486 | 2.8% |
| DDPM | 20 | 0.431 | 13.8% |
| DDIM | 20 | 0.467 | 6.6% |
| DPM-Solver++(2M) | 20 | 0.503 | 0.6% |
| DPM-Solver++(2M) | 10 | 0.480 | 4.0% |

**Gradient accumulation** (`pytest tests/test_training.py`): gradients from 2×
and 4× accumulated micro-batches match the full-batch gradient to within
float32 tolerance.

**Evaluator** (`python train_evaluator.py`, default settings): the attribute
classifier labels 5,000 of 5,000 held-out real images correctly (100%) on all
four attributes. This validates the scorer used by the benchmarks. On clean
synthetic data, perfect accuracy is expected.

### Preliminary (Apple M3, 8 GB — approximate)

**Memory** (`python bench_memory.py --budget-gb 2`): with a 2 GB budget, the
largest training batch is 32 in FP32 and 40 in FP16 (+25%). MPS has no exact
peak-memory counter, so peaks are sampled after each forward pass. FP16 was
not faster than FP32 on this GPU. Exact numbers need a CUDA run.

### Not yet run

The full 20k-step training run and `bench_sampling.py` (wall-clock throughput,
Fréchet distance and prompt accuracy per sampler) on a GPU. Results will be
added here when that run completes.
