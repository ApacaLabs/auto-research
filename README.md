# Neural Image Codec Challenge

Build the best neural image codec for TinyImageNet 64x64 in 30 minutes of training time.

## Overview

You are given a simple convolutional autoencoder that compresses 64x64 RGB images into quantized integer latents. Your goal is to maximize the **score**, which rewards both reconstruction quality (PSNR) and compression efficiency (rate).

## Scoring

```
score = PSNR + 5 * log2(1 / rate_bpppc)
```

Where:
- **PSNR** (dB) = `10 * log10(1 / MSE)` over all validation pixels
- **rate_bpppc** = bits per pixel per channel, computed from Shannon entropy of your integer-valued latents

**Worked example:** If your codec achieves PSNR = 28 dB at rate = 0.25 bpppc:
```
score = 28 + 5 * log2(1 / 0.25) = 28 + 5 * 2 = 38
```

Higher score is better. You can improve by increasing PSNR, decreasing rate, or both.

## Quick start

```bash
# Install dependencies
uv sync

# Download TinyImageNet (~237 MB, one-time)
uv run prepare.py

# Train the starter model (30 min)
uv run train.py
```

## Rules

1. **Modify or create any files except `prepare.py`** — the evaluation harness is fixed
2. **No pretrained weights** — train from scratch within the time budget
3. **No external data** — only TinyImageNet-200 (downloaded by `prepare.py`)
4. **No new dependencies** — use only what's in `pyproject.toml` (PyTorch, torchvision, numpy, pillow, matplotlib)
5. **30-minute training budget** — wall-clock time, excluding startup/compilation warmup
6. **Codec interface** — your model must implement `encode(images) -> integer latents` and `decode(latents) -> reconstructions`

## File structure

```
prepare.py      — fixed evaluation harness + data download (DO NOT MODIFY)
train.py        — starter model + training loop (modify this)
pyproject.toml  — dependencies
README.md       — this file
```

## Codec interface

Your model must implement these two methods:

```python
model.encode(images: Tensor[B,3,64,64]) -> Tensor[B, ...]   # integer-valued quantized latents
model.decode(latents: Tensor[B, ...]) -> Tensor[B,3,64,64]   # reconstructed images in [0,1]
```

Latents are cast to `long` during evaluation — they must be integer-valued.

## Submission

Submit your final repository state. Your score will be verified by running:

```bash
uv sync && uv run prepare.py && uv run train.py
```

The `---` block printed at the end of training is your official result.

## License

MIT
