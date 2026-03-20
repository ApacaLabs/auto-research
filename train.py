"""
Neural Image Codec Challenge — Starter Model
A simple convolutional autoencoder with round+STE quantization.

Usage: uv run train.py

This is the file you modify. Everything is fair game: architecture,
hyperparameters, loss function, quantization method, etc.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import IMG_SIZE, IMG_CHANNELS, TIME_BUDGET, make_dataloader, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters (edit these freely)
# ---------------------------------------------------------------------------

NUM_LATENT_CHANNELS = 16    # latent channels (more = higher rate, better quality)
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, latent_ch=NUM_LATENT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),       # -> (64, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),     # -> (128, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),    # -> (256, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_ch, kernel_size=3, stride=1, padding=1),  # -> (latent_ch, 8, 8)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_ch=NUM_LATENT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 16, 16)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # -> (128, 32, 32)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # -> (64, 64, 64)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),           # -> (3, 64, 64)
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class CodecModel(nn.Module):
    """
    Simple convolutional autoencoder with round+STE quantization.

    Interface contract (do not change method signatures):
        encode(images) -> integer-valued latent tensor
        decode(latents) -> reconstructed images in [0, 1]
    """
    def __init__(self, latent_ch=NUM_LATENT_CHANNELS):
        super().__init__()
        self.encoder = Encoder(latent_ch)
        self.decoder = Decoder(latent_ch)

    def encode(self, x):
        """Encode images to integer-valued latents."""
        z = self.encoder(x)
        z_q = torch.round(z)
        return z_q

    def decode(self, z_q):
        """Decode integer latents to reconstructed images."""
        return self.decoder(z_q)

    def forward(self, x):
        """Forward pass with straight-through estimator for training."""
        z = self.encoder(x)
        # Round + straight-through estimator
        z_q = z + (torch.round(z) - z).detach()
        x_recon = self.decoder(z_q)
        return x_recon


def build_model(device="cuda"):
    """Build and return the codec model (used by prepare.py --eval)."""
    model = CodecModel(NUM_LATENT_CHANNELS)
    model = model.to(device)
    return model

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    # Model
    model = build_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine LR schedule (computed based on estimated total steps)
    estimated_steps = int(TIME_BUDGET / 0.1)  # rough estimate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=estimated_steps, eta_min=LEARNING_RATE * 0.01
    )

    # Dataloader
    train_loader = make_dataloader(BATCH_SIZE, "train")

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Latent channels: {NUM_LATENT_CHANNELS}")
    print(f"Latent spatial: 8x8 -> {NUM_LATENT_CHANNELS * 8 * 8} integers/image")
    print()

    # Training loop (time-budgeted)
    model.train()
    t_start_training = time.time()
    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0
    warmup_steps = 10  # exclude first steps (compilation etc.)

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        images = next(train_loader).to(device, non_blocking=True)
        recon = model(images)
        loss = F.mse_loss(recon, images)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > warmup_steps:
            total_training_time += dt

        # Logging
        loss_f = loss.item()
        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_f
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

        psnr_approx = -10 * math.log10(max(debiased_loss, 1e-10))
        progress = min(total_training_time / TIME_BUDGET, 1.0) if TIME_BUDGET > 0 else 0
        remaining = max(0, TIME_BUDGET - total_training_time)
        lr_now = optimizer.param_groups[0]["lr"]

        if step % 50 == 0:
            print(f"\rstep {step:05d} ({100*progress:.1f}%) | loss: {debiased_loss:.6f} | ~PSNR: {psnr_approx:.1f}dB | lr: {lr_now:.2e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

        # GC management
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

        step += 1

        # Time's up
        if step > warmup_steps and total_training_time >= TIME_BUDGET:
            break

    print()
    print(f"Training done: {step} steps in {total_training_time:.1f}s")
    print()

    # Save checkpoint
    torch.save(model.state_dict(), "checkpoint.pt")
    print("Saved checkpoint.pt")

    # Final evaluation
    print("Evaluating on validation set...")
    results = evaluate(model, device=str(device), batch_size=BATCH_SIZE)

    # Summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print()
    print("---")
    print(f"score:            {results['score']:.2f}")
    print(f"psnr_db:          {results['psnr_db']:.2f}")
    print(f"rate_bpppc:       {results['rate_bpppc']:.4f}")
    print(f"encode_time_s:    {results['encode_time_s']:.1f}")
    print(f"decode_time_s:    {results['decode_time_s']:.1f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
