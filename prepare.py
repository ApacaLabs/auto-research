"""
Fixed evaluation harness for the Neural Image Codec Challenge.
Downloads TinyImageNet, provides a dataloader, and evaluates codec quality.

Usage:
    python prepare.py          # download data and verify
    python prepare.py --eval   # evaluate a trained model (imports train.py)

DO NOT MODIFY THIS FILE — this is the fixed scoring harness.
"""

import os
import sys
import time
import math
import zipfile
import urllib.request
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMG_SIZE = 64
IMG_CHANNELS = 3
SCORE_WEIGHT = 5.0              # w in score = PSNR + w * log2(1/rate)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "neural-codec")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_FILENAME = "tiny-imagenet-200.zip"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_with_progress(url, dest_path):
    """Download a file with progress bar."""
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "neural-codec-challenge/1.0"})
    response = urllib.request.urlopen(req, timeout=120)
    total = int(response.headers.get("Content-Length", 0))
    chunk_size = 1024 * 1024  # 1 MB
    downloaded = 0
    temp_path = dest_path + ".tmp"
    with open(temp_path, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = 100 * downloaded / total
                print(f"\r  {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    os.rename(temp_path, dest_path)


def download_data():
    """Download and extract TinyImageNet-200."""
    os.makedirs(DATA_DIR, exist_ok=True)
    extract_dir = os.path.join(DATA_DIR, "tiny-imagenet-200")

    if os.path.isdir(extract_dir):
        print(f"Data: already extracted at {extract_dir}")
        return

    zip_path = os.path.join(DATA_DIR, ZIP_FILENAME)
    if not os.path.exists(zip_path):
        _download_with_progress(TINYIMAGENET_URL, zip_path)
    else:
        print(f"Data: zip already downloaded at {zip_path}")

    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Data: extracted to {extract_dir}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _collect_image_paths(split):
    """Return list of image file paths for a split."""
    root = os.path.join(DATA_DIR, "tiny-imagenet-200")
    paths = []
    if split == "train":
        train_dir = os.path.join(root, "train")
        for class_dir in sorted(os.listdir(train_dir)):
            images_dir = os.path.join(train_dir, class_dir, "images")
            if not os.path.isdir(images_dir):
                continue
            for fname in sorted(os.listdir(images_dir)):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    paths.append(os.path.join(images_dir, fname))
    elif split == "val":
        val_images_dir = os.path.join(root, "val", "images")
        for fname in sorted(os.listdir(val_images_dir)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(val_images_dir, fname))
    else:
        raise ValueError(f"Unknown split: {split}")
    return paths


class TinyImageNetDataset(Dataset):
    def __init__(self, split, transform=None):
        self.paths = _collect_image_paths(split)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def make_dataloader(batch_size, split):
    """
    Returns an infinite iterator of (B, 3, 64, 64) float tensors in [0, 1].
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = TinyImageNetDataset(split, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    def infinite():
        while True:
            for batch in loader:
                yield batch

    return infinite()

# ---------------------------------------------------------------------------
# Rate computation
# ---------------------------------------------------------------------------

def compute_rate_bpppc(all_latents_flat):
    """
    Compute empirical Shannon entropy rate in bits per pixel per channel.

    Args:
        all_latents_flat: 1D tensor of all integer-valued latent elements across
                         the entire val set, shape (total_latent_elements,)
        num_images: number of images encoded
    Returns:
        rate in bits per pixel per channel (bpppc)
    """
    # Count occurrences of each symbol
    values = all_latents_flat.cpu().numpy()
    counts = Counter(values)
    total = len(values)

    # Shannon entropy in bits
    entropy_bits = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy_bits -= p * math.log2(p)

    # Rate = entropy * (num latent elements per image) / (H * W * C)
    # Since all_latents_flat contains all elements, and entropy is per-element:
    # total bits per image = entropy_bits * (total_elements / num_images)
    # bpppc = total_bits_per_image / (H * W * C)
    # Simplifying: bpppc = entropy_bits * elements_per_image / (H * W * C)
    # But we need to know elements_per_image from outside. Instead, we compute:
    # bpppc = entropy_bits * total_elements / (num_images * H * W * C)
    return entropy_bits  # caller multiplies by elements_per_image / pixels_per_image

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, device="cuda", batch_size=256):
    """
    Evaluate a codec model on the TinyImageNet validation set.

    The model must implement:
        model.encode(images: Tensor[B,3,64,64]) -> Tensor[B, ...]  (integer-valued latents)
        model.decode(latents: Tensor[B, ...]) -> Tensor[B,3,64,64] (reconstructions in [0,1])

    Returns dict with: score, psnr_db, rate_bpppc, encode_time_s, decode_time_s
    """
    model.eval()
    val_loader = make_dataloader(batch_size, "val")
    val_paths = _collect_image_paths("val")
    num_val = len(val_paths)
    num_batches = math.ceil(num_val / batch_size)
    pixels_per_image = IMG_SIZE * IMG_SIZE * IMG_CHANNELS

    all_latents = []
    total_mse = 0.0
    total_pixels = 0
    encode_time = 0.0
    decode_time = 0.0
    images_processed = 0

    for i in range(num_batches):
        images = next(val_loader).to(device)
        actual_batch = images.shape[0]

        # Encode
        torch.cuda.synchronize()
        t0 = time.time()
        latents = model.encode(images)
        torch.cuda.synchronize()
        t1 = time.time()
        encode_time += t1 - t0

        # Cast to integer (enforced)
        latents = latents.long()

        # Decode
        torch.cuda.synchronize()
        t2 = time.time()
        recon = model.decode(latents.float())
        torch.cuda.synchronize()
        t3 = time.time()
        decode_time += t3 - t2

        # Clamp reconstruction to [0, 1]
        recon = recon.clamp(0, 1)

        # MSE
        mse = (images - recon).square().sum().item()
        total_mse += mse
        total_pixels += actual_batch * pixels_per_image

        # Collect latents for entropy computation
        all_latents.append(latents.reshape(actual_batch, -1))
        images_processed += actual_batch

    # PSNR
    mean_mse = total_mse / total_pixels
    if mean_mse == 0:
        psnr_db = 100.0  # perfect reconstruction
    else:
        psnr_db = 10 * math.log10(1.0 / mean_mse)

    # Rate (bpppc)
    all_latents_cat = torch.cat(all_latents, dim=0)  # (num_images, latent_elements_per_image)
    elements_per_image = all_latents_cat.shape[1]
    all_latents_flat = all_latents_cat.reshape(-1)

    entropy_per_element = compute_rate_bpppc(all_latents_flat)
    rate_bpppc = entropy_per_element * elements_per_image / pixels_per_image

    # Score
    if rate_bpppc > 0:
        score = psnr_db + SCORE_WEIGHT * math.log2(1.0 / rate_bpppc)
    else:
        score = psnr_db + 100.0  # perfect compression bonus

    return {
        "score": score,
        "psnr_db": psnr_db,
        "rate_bpppc": rate_bpppc,
        "encode_time_s": encode_time,
        "decode_time_s": decode_time,
        "num_val_images": images_processed,
        "latent_elements_per_image": elements_per_image,
        "entropy_per_element_bits": entropy_per_element,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare data for the Neural Image Codec Challenge")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained model (imports train.py)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data()
    print()

    # Verify dataset
    train_paths = _collect_image_paths("train")
    val_paths = _collect_image_paths("val")
    print(f"Train images: {len(train_paths):,}")
    print(f"Val images:   {len(val_paths):,}")
    print()

    if args.eval:
        # Import model from train.py and evaluate
        from train import build_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_model(device)
        # Load checkpoint if exists
        ckpt_path = "checkpoint.pt"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print("WARNING: No checkpoint found, evaluating untrained model")
        results = evaluate(model, device=device)
        print("---")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k + ':':30s} {v:.4f}")
            else:
                print(f"{k + ':':30s} {v}")
    else:
        print("Done! Ready to train: uv run train.py")