# Neural Image Codec Challenge — Agent Program

This is an experiment to have an LLM autonomously research the best neural image codec.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `codec/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b codec/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — challenge context and scoring formula.
   - `prepare.py` — fixed constants, data download, dataloader, evaluation harness. Do not modify.
   - `train.py` — the file you modify. Model architecture, loss function, training loop.
4. **Verify data exists**: Check that `~/.cache/neural-codec/data/tiny-imagenet-200/` exists. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 30 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, loss function, quantization method, training loop, hyperparameters, etc.
- Create new `.py` files if needed for model components.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation harness, data loading, and scoring constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Use pretrained weights or external data.

**The goal is simple: get the highest score.** The score formula is:

```
score = PSNR + 5 * log2(1 / rate_bpppc)
```

You can improve by:
- **Increasing PSNR** — better reconstruction quality (lower MSE)
- **Decreasing rate** — more efficient compression (lower entropy of latents)
- **Both** — the best experiments improve both simultaneously

Since the time budget is fixed, you don't need to worry about training time — it's always 30 minutes. Everything is fair game: change the architecture, the loss function, the quantization method, the optimizer, the hyperparameters.

**The codec interface is fixed.** Your model must implement:
```python
model.encode(images) -> integer-valued latents
model.decode(latents) -> reconstructed images in [0,1]
```
Latents are cast to `long` during evaluation, so they must be integer-valued.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful score gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
score:            42.35
psnr_db:          30.50
rate_bpppc:       0.2500
encode_time_s:    2.3
decode_time_s:    1.8
training_seconds: 1800.0
peak_vram_mb:     12345.6
num_params_M:     5.2
```

You can extract the key metrics from the log file:

```
grep "^score:\|^psnr_db:\|^rate_bpppc:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	score	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. score achieved (e.g. 42.35) — use 0.00 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	score	memory_gb	status	description
a1b2c3d	38.50	8.2	keep	baseline
b2c3d4e	40.10	8.4	keep	add rate penalty to loss
c3d4e5f	37.20	8.2	discard	switch to VQ (rate too high)
d4e5f6g	0.00	0.0	crash	attention in bottleneck (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `codec/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^score:\|^psnr_db:\|^rate_bpppc:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If score improved (higher), you "advance" the branch, keeping the git commit
9. If score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~30 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 40 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes, read about neural compression techniques. The loop runs until the human interrupts you, period.

## Ideas to explore

Here are some directions (in rough order of expected impact):

- **Rate regularization**: The starter uses MSE-only loss. Adding a rate penalty (e.g. entropy estimate of latents) can dramatically improve score by reducing rate.
- **Better quantization**: Replace round+STE with vector quantization (VQ-VAE), noise relaxation (additive uniform noise during training), or soft-to-hard annealing.
- **Entropy model**: Add a learned prior / hyperprior to reduce the gap between marginal entropy and conditional entropy.
- **Architecture upgrades**: ResBlocks, skip connections, attention in the bottleneck, U-Net structure.
- **Perceptual tricks**: While scoring is MSE-based (PSNR), architectural changes that help the model focus on perceptually important features can still help MSE.
- **torch.compile**: The starter doesn't use it — enabling it can give a significant speedup, allowing more training steps in the same time budget.
- **Data augmentation**: Random horizontal flips, small crops/shifts.
- **Learning rate tuning**: The default 1e-3 may not be optimal for modified architectures.
- **Latent channel count**: Fewer channels = lower rate but potentially worse PSNR. Finding the sweet spot matters.
