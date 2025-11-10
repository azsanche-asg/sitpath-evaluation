# SitPath Evaluation Suite (Colab-friendly)

This repo hosts the lightweight, Colab-ready evaluation harness for SitPath, letting you reproduce the paper's matched-capacity experiments, compare tokenization ablations, and export standardized tables/figures without leaving notebooks.

## Goals
- Reproduce SitPath results across pedestrian and autonomous driving datasets.
- Stress-test symbolic situation tokens versus matched-capacity neural baselines.
- Provide deterministic preprocessing so controllability edits and seed sweeps align with the published protocol.

## Datasets
- **ETH/UCY**: classic pedestrian crowds with shared train/val/test splits.
- **Stanford Drone mini (SDD-mini)**: curated subset for faster Colab turnaround.
- **nuScenes-mini**: multi-agent vehicle trajectories (download steps documented separately).

`make data` pulls ETH/UCY + SDD-mini scripts locally and prints notes for nuScenes credentials.

## Tokenization setup
- Default symbolic parameters come from the paper: `M=16` codes, `R=5m` spatial radius, `B=4` temporal bins, `Kτ=3` interaction hops.
- Ablations fall back to `{M=8, R=10m, B=2, Kτ=1}` to stress reduced capacity while keeping inference cost comparable.
- `scripts/precompute_tokens.py` caches tokens per dataset so Colab sessions only need lightweight finetuning/eval runs.

## Matched-capacity baselines
We keep parameter counts within ±2% of SitPath-Transformer: Social-STGCNN, PECNet, and plain Transformer encoders are configured with identical depth/width so comparisons focus on symbolic conditioning rather than raw capacity.

## Metrics
Reported metrics match the SitPath paper: ADE, FDE, minADE20, MR@2m, NLL, ECE, and trajectory diversity (unique tokenized modes / 20 samples).

## Controllability edits
Token sequences can be edited before decoding (e.g., swapping interaction codes or adjusting occupancy bins). Scripts log both the original and edited rollouts for qualitative inspection.

## Seeds & reproducibility
- Experiments run seeds `{0, 1, 2}`; average + std are stored under `tables/`.
- `outputs/seed_{k}` checkpoints carry optimizer states so resumed Colab runs stay deterministic after `torch.set_float32_matmul_precision("high")` and `cudnn.deterministic=True`.
- All random draws pass through `sitpath_eval.utils.random.set_seed`.

## Quickstart commands
Run the following in order (Colab or local shell):
```bash
pip install -e .
make data
python scripts/precompute_tokens.py --dataset eth_ucy
python scripts/train.py --cfg configs/eth_ucy/sitpath_transformer.yaml
python scripts/evaluate.py --split test --save_csv tables/results_eth_ucy.csv
```

## Colab notes
1. Start a GPU runtime, `git clone` this repo, and run the commands above.
2. Use `pip install -r requirements.txt` if editable installs are blocked, then re-run `pip install -e .` once to register entry points.
3. Persist `outputs/` and `cache/` via Google Drive mounting or `gsutil` if you need longer sweeps.

## Config provenance
Every hyperparameter is copied verbatim from the SitPath paper/protocol:
- Transformer depth 8, model dim 256, FF dim 1024, 8 heads, dropout 0.1.
- Token embedding dim 128, residual dropout 0.05, positional temperature 0.7.
- Optimizer AdamW (lr 2e-4, weight decay 0.01, betas 0.9/0.95), cosine schedule with 4k warmup.
- Batch size 64, context 8 seconds, prediction 12 seconds, frame rate 2 Hz (ETH/UCY) and 5 Hz (SDD/nuScenes-mini).
- Evaluation draws 20 trajectories per agent with nucleus 0.85 sampling during controllability sweeps.

## Reproducibility checklist
- Versioned configs live under `configs/{eth_ucy,sdd_mini,nuscenes_mini}` and are never edited in notebooks; use overrides via CLI flags instead.
- `tables/` and `figs/` scripts regenerate all reported numbers; CSV/PNG artifacts include git commit metadata for traceability.
- All dataset scripts verify checksums before moving files into `cache/`.

## Citation
Please cite the SitPath paper when using this suite:
```bibtex
@inproceedings{sitpath2024,
  title     = {SitPath: Symbolic Situation Tokens for Spatio-Temporal Path Forecasting},
  author    = {Doe, Jane and Smith, John and Taylor, Alex},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```
