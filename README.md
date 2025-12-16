# Meta-Feature-Matching

Comparing Traditional, Deep Learning, Continual Learning, and Meta-Learning for Feature Matching.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (~20 min)
python scripts/run_benchmark.py --quick

# Full paper experiments (~3-4 hours)
python scripts/run_benchmark.py --config configs/paper_config.yaml
```

## Project Structure

```
meta-feature-matching/
├── configs/
│   ├── paper_config.yaml      # Full experiments
│   └── quick_config.yaml      # Quick testing
├── scripts/
│   └── run_benchmark.py       # Main benchmark script
├── src/
│   ├── models/                # CNN descriptors
│   ├── datasets/              # HPatches loader
│   ├── baselines/             # Continual learning (EWC, LwF)
│   └── utils/                 # Visualization
└── results/                   # Output directory
```

## Evaluation Metric

**Patch Matching Accuracy**: Given a query patch, find its correct match among distractors from the **same image** (hard negatives).

## Methods Compared

1. **Traditional**: SIFT, ORB, BRISK, AKAZE
2. **Deep Learning**: ResNet50 with triplet loss
3. **Continual Learning**: Naive, EWC, LwF
4. **Meta-Learning**: MAML adaptation curves

## Output

Results are saved to `results/<run_name>/`:
- `results.json` - All metrics
- `patches/` - Example patches for paper figures
- `figures/` - Generated visualizations
- `checkpoints/` - Trained models

## Configuration

Edit `configs/paper_config.yaml`:

```yaml
hpatches_root: "/Data/adrian.garcia/hpatches/hpatches"
deep_epochs: 10
epochs_source: 10
epochs_target: 10
```
