# Meta-Learning for Feature Matching

A comprehensive benchmark comparing traditional and deep learning approaches for local feature matching, with a focus on continual learning and domain adaptation.

## Overview

This project investigates:

1. **Traditional vs. Deep Learning Descriptors**: Comparing classical methods (SIFT, ORB, BRISK, AKAZE) with learned CNN descriptors on the HPatches benchmark.

2. **Domain Adaptation**: Evaluating how well models transfer between illumination and viewpoint variations.

3. **Continual Learning**: Comparing methods to prevent catastrophic forgetting when adapting to new domains:
   - Naive fine-tuning (baseline)
   - EWC (Elastic Weight Consolidation)
   - LwF (Learning without Forgetting)
   - SI (Synaptic Intelligence)
   - MAML (Model-Agnostic Meta-Learning)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/meta-feature-matching.git
cd meta-feature-matching

# Create conda environment
conda env create -f environment.yml
conda activate meta-matching

# Or install with pip
pip install -r requirements.txt

# Set HPatches dataset path
export HPATCHES_ROOT="/path/to/hpatches"
```

### Download HPatches Dataset

```bash
cd /path/to/data
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xvf hpatches-sequences-release.tar.gz
export HPATCHES_ROOT="/path/to/data/hpatches-sequences-release"
```

## Project Structure

```
meta-feature-matching/
├── configs/                    # Configuration files
│   └── benchmark.yaml          # Main benchmark config
├── scripts/                    # Executable scripts
│   └── run_benchmark.py        # Full benchmark suite
├── src/
│   ├── baselines/              # Baseline methods
│   │   ├── traditional_matchers.py  # SIFT, ORB, BRISK, AKAZE
│   │   ├── deep_matcher.py          # Deep learning matcher
│   │   └── continual_learning.py    # EWC, LwF, SI
│   ├── datasets/
│   │   └── hpatches_loader.py  # HPatches data loading
│   ├── evaluation/
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── visualization.py    # Plotting utilities
│   ├── meta/
│   │   └── maml.py             # MAML implementation
│   ├── models/
│   │   └── descriptor_wrapper.py  # CNN descriptor models
│   └── utils/
│       └── logger.py           # Logging utilities
├── results/                    # Output directory
├── environment.yml             # Conda environment
└── README.md
```

## Quick Start

### Run Full Benchmark

```bash
python scripts/run_benchmark.py --config configs/benchmark.yaml
```

### Quick Test (fewer epochs)

```bash
python scripts/run_benchmark.py --quick
```

### Individual Evaluations

```python
from src.baselines.traditional_matchers import get_all_matchers
from src.datasets.hpatches_loader import get_sequence_dataloader
from src.evaluation.metrics import evaluate_traditional_matcher

# Evaluate SIFT
matchers = get_all_matchers()
loader = get_sequence_dataloader("viewpoint")
results = evaluate_traditional_matcher(matchers["SIFT_BF"], loader)
print(f"MMA@3px: {results['mma'].mma_3px:.4f}")
```

## Methods

### Traditional Matchers

| Method | Detector | Descriptor | Matcher |
|--------|----------|------------|---------|
| SIFT_BF | SIFT | SIFT (128-D) | Brute Force |
| SIFT_FLANN | SIFT | SIFT (128-D) | FLANN |
| ORB_BF | ORB | ORB (256-bit) | Brute Force |
| BRISK_BF | BRISK | BRISK (512-bit) | Brute Force |
| AKAZE_BF | AKAZE | AKAZE (486-bit) | Brute Force |

### Deep Learning Models

| Model | Backbone | Params | Output Dim |
|-------|----------|--------|------------|
| ResNet50 | ResNet-50 | ~25M | 512 |
| ResNet18 | ResNet-18 | ~11M | 512 |
| Lightweight | Custom CNN | ~0.5M | 128 |
| HardNet | HardNet-style | ~1M | 128 |

### Continual Learning Methods

| Method | Key Idea | Hyperparameters |
|--------|----------|-----------------|
| Naive | No protection | - |
| EWC | Fisher information penalty | λ=1000 |
| LwF | Knowledge distillation | λ=1.0, T=2.0 |
| SI | Online importance | λ=1.0 |
| MAML | Meta-learning for adaptation | lr_in=1e-3, lr_out=1e-4 |

## Metrics

### Matching Quality
- **MMA@t**: Mean Matching Accuracy at t-pixel threshold
- **Inlier Ratio**: Fraction of matches that are geometric inliers
- **Homography Accuracy**: Corner error < threshold

### Continual Learning
- **Forgetting**: Accuracy drop on source domain after training on target
- **Forward Transfer**: Improvement on target domain
- **Adaptation Curve**: Accuracy vs. adaptation steps (for MAML)

## Expected Results

### Traditional Matchers (HPatches)

| Method | MMA@3px (Illum) | MMA@3px (View) | Time (ms) |
|--------|-----------------|----------------|-----------|
| SIFT_BF | ~0.45 | ~0.35 | ~50 |
| ORB_BF | ~0.30 | ~0.20 | ~10 |
| AKAZE_BF | ~0.40 | ~0.30 | ~30 |

### Forgetting Comparison

| Method | Forgetting Rate (Illum→View) |
|--------|------------------------------|
| Naive | ~30-40% |
| EWC | ~15-25% |
| LwF | ~10-20% |
| SI | ~15-25% |
| MAML | ~5-15% |

## Configuration

See `configs/benchmark.yaml` for all configurable options:

```yaml
# Key hyperparameters
batch_size: 32
lr: 1.0e-4
deep_epochs: 15
epochs_source: 15
epochs_target: 15
meta_epochs: 10

# Method-specific
ewc_lambda: 1000.0
lwf_lambda: 1.0
meta_lr_inner: 1.0e-3
```

## Logging

Results are logged to:
- **Weights & Biases**: Real-time metrics and visualizations
- **TensorBoard**: Optional, enable in config
- **CSV files**: Local backup in `results/logs/`
- **JSON**: Complete results in `results.json`

## Citation

If you use this code, please cite:

```bibtex
@misc{meta-feature-matching,
  title={Meta-Learning for Feature Matching: A Benchmark},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/meta-feature-matching}}
}
```

## References

- [HPatches](https://github.com/hpatches/hpatches-benchmark) - Balntas et al., 2017
- [MAML](https://arxiv.org/abs/1703.03400) - Finn et al., 2017
- [EWC](https://arxiv.org/abs/1612.00796) - Kirkpatrick et al., 2017
- [LwF](https://arxiv.org/abs/1606.09282) - Li & Hoiem, 2017
- [SI](https://arxiv.org/abs/1703.04200) - Zenke et al., 2017
- [HardNet](https://arxiv.org/abs/1705.10872) - Mishchuk et al., 2017

## License

MIT License
