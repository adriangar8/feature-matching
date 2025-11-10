# 🧠 Meta-Feature Matching: From Forgetting to Meta-Learning Adaptation

This repository implements a complete experimental framework for studying **meta-learning in feature matching and descriptor adaptation**.  
We start from a **pretrained visual descriptor** (e.g. SuperPoint), **fine-tune** it on a specific domain to simulate *catastrophic forgetting*, and then apply **meta-learning (MAML/Reptile)** to **restore generalization and rapid adaptability** across new domains.

---

## 📖 Overview

Feature matching lies at the heart of 3D computer vision tasks such as **Structure from Motion**, **Visual SLAM**, and **Multi-View Stereo**.  
While deep descriptors (SuperPoint, D2-Net, R2D2) outperform classical ones (SIFT, ORB), they often **lose generality after domain-specific fine-tuning**.

This project explores how **meta-learning** can help such models **relearn adaptability** — recovering cross-domain robustness after forgetting.

---

## 🚀 Project Pipeline

```
Pretrained Descriptor (SuperPoint)
          ↓
 Fine-Tune on One Domain (e.g., indoor scenes)
          ↓
 Model overfits → forgets other domains
          ↓
 Meta-Learning (MAML/Reptile) across multiple domains
          ↓
 Regains fast adaptation and generalization ability
```

---

## 🧩 Key Features

- ✅ **Pretrained Descriptor Backbone**
  - Plug-and-play support for SuperPoint (extendable to D2-Net or R2D2)
- 🧠 **Meta-Learning Algorithms**
  - Implementations of MAML and Reptile using `higher`
- 🔁 **Catastrophic Forgetting Simulation**
  - Fine-tuning pipeline for controlled domain specialization
- 📊 **Cross-Domain Evaluation**
  - Adaptation speed and matching accuracy metrics
- 📈 **Integrated W&B Logging**
  - Metrics, loss curves, and keypoint visualizations logged automatically

---

## 🗂️ Repository Structure

```
meta-feature-matching/
├── README.md
├── requirements.txt
├── setup.sh
├── configs/
│   ├── base.yaml
│   ├── finetune.yaml
│   ├── maml.yaml
│   └── reptile.yaml
├── data/
│   ├── hpatches/
│   ├── megadepth/
│   └── ...
├── src/
│   ├── models/                # Descriptor architectures (SuperPoint, etc.)
│   ├── datasets/              # HPatches loaders & meta-task samplers
│   ├── meta/                  # MAML & Reptile implementations
│   ├── training/              # Fine-tuning & meta-learning scripts
│   ├── evaluation/            # Cross-domain and adaptation evaluations
│   └── utils/                 # Logger, metrics, visualization
└── results/
    ├── checkpoints/
    ├── logs/
    └── figures/
```

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/<yourusername>/meta-feature-matching.git
   cd meta-feature-matching
   ```

2. **Create environment**
   ```bash
   bash setup.sh
   ```

3. **Log in to Weights & Biases**
   ```bash
   wandb login
   ```

---

## 📚 Datasets

| Dataset | Use | Domains |
|----------|------|----------|
| [HPatches](https://github.com/hpatches/hpatches-dataset) | Training & Evaluation | Illumination, Viewpoint, Blur |
| [MegaDepth](https://megadepth.cs.cornell.edu/) | Optional fine-tuning | Indoor / Outdoor |
| ETH3D / PhotoTourism | Optional evaluation | Real scenes with viewpoint variation |

Each domain acts as a **meta-task** in MAML or Reptile.

---

## 🧪 Training Pipeline

### 1️⃣ Fine-tune to simulate forgetting
```bash
python src/training/train_finetune_forgetting.py --config configs/finetune.yaml
```

- Starts from a pretrained SuperPoint model.
- Fine-tunes on a single domain (e.g., illumination).
- Produces `superpoint_finetuned.pth`.

---

### 2️⃣ Meta-learning restoration (MAML)
```bash
python src/training/train_meta_restore.py --config configs/maml.yaml
```

- Loads the fine-tuned checkpoint.
- Performs meta-learning across multiple domains.
- Produces `maml_restored.pth`.

---

### 3️⃣ Evaluate cross-domain generalization
```bash
python src/evaluation/eval_cross_domain.py
```

Visualizes:
- Matching accuracy per domain.
- Adaptation curves over fine-tuning steps.
- Embedding space (e.g., t-SNE of descriptors).

---

## 📊 Experiment Tracking with W&B

All experiments are automatically logged to [Weights & Biases](https://wandb.ai):

- Training / validation losses
- Matching accuracy and precision–recall
- Keypoint match visualizations
- Hyperparameters (auto-logged from config files)

To view results:
```bash
wandb sync --clean
```

---

## 🧠 Results Overview

| Stage | In-Domain | Cross-Domain | Adaptation Speed |
|--------|------------|--------------|------------------|
| Pretrained SuperPoint | - | - | - |
| After Fine-Tuning | - | - | - |
| After Meta-Learning | - | - | - |

Meta-learning successfully restores generalization and adaptability lost after fine-tuning.

---

## 🧰 Technologies

- **Python 3.10**, **PyTorch ≥ 2.0**
- **Higher** for differentiable inner loops
- **Weights & Biases** for experiment tracking
- **OpenCV / Matplotlib** for visualization

---

## 📄 Citation

If you use this work, please cite or reference:

```bash
@project{meta_feature_matching_2025,
  title={Meta-Feature Matching: From Forgetting to Meta-Learning Adaptation},
  author={Zeeny, Karl},
  institution={École Polytechnique},
  year={2025},
  url={https://github.com/<yourusername>/meta-feature-matching}
}
```

---

## 🧩 Acknowledgments

This project builds upon open-source descriptors:
- [SuperPoint (Magic Leap)](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [D2-Net](https://github.com/mihaidusmanu/d2-net)
- [R2D2 (Naver Labs)](https://github.com/naver/r2d2)

And meta-learning frameworks:
- [Finn et al., *Model-Agnostic Meta-Learning*, ICML 2017]
- [Shalam et al., *Matching by Few-Shot Classification*, BMVC 2023]

---

## ✨ Author

**Karl Zeeny**  
**Adrián García**
