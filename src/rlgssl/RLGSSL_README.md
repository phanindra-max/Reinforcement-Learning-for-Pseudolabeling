# RLGSSL Implementation

This repository contains a complete implementation of **Reinforcement Learning-Guided Semi-Supervised Learning (RLGSSL)** based on the paper by Heidari et al. (arXiv:2405.01760v1).

## Overview

RLGSSL formulates semi-supervised learning as a one-armed bandit problem and uses reinforcement learning to adaptively guide the learning process. The key innovation is using a mixup-based reward function to evaluate pseudo-label quality.

### Key Features

- ✅ Complete implementation of Algorithm 1 from the paper
- ✅ CNN-13 and WRN-28 network architectures
- ✅ Teacher-student framework with EMA updates
- ✅ Mixup-based reward function for RL guidance
- ✅ KL-divergence weighted RL loss
- ✅ Support for CIFAR-10, CIFAR-100, and SVHN datasets
- ✅ Extensive logging and experiment tracking
- ✅ Paper reproduction experiments

## Installation

### Prerequisites

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements_rlgssl.txt
```

### Quick Setup

1. **Clone/download the implementation files**
2. **Install dependencies**: `pip install -r requirements_rlgssl.txt`
3. **Run demo**: `python src/demo.py`

## Usage

### Quick Demo

Test the implementation with a minimal example:

```bash
cd src
python demo.py
```

This will:
- Test all individual components
- Run a short training example (5 epochs total)
- Verify the implementation works correctly

### Single Experiment

Run a single experiment:

```bash
cd src
python run_experiments.py --mode single --dataset cifar10 --architecture cnn13 --num_labeled 1000
```

### Quick Test

Run a fast test with reduced epochs:

```bash
cd src
python run_experiments.py --mode quick
```

### Paper Reproduction

Run all experiments from the paper (⚠️ **WARNING: This takes days if GPU is not used!**):

```bash
cd src
python run_experiments.py --mode paper
```

### Custom Training

Use the trainer directly for custom experiments:

```python
from train_rlgssl import RLGSSLConfig, RLGSSLTrainer

# Create custom configuration
config = RLGSSLConfig()
config.dataset_name = 'cifar10'
config.num_labeled = 1000
config.architecture = 'cnn13'
config.warmup_epochs = 50
config.rlgssl_epochs = 400

# Run training
trainer = RLGSSLTrainer(config)
results = trainer.train()
```

## Architecture

### Core Components

1. **Data Loading** (`src/data/ssl_datasets.py`)
   - Semi-supervised dataset splitting
   - Balanced labeled/unlabeled data handling
   - Data augmentation for SSL

2. **Models** (`src/models/networks.py`)
   - CNN-13: 13-layer CNN architecture
   - WRN-28: Wide ResNet with 28 layers
   - PolicyNetwork wrapper for RL formulation

3. **Teacher-Student Framework** (`src/rlgssl/teacher_student.py`)
   - EMA-based teacher model
   - Consistency loss computation
   - Pseudo-label generation

4. **Mixup** (`src/rlgssl/mixup.py`)
   - Linear interpolation between labeled/pseudo-labeled data
   - Size mismatch handling via replication
   - Adaptive mixup strategies

5. **Reward Function** (`src/rlgssl/reward_function.py`)
   - MSE-based reward on mixup data
   - Detailed reward analysis
   - Adaptive reward components

6. **RL Loss** (`src/rlgssl/rl_loss.py`)
   - KL-divergence weighted negative reward
   - Combined loss with supervised and consistency terms
   - Adaptive RL loss variants

7. **Training** (`src/train_rlgssl.py`)
   - Complete Algorithm 1 implementation
   - Warmup + RLGSSL training phases
   - Experiment tracking and checkpointing

### Training Algorithm

The implementation follows Algorithm 1 from the paper:

1. **Warmup Phase** (50 epochs): Standard teacher-student SSL
2. **RLGSSL Phase** (400 epochs): 
   - Generate pseudo-labels with teacher model
   - Create mixup data from labeled and pseudo-labeled samples
   - Compute reward based on model performance on mixup data
   - Update student with combined RL + supervised + consistency loss
   - Update teacher with EMA

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_sup` | 0.1 | Weight for supervised loss |
| `lambda_cons` | 0.1 | Weight for consistency loss |
| `ema_decay` | 0.999 | EMA decay rate for teacher |
| `mixup_alpha` | 1.0 | Beta distribution parameter for mixup |
| `learning_rate` | 0.001 | Learning rate |
| `batch_size` | 128 | Batch size |
| `warmup_epochs` | 50 | Warmup training epochs |
| `rlgssl_epochs` | 400 | RLGSSL training epochs |

### Dataset Configurations

**CIFAR-10/100:**
- Image size: 32×32×3
- Normalization: ImageNet statistics
- Augmentation: RandomHorizontalFlip, RandomCrop with padding

**SVHN:**
- Image size: 32×32×3
- Normalization: SVHN statistics
- Augmentation: RandomCrop with padding

## Paper Results Reproduction

### Expected Results (Test Error %)

| Dataset | Labels | Architecture | Paper Result | 
|---------|--------|--------------|--------------|
| CIFAR-10 | 1000 | CNN-13 | **9.15** |
| CIFAR-10 | 2000 | CNN-13 | **7.18** |
| CIFAR-10 | 4000 | CNN-13 | **6.11** |
| CIFAR-100 | 4000 | CNN-13 | **33.62** |
| CIFAR-100 | 10000 | CNN-13 | **29.12** |
| SVHN | 500 | CNN-13 | **3.12** |
| SVHN | 1000 | CNN-13 | **3.05** |
| CIFAR-10 | 1000 | WRN-28 | **4.92** |
| CIFAR-10 | 4000 | WRN-28 | **3.52** |
| SVHN | 1000 | WRN-28 | **1.92** |

### Running Reproduction Experiments

```bash
# Run all paper experiments (WARNING: Takes days!)
python run_experiments.py --mode paper

# Run specific configuration
python run_experiments.py --mode single \
  --dataset cifar10 \
  --architecture cnn13 \
  --num_labeled 1000
```

## Implementation Details

### Key Design Decisions

1. **Stop Gradient in Reward**: The reward function uses `sg[θ]` to prevent direct gradient flow, adhering to RL principles.

2. **EMA Teacher Updates**: Teacher parameters are updated after each training step using exponential moving average.

3. **Mixup Data Replication**: When labeled data is smaller than unlabeled, we replicate labeled data to match sizes.

4. **Adaptive Components**: Optional adaptive reward and RL loss functions for enhanced performance.

5. **Curriculum Learning**: Gradual increase in focus on high-confidence pseudo-labels.

### Files Structure

```
src/
├── data/
│   ├── __init__.py
│   └── ssl_datasets.py          # SSL dataset handling
├── models/
│   ├── __init__.py
│   └── networks.py              # CNN-13 and WRN-28 architectures
├── rlgssl/
│   ├── __init__.py
│   ├── teacher_student.py       # Teacher-student framework
│   ├── mixup.py                 # Mixup functionality
│   ├── reward_function.py       # RL reward function
│   └── rl_loss.py              # RL loss computation
├── train_rlgssl.py             # Main training script
├── run_experiments.py          # Experiment runner
└── demo.py                     # Quick demo
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `config.batch_size = 64`
   - Use gradient accumulation
   - Try CPU training for small experiments

2. **Slow Training**
   - Use GPU if available
   - Reduce dataset size for testing
   - Use quick mode: `--mode quick`

3. **Poor Performance**
   - Check hyperparameters match paper
   - Ensure proper data augmentation
   - Verify dataset splits are balanced

### Debug Mode

Run with verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{heidari2024rlgssl,
  title={Reinforcement Learning-Guided Semi-Supervised Learning},
  author={Heidari, Marzi and Zhang, Hanping and Guo, Yuhong},
  journal={arXiv preprint arXiv:2405.01760},
  year={2024}
}
```

## License

This implementation is provided for research and educational purposes. Please refer to the original paper for the research contributions.

## Contributing

Feel free to submit issues, improvements, or extensions to this implementation!

