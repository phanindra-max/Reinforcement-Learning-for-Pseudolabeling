# RLGSSL Implementation Documentation

- Author: Satya Phanindra Kumar Kalaga
- GitHub: [phanindra-max](https://github.com/phanindra-max)

## Overview

This implementation provides a complete reproduction of **"Reinforcement Learning-Guided Semi-Supervised Learning"** (Heidari et al., arXiv:2405.01760v1) with significant adaptive enhancements for improved performance and stability.

## Core Implementation Details

### 1. Architecture Components

#### **Teacher-Student Framework** (`src/rlgssl/teacher_student.py`)
- **EMA Teacher**: Exponential Moving Average with decay β = 0.999
- **Update Formula**: `θ_T = β × θ_T + (1 - β) × θ_S`
- **Purpose**: Provides stable pseudo-labels through temporal ensemble
- **Evaluation**: Both student and teacher accuracies tracked simultaneously

#### **Mixup Data Generation** (`src/rlgssl/mixup.py`)
- **Formula**: `x_i^m = μ × x_i^u + (1-μ) × x_i^l` and `y_i^m = μ × y_i^u + (1-μ) × y_i^l`
- **μ Sampling**: Beta(1.0, 1.0) distribution (uniform over [0,1])
- **Size Handling**: Automatic replication of labeled data to match unlabeled batch size
- **Label Conversion**: Automatic hard-to-soft label conversion via one-hot encoding

#### **Reward Function** (`src/rlgssl/reward_function.py`)
- **Core Formula**: `R(s,a;sg[θ]) = -MSE(P_θ(x_i^m), y_i^m)`
- **Stop Gradient**: Implements `sg[θ]` to prevent direct gradient flow
- **Evaluation**: Model predictions on synthetic mixup data vs mixup labels

#### **RL Loss Function** (`src/rlgssl/rl_loss.py`)
- **Core Formula**: `L_rl = -E[y_i^u ~ π_θ] KL(e, y_i^u) × R(s,a;sg[θ])`
- **KL Weighting**: Uniform baseline `e` for policy gradient computation
- **Integration**: Combined with supervised and consistency losses

## Personal Notes: RL vs RLGSSL

I was getting confused thinking I needed to build separate actor-critic networks or something. But actually:

- My CNN-13/WRN-28 models (wrapped in PolicyNetwork) **are** the RL policy
- The "actions" are just the pseudo-labels (probability distributions) they output
- "States" are the unlabeled data points
- "Rewards" come from how well the model does on mixup data
- Policy gets updated through my RL loss function

So the RL part isn't separate - it's built into how I'm already training.

### The math is different from normal RL

```python
# What I'm doing (from the paper):
L_rl = -E[y_i^u ~ π_θ] KL(e, y_i^u) × R(s,a;sg[θ])

# Normal policy gradient would be:
L_pg = -E[log π_θ(a|s) × R(s,a)]

# PPO would add clipping:
L_ppo = -E[min(ratio × A(s,a), clip(ratio, 1-ε, 1+ε) × A(s,a))]
```

### Do I need PPO clipping or policy gradients?

No, because:

**PPO clipping**: That's for when you're doing sequential RL (like games) and need to prevent huge policy updates. But I'm just doing one-shot pseudo-labeling per batch - no sequences. Plus I already have stability from `stop_gradient=True` and curriculum learning.

**Traditional policy gradients**: The reward in my setup is a scaling factor, not a direct optimization target. The KL divergence weighting + curriculum learning actually works better for this specific problem.

This is a novel RL formulation from the paper, not standard REINFORCE or anything.

### What about a custom environment?

Don't need one! My `rlgssl_training_step()` function already IS my environment:

```python
# It already handles:
# - State: unlabeled_data
# - Action: pseudo_labels from teacher  
# - Environment dynamics: mixup generation + model updates
# - Reward: computed from mixup performance
# - State transitions: EMA updates + gradient steps
```

Building a separate gym.Env would actually be worse because:
- Extra overhead from step/reset calls
- Can't process batches efficiently 
- Unnecessary complexity

### What I actually have vs what I thought I needed

✅ Already have:
- RL policy (PolicyNetwork)
- Reward function (RLRewardFunction + adaptive version)
- RL loss (RLLossFunction + adaptive version)
- Environment (implicit in training loop)
- Policy updates (via loss functions)
- Stability mechanisms (stop gradients + curriculum learning)

❌ Don't need:
- Separate actor-critic networks
- PPO clipping 
- Traditional policy gradients
- Custom gym environment
- Q-networks or value functions

**Bottom line**: My implementation looks complete and optimal. I was just confused about what "RL" meant in this context!

---

## Enhancements Over Original Paper

### 1. Adaptive Reward Function Enhancements

**Beyond basic negative MSE, this implementation adds:**

#### **Confidence Bonus** (Weight: 0.1)
```python
confidence_alignment = pred_confidence × target_confidence
adaptive_reward += 0.1 × confidence_alignment.mean()
```
- **Purpose**: Reward confident predictions on confident targets
- **Benefit**: Faster learning on clear cases, reduced noise on uncertain cases

#### **Diversity Bonus** (Weight: 0.05)
```python
batch_entropy = -Σ(batch_mean_pred × log(batch_mean_pred))
normalized_entropy = batch_entropy / log(num_classes)
adaptive_reward += 0.05 × normalized_entropy
```
- **Purpose**: Encourage diverse predictions across batch
- **Benefit**: Prevents mode collapse, maintains exploration

#### **Mixup Effectiveness Bonus**
```python
mu_diversity = std(mu_values)
mixup_bonus = per_sample_accuracy.mean() × (1 + mu_diversity)
```
- **Purpose**: Reward performance across diverse mixup ratios
- **Benefit**: Ensures robustness across interpolation spectrum

### 2. Adaptive RL Loss Enhancements

#### **Curriculum Learning**
```python
curriculum_weight = min(1.0, epoch / (max_epochs × 0.5))
adaptive_weights = kl_weights × (1 + curriculum_weight × confidence_mask)
```
- **Purpose**: Gradually focus on high-confidence samples
- **Benefit**: Stable training progression from easy to hard samples

#### **Entropy Regularization** (Weight: 0.01)
```python
entropy_reg = -Σ(pseudo_labels × log(pseudo_labels))
rl_loss = -weighted_reward + 0.01 × entropy_reg
```
- **Purpose**: Prevent overconfident pseudo-label predictions
- **Benefit**: Maintains exploration, reduces overfitting

#### **Confidence-Based Sample Weighting**
```python
confidence_mask = (max_probs > 0.8).float()
adaptive_weights = kl_weights × (1 + curriculum × confidence_mask)
```
- **Purpose**: Focus learning on reliable pseudo-labels
- **Benefit**: Reduced noise from uncertain predictions

### 3. Adaptive Mixup Strategies

#### **Confidence-Based Mixup**
```python
confidence_boost = (confidence_scores > 0.8).float() × 0.2
adaptive_mu = clamp(base_mu + confidence_boost, 0.0, 1.0)
```
- **Purpose**: Higher μ (more unlabeled) for confident pseudo-labels
- **Benefit**: Leverages high-quality pseudo-labels more effectively

#### **Entropy-Based Mixup**
```python
entropy_penalty = normalized_entropy × 0.3
adaptive_mu = clamp(base_mu - entropy_penalty, 0.0, 1.0)
```
- **Purpose**: Lower μ (more labeled) for uncertain pseudo-labels
- **Benefit**: Reduces reliance on noisy pseudo-labels

### 4. Configuration & Default Settings

- For the reward function, the original paper just used negative MSE, but I added confidence and diversity bonuses. 
- These enhancements are enabled by default.
- The RL loss in the paper was just basic KL weighting, but I included curriculum learning and entropy regularization. 
- These are also on by default.

**Key Configuration**:
```python
self.use_adaptive_rl = True  # Enhanced components active by default
self.confidence_weight = 0.1
self.diversity_weight = 0.05
self.confidence_threshold = 0.8
self.entropy_regularization = 0.01
```

---

# Experimental Results Template (To Be Filled)

## Experimental Setup

### Dataset Configuration
- **Dataset**: [CIFAR-10/CIFAR-100/SVHN]
- **Total Training Samples**: [50,000]
- **Labeled Samples**: [1000/2000/4000]
- **Unlabeled Samples**: [49,000/48,000/46,000]
- **Test Samples**: [10,000]

### Model Configuration
- **Architecture**: [CNN-13/WRN-28]
- **Total Parameters**: 3,567,370
- **Trainable Parameters**: 3,567,370

### Training Configuration
- **Warmup Epochs**: 50
- **RLGSSL Epochs**: 400
- **Total Epochs**: 450
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Device**: [CPU/CUDA]

### Hyperparameters
- **EMA Decay (β)**: 0.999
- **Mixup Alpha (α)**: 1.0
- **λ_sup**: 0.1
- **λ_cons**: 0.1
- **Confidence Threshold**: 0.8
- **Enhancement Weights**: confidence=0.1, diversity=0.05

## Training Results

### Warmup Phase Results (After 5 Epochs)
- **Student Accuracy**: [29.12%]
- **Teacher Accuracy**: [10.00%]
- **Analysis**: [Expected behavior - student learns faster initially due to direct optimization while teacher follows via EMA]

### RLGSSL Phase Progress

| Epoch | Accuracy (%) | Loss | RL Loss | Reward | Notes |
|-------|--------------|------|---------|--------|-------|
| 60    | 36.51         | 0.2101 | 0.0102 | -0.0298 | stable initial training |
| 120   | 43.06         | 0.1808 | -0.0018 | -0.0298 | steady improvement in accuracy |
| 175   | 45.55         | 0.1285  | -0.0383 | -0.0294 | highest accuracy before my system started hanging |
| 180   | 44.98         | 0.1180 | -0.0459 | -0.0292 | system still hanging |
| **240   | 36.20         | -0.0485 | -0.2837 | -0.0342 | system crashed after ~275 epochs. Likely due to the reward function enhancements. TODO: Check if simple negative MSE reward is stable. |
| 300   | -         | - | - | - | - |
| 360   | -         | - | - | - | - |
| 420   | -         | - | - | - | - |
| 450   | -         | - | - | - | - |

### Final Results (To Be Filled)

#### **Performance Metrics**
- **Best Student Accuracy**: 45.55% (Epoch 175)
- **Final Student Accuracy**: [%]
- **Final Teacher Accuracy**: [%]
- **Training Time**: 15 hours + system crash (TODO: Re-run the experiments on GPU instances)

#### **Loss Component Analysis**
- **Final RL Loss**: [value]
- **Final Supervised Loss**: [value] 
- **Final Consistency Loss**: [value]
- **Final Reward**: [value]


## Comparison with Paper Results (To Be Filled)
Note: Not a priority. Could be relevant for the final report/publication.

### Expected vs Actual Performance

| Dataset | Labels | Architecture | Paper Result | Our Result | Difference | Analysis |
|---------|--------|--------------|--------------|------------|------------|----------|
| CIFAR-10 | 1000 | CNN-13 | **9.15%** error | [X.XX%] error | [±X.XX%] | [Analysis of difference] |
| CIFAR-10 | 2000 | CNN-13 | **7.18%** error | [X.XX%] error | [±X.XX%] | [Analysis of difference] |
| CIFAR-10 | 4000 | CNN-13 | **6.11%** error | [X.XX%] error | [±X.XX%] | [Analysis of difference] |

**Note**: Paper results are test **error rates**, convert your accuracy to error: `Error = 100 - Accuracy`

