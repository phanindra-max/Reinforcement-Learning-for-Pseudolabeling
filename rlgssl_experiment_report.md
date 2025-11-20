# RLGSSL Debugging Guide

## 🚨 Critical Issues Found (November 5, 2025)

### Summary of Problems
Your RLGSSL implementation had **4 critical bugs** causing catastrophic performance:

1. ✅ **FIXED**: Inverted RL loss formula
2. ✅ **FIXED**: Lambda values too small (0.1 → 1.0)
3. ✅ **FIXED**: Stop gradient blocking backpropagation (MOST CRITICAL!)
4. ⚠️ **NEEDS VERIFICATION**: Reward function implementation

---

## Bug #1: Inverted RL Loss Formula ✅ FIXED

### The Problem
**Location**: `src/rlgssl/rl_loss.py`, line 100

**Original (BROKEN) Code**:
```python
weighted_loss = -kl_weights.mean() * reward  # WRONG!
```

**Why It Failed**:
- Reward is **already negative** (≤ 0) from MSE: `reward = -mse_loss`
- Adding another negative sign: `loss = -(kl * reward)` = `loss = -(kl * (-mse))`
- This becomes: `loss = kl * mse` (positive)
- **BUT**: When you do gradient descent (minimize loss), you push toward **MORE negative rewards** (worse predictions)!

**Symptoms**:
- Performance degrades after warmup (46% → 41%)
- Best accuracy at epoch 60 (right after warmup ends)
- Teacher accuracy at 10-14% (random guessing)
- CIFAR-100 collapse: 42% → 0.99%

**Fixed Code**:
```python
weighted_loss = kl_weights.mean() * (-reward)  # CORRECT!
```

**Verification**:
- After warmup, accuracy should **improve** or **stabilize**, not degrade
- Reward values should trend toward 0 (less negative)
- RL loss should decrease over time

---

## Bug #2: Lambda Values Too Small ✅ FIXED

### The Problem
**Location**: `src/train_rlgssl.py`, lines 51-52

**Original**:
```python
self.lambda_sup = 0.1  # TOO SMALL
self.lambda_cons = 0.1  # TOO SMALL
```

**Why It Failed**:
Your loss components were:
- RL loss: ~0.1-0.3
- Supervised: ~0.9-2.3 × **0.1** = 0.09-0.23
- Consistency: ~0.05-0.6 × **0.1** = 0.005-0.06

The **broken RL loss dominated** the optimization, destroying the model.

**Fixed**:
```python
self.lambda_sup = 1.0  # Balanced
self.lambda_cons = 1.0  # Balanced
```

**Verification**:
- All three loss components should be similar magnitude
- Supervised loss should guide the model with strong signal
- TensorBoard: Check `Loss/Supervised` vs `Loss/RL_Loss` balance

---

## Bug #3: Stop Gradient Blocking Backpropagation ✅ FIXED (CRITICAL!)

### The Problem
**Location**: `src/rlgssl/rl_loss.py`, line 99-139

**Original (CATASTROPHICALLY BROKEN) Code**:
```python
# Compute reward on mixup data (with stop gradient)
reward = self.reward_function.compute_reward(
    model, mixup_data, mixup_labels, stop_gradient=True  # WRONG!
)

# RL loss using scalar reward
weighted_loss = kl_weights.mean() * (-reward)  # NO GRADIENT FLOW!
```

**Why It Failed**:
This was the **MOST CRITICAL BUG** - the RL loss had **ZERO gradient flow**!

1. **Stop Gradient Problem**: `stop_gradient=True` meant the reward was computed with `torch.no_grad()`, so no gradients could flow back to the student model
2. **Scalar Reward Problem**: The reward was a single scalar value (mean over batch), not per-sample
3. **Result**: The RL loss component was **completely disconnected** from the model's parameters - it couldn't train anything!

**What Actually Happened**:
- Warmup (epochs 0-50): Model trained with supervised + consistency losses → 48% accuracy ✅
- RLGSSL phase (epochs 51+): RL loss had no gradients, so only supervised + consistency trained
- But supervised loss was on **tiny labeled set** (1000 samples)
- Consistency loss became useless as teacher degraded (following broken student)
- Model collapsed to 10% (random guessing)

**Fixed Code**:
```python
# Get model predictions on mixup data WITH gradients
model_predictions = model.get_probabilities(mixup_data)

# Compute per-sample MSE
per_sample_mse = F.mse_loss(
    model_predictions, mixup_labels, reduction='none'
).mean(dim=1)

# Per-sample rewards (negative MSE)
per_sample_rewards = -per_sample_mse

# RL loss with proper gradient flow
per_sample_loss = kl_weights * (-per_sample_rewards)
weighted_loss = per_sample_loss.mean()
```

**Key Changes**:
1. ✅ Removed `stop_gradient=True` - predictions computed WITH gradients
2. ✅ Per-sample rewards instead of scalar mean
3. ✅ Proper gradient flow: `model → predictions → MSE → reward → loss → backprop → model`

**Understanding sg[θ] in the Paper**:
- The paper's notation `R(s,a;sg[θ])` was misunderstood
- It means: stop gradient through **mixup labels** (which come from teacher)
- It does **NOT** mean: stop gradient through student model predictions!
- The student model MUST receive gradients from the RL loss

**Verification**:
```python
# Test gradient flow
model.zero_grad()
loss = rl_loss_fn.compute_rl_loss(model, unlabeled_data, mixup_data, mixup_labels)
loss['rl_loss'].backward()

# Should print True and show non-zero gradient norm
has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"Gradients computed: {has_gradients}")
```

---

## Bug #4: Reward Function (NEEDS VERIFICATION)

### What to Check

**Location**: `src/rlgssl/reward_function.py`

The reward function should return:
1. **Scalar reward** (mean over batch) for standard RL loss
2. **Negative MSE**: `reward = -mse_loss` where MSE ≥ 0, so reward ≤ 0

**Verify**:
```python
# Check your reward_function.py
def compute_reward(self, model, mixup_data, mixup_labels, stop_gradient=True):
    # Should compute: -MSE between predictions and mixup_labels
    # Return should be ≤ 0 (scalar)
```

**Test Script**:
```bash
cd src
python -c "
from rlgssl.reward_function import RLRewardFunction
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reward_fn = RLRewardFunction(device=device)

# Create dummy model and data
from models.networks import create_model
model = create_model('cnn13', num_classes=10).to(device)
mixup_data = torch.randn(16, 3, 32, 32, device=device)
mixup_labels = torch.softmax(torch.randn(16, 10, device=device), dim=1)

reward = reward_fn.compute_reward(model, mixup_data, mixup_labels)
print(f'Reward value: {reward.item():.6f}')
print(f'Is scalar: {reward.dim() == 0}')
print(f'Is negative: {reward.item() <= 0}')
"
```

**Expected Output**:
```
Reward value: -X.XXXXXX  (some negative number)
Is scalar: True
Is negative: True
```

---

## How to Test the Fixes

### 1. Quick Verification Test (5 minutes)

```bash
cd /home/g34923353/Desktop/group-11/src
python run_experiments.py --mode quick
```

**What to Look For**:
- ✅ Accuracy should **increase** or **stay stable** after epoch 60 (warmup end)
- ✅ Teacher accuracy should be **similar** to student (within 5-10%)
- ✅ Reward should trend toward 0 (becoming less negative)
- ✅ Final accuracy > warmup accuracy
- ✅ **NEW**: RL loss should show non-zero gradients in backprop

**Red Flags**:
- ❌ Accuracy drops significantly after warmup
- ❌ Teacher accuracy < 20% (random guessing)
- ❌ Reward becoming more negative over time
- ❌ **NEW**: RL loss gradient norm = 0 (no training happening)

### 2. Single Experiment Test (2 hours)

```bash
cd /home/g34923353/Desktop/group-11/src
python3 run_experiments.py --mode single --dataset cifar10 --architecture cnn13 --num_labeled 1000
```

**Expected Results** (CIFAR-10, 1000 labels, CNN-13):
- Warmup accuracy: ~50-60%
- Best accuracy: **75-85%** (compared to paper: ~85-90%)
- Final accuracy: **70-80%**
- Teacher-student gap: **< 10%**

### 3. Monitor with TensorBoard

```bash
# In a separate terminal
cd /home/g34923353/Desktop/group-11
tensorboard --logdir=experiments
```

**Open**: http://localhost:6006/

**Check These Plots**:
1. **Accuracy/Student**: Should show upward trend after warmup
2. **RL/Reward**: Should trend toward 0 (less negative)
3. **Loss/Total**: Should decrease over time
4. **Loss/RL_Loss**: Should be similar magnitude to supervised loss

---

## Expected Performance Targets

### CIFAR-10 (from paper Table 1)

| Labels | Architecture | Paper Result | Your Target |
|--------|-------------|--------------|-------------|
| 1000   | CNN-13      | ~85-90%      | **75-85%**  |
| 2000   | CNN-13      | ~88-92%      | **80-88%**  |
| 4000   | CNN-13      | ~90-93%      | **85-90%**  |
| 1000   | WRN-28      | ~87-92%      | **78-88%**  |

### CIFAR-100 (from paper Table 1)

| Labels | Architecture | Paper Result | Your Target |
|--------|-------------|--------------|-------------|
| 4000   | CNN-13      | ~48-52%      | **45-50%**  |
| 10000  | CNN-13      | ~55-60%      | **50-55%**  |

**Note**: Some gap from paper is expected due to:
- Implementation details
- Hyperparameter tuning
- Data augmentation differences

---

## Additional Diagnostics

### Check for Other Issues

1. **Data Augmentation**:
   ```bash
   cd src/data
   python ssl_datasets.py
   ```
   Should show normalized images with proper augmentation.

2. **Model Architecture**:
   ```python
   from models.networks import create_model
   model = create_model('cnn13', num_classes=10)
   print(sum(p.numel() for p in model.parameters()))
   # CNN-13: ~6-10M parameters
   # WRN-28: ~36M parameters
   ```

3. **Teacher-Student EMA**:
   - Teacher should **slowly** track student
   - EMA decay = 0.999 means 99.9% old, 0.1% new
   - After warmup, teacher should have similar accuracy

### Common Issues

**Issue**: Teacher accuracy stuck at 10-20%
- **Cause**: EMA update not being called
- **Fix**: Check `framework.update_teacher()` is called after each batch

**Issue**: NaN losses
- **Cause**: Gradient explosion or numerical instability  
- **Fix**: Enable gradient clipping:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

**Issue**: Out of memory
- **Cause**: Batch size too large
- **Fix**: Reduce batch_size from 128 to 64 or 32

---

## Next Steps

1. ✅ **Fixes Applied**: RL loss formula and lambda values corrected
2. ✅ **Fixes Applied**: Stop gradient blocking backpropagation fixed
3. ⚠️ **TODO**: Verify reward function returns correct values
4. 🧪 **Run Test**: Execute quick test to verify fixes work
5. 📊 **Monitor**: Use TensorBoard to track training progress
6. 📈 **Validate**: Compare results against expected targets

---
