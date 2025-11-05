# TensorBoard Integration Guide

## Overview
Your RLGSSL implementation now includes full TensorBoard support for real-time visualization of training metrics including RL loss, rewards, and accuracy.

---

## 🚀 Quick Start

### 1. Run Training (TensorBoard enabled by default)

```bash
# Run demo
python src/demo.py

# Run single experiment
python src/run_experiments.py --mode single --dataset cifar10 --num_labeled 1000

# Run quick test
python src/run_experiments.py --mode quick
```

### 2. Launch TensorBoard

While training is running (or after it completes):

```bash
# View specific experiment
tensorboard --logdir=experiments/{experiment_name}/tensorboard

# View demo results
tensorboard --logdir=experiments/demo_run/tensorboard

# View all experiments at once
tensorboard --logdir=experiments
```

### 3. Open in Browser

TensorBoard will display:
```
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```

Open your browser and navigate to: **http://localhost:6006/**

---

## 📊 Available Metrics

TensorBoard logs the following metrics organized into categories:

### **Loss/** (Loss Components)
- `Loss/Total` - Combined total loss
- `Loss/RL_Loss` - Reinforcement learning loss
- `Loss/Supervised` - Supervised learning loss (on labeled data)
- `Loss/Consistency` - Consistency loss (teacher-student)

### **RL/** (Reinforcement Learning Metrics)
- `RL/Reward` - Average reward from RL formulation
- `RL/Confidence_Ratio` - Ratio of high-confidence predictions (if using adaptive RL)

### **Accuracy/** (Performance Metrics)
- `Accuracy/Student` - Student model test accuracy
- `Accuracy/Teacher` - Teacher model test accuracy (EMA)
- `Accuracy/Best` - Best accuracy achieved so far

### **Training/** (Training Information)
- `Training/Learning_Rate` - Current learning rate
- `Training/Epoch_Time` - Time per epoch (seconds)

### **Warmup/** (Warmup Phase Results)
- `Warmup/Student_Accuracy` - Accuracy after warmup
- `Warmup/Teacher_Accuracy` - Teacher accuracy after warmup

---

## 🎛️ Configuration

### Enable/Disable TensorBoard

TensorBoard is **enabled by default**. To disable:

```python
from train_rlgssl import RLGSSLConfig, RLGSSLTrainer

config = RLGSSLConfig()
config.use_tensorboard = False  # Disable TensorBoard
config.experiment_name = 'my_experiment'

trainer = RLGSSLTrainer(config)
trainer.train()
```

### Change Log Frequency

```python
config = RLGSSLConfig()
config.eval_frequency = 10  # Log metrics every 10 epochs (default)
```

More frequent logging = more data points in TensorBoard (but slightly slower training)

---

## 📁 File Structure

After running experiments, your directory will look like:

```
experiments/
├── demo_run/
│   ├── tensorboard/               # TensorBoard event files
│   │   └── events.out.tfevents.*
│   ├── config.json
│   ├── training_summary.json
│   └── best_model.pth
├── cifar10_cnn13_1000labels_20250105_143022/
│   ├── tensorboard/               # TensorBoard event files
│   │   └── events.out.tfevents.*
│   ├── config.json
│   ├── training_summary.json
│   └── best_model.pth
...
```

---

## 💡 Usage Examples

### Example 1: Monitor Training in Real-Time

**Terminal 1** (start training):
```bash
python src/run_experiments.py --mode single --dataset cifar10 --num_labeled 1000
```

**Terminal 2** (launch TensorBoard):
```bash
tensorboard --logdir=experiments --reload_interval=30
```

Now watch metrics update in real-time as training progresses!

### Example 2: Compare Multiple Experiments

```bash
# Run multiple experiments
python src/run_experiments.py --mode single --dataset cifar10 --num_labeled 1000
python src/run_experiments.py --mode single --dataset cifar10 --num_labeled 2000
python src/run_experiments.py --mode single --dataset cifar10 --num_labeled 4000

# View all experiments together
tensorboard --logdir=experiments
```

TensorBoard will overlay all experiments on the same graphs for easy comparison!

### Example 3: Custom Port

```bash
# Use custom port if 6006 is busy
tensorboard --logdir=experiments --port=6007
```

### Example 4: Share TensorBoard (Remote Server)

If running on a remote GPU server:

```bash
# On server
tensorboard --logdir=experiments --host=0.0.0.0 --port=6006

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@server_address
```

Then open http://localhost:6006/ on your local machine.

---

## 📈 TensorBoard Features to Use

### 1. **Smoothing**
- Adjust smoothing slider (top left) to smooth noisy curves
- Useful for reward and loss plots

### 2. **Compare Runs**
- Select multiple runs in the left sidebar
- Compare different hyperparameters or datasets

### 3. **Download Data**
- Click "Show data download links" (top left)
- Export metrics as CSV or JSON

### 4. **Custom Scalars**
- Create custom dashboards in the "Time Series" tab
- Group related metrics together

### 5. **HParams (Future Enhancement)**
- Compare hyperparameters across runs
- Find best configurations

---

## 🔍 What to Look For

### Healthy Training Signs
✅ **RL Loss** - Should decrease over time  
✅ **Reward** - Should increase over time  
✅ **Student Accuracy** - Steady increase  
✅ **Consistency Loss** - Should decrease as student matches teacher  

### Potential Issues
⚠️ **Oscillating Accuracy** - May need lower learning rate  
⚠️ **Flat Reward** - Check if mixup is working properly  
⚠️ **Growing Loss** - Possible instability, reduce learning rate  

---

## 🛠️ Troubleshooting

### TensorBoard not showing data
```bash
# Check if event files exist
ls experiments/*/tensorboard/

# Try clearing cache
rm -rf /tmp/.tensorboard-info/

# Reload TensorBoard
tensorboard --logdir=experiments --reload_interval=5
```

### Port already in use
```bash
# Kill existing TensorBoard
pkill -f tensorboard

# Or use different port
tensorboard --logdir=experiments --port=6007
```

### Metrics not updating
- Check that `config.use_tensorboard = True`
- Verify training is actually running
- Ensure `eval_frequency` is set appropriately
- Try decreasing `reload_interval`

---

## 📊 Advanced: Custom Metrics

To add your own metrics, edit `src/train_rlgssl.py`:

```python
# In rlgssl_phase method, add:
if self.writer is not None:
    # Your custom metric
    self.writer.add_scalar('Custom/MyMetric', my_value, epoch + 1)
    
    # Histogram of model weights
    for name, param in self.framework.student_model.named_parameters():
        self.writer.add_histogram(f'Weights/{name}', param, epoch + 1)
    
    # Images (e.g., augmented samples)
    self.writer.add_images('Augmented/Samples', augmented_imgs, epoch + 1)
```

---

## 🎯 Example Workflow

### Complete monitoring setup:

```bash
# Terminal 1: Start training with demo
cd /path/to/your/project
python src/demo.py

# Terminal 2: Launch TensorBoard
tensorboard --logdir=experiments/demo_run/tensorboard --reload_interval=10

# Terminal 3: Monitor GPU usage
watch -n 1 nvidia-smi

# Browser: Open http://localhost:6006/
```

Watch your metrics update in real-time! 🎉

---

## 📝 Tips

1. **Start TensorBoard before training** - See metrics from epoch 0
2. **Use `--reload_interval`** - Faster updates (default: 5 seconds)
3. **Compare experiments** - Use consistent naming schemes
4. **Save screenshots** - Document interesting findings
5. **Export data** - Use CSV exports for custom plots

---

## 🌐 Integration with Remote Environments

### Jupyter Notebook
```python
%load_ext tensorboard
%tensorboard --logdir experiments/
```

### VS Code
- Install "TensorBoard" extension
- Right-click on tensorboard folder → "Launch TensorBoard"

### Google Colab
```python
%load_ext tensorboard
%tensorboard --logdir experiments/
```

---

## ✅ Summary

| Feature | Status | Location |
|---------|--------|----------|
| RL Loss Logging | ✅ Enabled | `Loss/RL_Loss` |
| Reward Logging | ✅ Enabled | `RL/Reward` |
| Accuracy Logging | ✅ Enabled | `Accuracy/Student`, `Accuracy/Teacher` |
| All Loss Components | ✅ Enabled | `Loss/*` |
| Learning Rate | ✅ Enabled | `Training/Learning_Rate` |
| Epoch Time | ✅ Enabled | `Training/Epoch_Time` |
| Real-time Updates | ✅ Supported | Auto-refresh |
| Multi-experiment Compare | ✅ Supported | Use parent dir in `--logdir` |

**TensorBoard is ready to use!** Just run your experiments and launch TensorBoard. 🚀

