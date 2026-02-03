# 🚀 Google Colab Training Setup for Titan-GPT V3

Complete guide to train Titan-GPT V3 models on Google Colab with GPU and save checkpoints to Google Drive.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Two Methods](#two-methods)
- [Requirements](#requirements)
- [Step-by-Step Guide](#step-by-step-guide)
- [Configuration](#configuration)
- [GPU Selection](#gpu-selection)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## 🎯 Quick Start

### Method 1: Jupyter Notebook (Recommended)
1. Upload `Titan_V3_Colab_Training.ipynb` to Google Colab
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Run all cells in order
4. Checkpoints automatically save to Google Drive

### Method 2: Standalone Script
1. Upload `colab_train_standalone.py` to Colab
2. Enable GPU: `Runtime → Change runtime type → GPU`
3. Mount Google Drive manually
4. Run: `!python colab_train_standalone.py`

---

## 📦 Requirements

### Google Colab Resources
- **Free Tier**: T4 GPU (16GB VRAM) - Good for small model
- **Colab Pro**: V100 GPU (16GB VRAM) - Better performance
- **Colab Pro+**: A100 GPU (40GB VRAM) - Best for medium model

### Storage
- Google Drive with at least 2-5 GB free space for checkpoints
- Each checkpoint: ~500MB (small) or ~1.3GB (medium)

### Time Estimates
| Model | GPU | Epochs | Time |
|-------|-----|--------|------|
| Small | T4 | 10 | ~30-45 min |
| Small | V100 | 10 | ~15-20 min |
| Medium | T4 | 10 | ~1.5-2 hours |
| Medium | V100 | 10 | ~45-60 min |
| Medium | A100 | 10 | ~20-30 min |

---

## 📚 Step-by-Step Guide

### Step 1: Upload Files to Colab

**Option A: Upload Notebook**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File → Upload notebook`
3. Upload `Titan_V3_Colab_Training.ipynb`

**Option B: Upload from GitHub**
1. In Colab: `File → Open notebook → GitHub`
2. Enter your repository URL
3. Select the notebook

### Step 2: Enable GPU

1. Click `Runtime → Change runtime type`
2. Hardware accelerator: Select `GPU`
3. GPU type: 
   - Free: Automatic (usually T4)
   - Pro: T4 or V100
   - Pro+: V100 or A100
4. Click `Save`

### Step 3: Verify GPU

Run this cell:
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### Step 4: Mount Google Drive

When prompted, authorize Google Drive access:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 5: Run Training

**For Notebook**: Run all cells sequentially

**For Standalone Script**:
```python
!python colab_train_standalone.py
```

### Step 6: Monitor Progress

Training will show:
- Epoch progress with tqdm bars
- Loss values (train and validation)
- Perplexity metrics
- Checkpoint saves

### Step 7: Access Results

After training, find files in:
```
Google Drive → MyDrive → Titan_V3_Checkpoints/
```

Files saved:
- `titan_v3_small_best.pth` - Best model checkpoint
- `titan_v3_small_epoch_N.pth` - Epoch checkpoints
- `training_results_small.json` - Training metrics
- `training_progress.png` - Visualization

---

## ⚙️ Configuration

### Model Sizes

**Small Model (124M parameters)**
```python
CONFIG = {
    "model_size": "small",
    "batch_size": 8,       # For T4/V100
    "num_epochs": 10,
}
```

**Medium Model (340M parameters)**
```python
CONFIG = {
    "model_size": "medium",
    "batch_size": 4,       # For V100
    "num_epochs": 10,
}
```

### Batch Size by GPU

| GPU | VRAM | Small Model | Medium Model |
|-----|------|-------------|--------------|
| T4 | 16GB | 8-12 | 2-4 |
| V100 | 16GB | 12-16 | 4-6 |
| A100 | 40GB | 16-24 | 8-12 |

### Hyperparameters

```python
CONFIG = {
    "learning_rate": 5e-4,              # AdamW learning rate
    "weight_decay": 0.1,                # L2 regularization
    "gradient_accumulation_steps": 4,   # Effective batch size multiplier
    "save_every_n_epochs": 2,           # Checkpoint frequency
    "context_length": 512,              # Sequence length (small)
    # "context_length": 1024,           # Sequence length (medium)
}
```

---

## 🎮 GPU Selection

### Free Tier (T4)
- **Pros**: Free, 16GB VRAM, sufficient for small model
- **Cons**: Limited runtime (12 hours), may disconnect
- **Best for**: Small model experimentation

### Colab Pro ($10/month)
- **Pros**: V100 access, longer runtime (24 hours), fewer disconnects
- **Cons**: Costs money
- **Best for**: Regular training, small-medium models

### Colab Pro+ ($50/month)
- **Pros**: A100 access, longest runtime, fastest training
- **Cons**: Higher cost
- **Best for**: Large-scale training, production experiments

### Selecting GPU Type (Pro/Pro+)
1. `Runtime → Change runtime type`
2. `Hardware accelerator: GPU`
3. `GPU type: V100` or `A100` (if available)

---

## 🔧 Troubleshooting

### Issue: GPU Not Available

**Error**: `GPU not available, using CPU`

**Solutions**:
1. Check runtime type: `Runtime → Change runtime type → GPU`
2. Restart runtime: `Runtime → Restart runtime`
3. Check GPU quota (free tier has limits)
4. Try again later if quota exceeded

### Issue: Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size:
   ```python
   CONFIG["batch_size"] = 4  # or even 2
   ```
2. Increase gradient accumulation:
   ```python
   CONFIG["gradient_accumulation_steps"] = 8
   ```
3. Use smaller model:
   ```python
   CONFIG["model_size"] = "small"
   ```
4. Reduce context length:
   ```python
   CONFIG["context_length"] = 256
   ```
5. Restart runtime to clear memory:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Issue: Runtime Disconnected

**Error**: Runtime disconnected during training

**Solutions**:
1. **Prevention**:
   - Use Colab Pro for longer runtimes
   - Keep browser tab active
   - Run overnight with Pro+
   
2. **Recovery**:
   - Checkpoints are saved to Drive
   - Load last checkpoint and continue:
   ```python
   checkpoint = torch.load('path/to/last_checkpoint.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   start_epoch = checkpoint['epoch']
   ```

### Issue: Google Drive Full

**Error**: `No space left on device`

**Solutions**:
1. Delete old checkpoints from Drive
2. Reduce checkpoint frequency:
   ```python
   CONFIG["save_every_n_epochs"] = 5
   ```
3. Upgrade Google Drive storage
4. Save only best model:
   ```python
   # Modify code to save only when val_loss improves
   ```

### Issue: Slow Training

**Problem**: Training slower than expected

**Solutions**:
1. Verify GPU is being used:
   ```python
   print(f"Device: {next(model.parameters()).device}")
   ```
2. Check GPU utilization:
   ```bash
   !nvidia-smi
   ```
3. Increase batch size if GPU underutilized
4. Check for CPU bottlenecks (use `num_workers=2`)

### Issue: Import Errors

**Error**: `ModuleNotFoundError: No module named 'titan_gpt_v3'`

**Solutions**:
1. Ensure repository is cloned:
   ```bash
   !git clone https://github.com/rasbt/LLMs-from-scratch.git
   ```
2. Check paths are added:
   ```python
   import sys
   sys.path.append('/content/LLMs-from-scratch')
   sys.path.append('/content/LLMs-from-scratch/titan_optimal')
   ```
3. Verify files exist:
   ```bash
   !ls -la /content/LLMs-from-scratch/titan_optimal/models/
   ```

---

## 🚀 Advanced Usage

### Resume Training from Checkpoint

```python
# Load checkpoint
checkpoint_path = '/content/drive/MyDrive/Titan_V3_Checkpoints/titan_v3_small_epoch_6.pth'
checkpoint = torch.load(checkpoint_path)

# Restore model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

# Continue training
for epoch in range(start_epoch, num_epochs):
    # Training code...
```

### Custom Training Data

Replace the default training data:

```python
# Option 1: Upload your own text file
from google.colab import files
uploaded = files.upload()

# Option 2: Load from URL
import requests
url = "YOUR_TEXT_FILE_URL"
response = requests.get(url)
with open("/content/custom_data.txt", "w") as f:
    f.write(response.text)

# Option 3: Load from Google Drive
text_data_path = "/content/drive/MyDrive/your_data.txt"
with open(text_data_path, "r") as f:
    text_data = f.read()
```

### Multi-GPU Training

For Colab Pro+ with multiple GPUs:

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
```

### Export to Local Machine

Download checkpoints to your computer:

```python
from google.colab import files

# Download best model
files.download('/content/drive/MyDrive/Titan_V3_Checkpoints/titan_v3_small_best.pth')

# Download all checkpoints (zip first)
!zip -r checkpoints.zip /content/drive/MyDrive/Titan_V3_Checkpoints/
files.download('checkpoints.zip')
```

### Convert to ONNX for Deployment

```python
import torch.onnx

# Load model
model.eval()

# Create dummy input
dummy_input = torch.randint(0, 50257, (1, 512)).to(device)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "/content/drive/MyDrive/titan_v3_small.onnx",
    export_params=True,
    opset_version=12,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

### Hyperparameter Tuning

Run multiple experiments with different configs:

```python
configs = [
    {"learning_rate": 3e-4, "batch_size": 8},
    {"learning_rate": 5e-4, "batch_size": 8},
    {"learning_rate": 7e-4, "batch_size": 8},
]

for i, config in enumerate(configs):
    print(f"\nExperiment {i+1}/{len(configs)}")
    CONFIG.update(config)
    results = train_titan_v3_colab(
        checkpoint_dir=f"/content/drive/MyDrive/Experiments/exp_{i+1}"
    )
```

---

## 📊 Monitoring and Logging

### Real-time GPU Monitoring

In a separate cell, run continuously:
```python
import time
while True:
    !nvidia-smi
    time.sleep(30)  # Update every 30 seconds
```

### TensorBoard Integration

Add TensorBoard logging:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('/content/drive/MyDrive/runs')

# In training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Perplexity', perplexity, epoch)

# View in Colab
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/runs
```

### Weights & Biases Integration

```python
!pip install wandb

import wandb
wandb.login()

wandb.init(project="titan-gpt-v3", config=CONFIG)

# In training loop
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "perplexity": perplexity,
    "epoch": epoch
})
```

---

## 📝 Best Practices

### 1. Start Small
- Begin with small model and 2-3 epochs
- Verify everything works
- Then scale up

### 2. Save Frequently
- Set `save_every_n_epochs = 1` for long training
- Prevents loss of progress

### 3. Monitor Resources
- Check GPU usage with `nvidia-smi`
- Ensure GPU utilization is high (>80%)

### 4. Use Mixed Precision (Optional)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    logits = model(input_batch)
    loss = criterion(logits, target_batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5. Clean Up After Training
```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Delete temporary files
!rm -rf /content/LLMs-from-scratch
```

---

## 🎓 Learning Resources

### Official Documentation
- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### Research Papers
- **Titans Architecture**: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **Compute-Optimal Sampling**: [arXiv:2408.16737](https://arxiv.org/abs/2408.16737)

---

## 💡 Tips & Tricks

1. **Keep Browser Active**: Colab may disconnect if tab is inactive (free tier)
2. **Use Colab Pro for Long Training**: Worth it for reliability
3. **Save Intermediate Results**: Don't rely solely on final output
4. **Test on Small Data First**: Validate code before full training
5. **Monitor Drive Space**: Checkpoints can fill up Drive quickly
6. **Use Version Control**: Track configuration changes
7. **Document Experiments**: Keep notes on what works/doesn't work

---

## 🆘 Getting Help

### Check These First
1. This README
2. Error message carefully
3. Colab FAQ
4. GitHub Issues

### Where to Ask
- GitHub Discussions
- Stack Overflow (tag: google-colab, pytorch)
- Reddit: r/MachineLearning, r/learnmachinelearning

---

## 📄 License

This training setup is part of the LLMs-from-scratch project. See main repository for license details.

---

## 🎉 Success Checklist

- [ ] GPU enabled and verified
- [ ] Google Drive mounted
- [ ] Repository cloned
- [ ] Training started successfully
- [ ] Checkpoints saving to Drive
- [ ] Training progress visible
- [ ] Final results saved
- [ ] Visualizations generated
- [ ] Checkpoints accessible in Drive

---

Happy Training! 🚀
