# 🚀 Google Colab Training Guide for Titan-Optimal

## Overview

This guide explains how to use the `Titan_Optimal_Colab_Training.ipynb` notebook to train Titan-Optimal models on Google Colab.

## ✅ What Has Been Fixed

### Folder Name Changes
- **OLD**: `titan-optimal/` (hyphens not supported by Colab)
- **NEW**: `titan_optimal/` (underscores work perfectly)

### All References Updated
All Python files, documentation, and notebooks have been updated to use `titan_optimal` instead of `titan-optimal`.

## 📋 Prerequisites

### Google Colab Account
- Free tier works for small model (124M parameters)
- Colab Pro recommended for medium model (340M parameters)

### Storage Requirements
- **Google Drive**: At least 2-5 GB free space
- **Checkpoint sizes**:
  - Small model: ~500MB per checkpoint
  - Medium model: ~1.3GB per checkpoint

### GPU Access
- **Free (T4)**: 16GB VRAM - good for small model
- **Pro (V100)**: 16GB VRAM - better for medium model
- **Pro+ (A100)**: 40GB VRAM - best for both models

## 🎯 Quick Start

### Step 1: Prepare Your Repository

1. **Commit changes to your repository**:
   ```bash
   git add .
   git commit -m "Rename titan-optimal to titan_optimal for Colab compatibility"
   ```

2. **Create the Colabnotebook branch**:
   ```bash
   git checkout -b Colabnotebook
   git push origin Colabnotebook
   ```

3. **Update the notebook**: Edit `Titan_Optimal_Colab_Training.ipynb` and replace:
   ```python
   # Change this line (Step 4 - Clone Repository):
   !git clone --depth 1 --branch Colabnotebook https://github.com/YOUR_USERNAME/LLMs-from-scratch.git {repo_path}
   
   # To your actual repository:
   !git clone --depth 1 --branch Colabnotebook https://github.com/yourusername/LLMs-from-scratch.git {repo_path}
   ```

### Step 2: Upload to Colab

**Option A: Direct Upload**
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Select `Titan_Optimal_Colab_Training.ipynb`

**Option B: From GitHub**
1. In Colab: File → Open notebook → GitHub
2. Enter your repository URL
3. Select `titan_optimal/Titan_Optimal_Colab_Training.ipynb`

**Option C: From Google Drive**
1. Upload notebook to your Google Drive
2. Right-click → Open with → Google Colaboratory

### Step 3: Enable GPU

1. Click `Runtime` → `Change runtime type`
2. Hardware accelerator: Select **GPU**
3. GPU type (if available):
   - Free: Automatic (usually T4)
   - Pro: T4 or V100
   - Pro+: V100 or A100
4. Click **Save**

### Step 4: Run the Notebook

1. **Run all cells in order**: Click `Runtime` → `Run all`
2. **Authorize Google Drive** when prompted
3. **Wait for training to complete**

The notebook will:
- ✅ Check GPU availability
- ✅ Mount Google Drive
- ✅ Install dependencies
- ✅ Clone repository (Colabnotebook branch)
- ✅ Load training data
- ✅ Create model
- ✅ Train model
- ✅ Save checkpoints to Drive
- ✅ Generate visualizations

## ⚙️ Configuration

### Model Size
```python
CONFIG = {
    "model_size": "small",  # Options: "small" (124M), "medium" (340M)
    ...
}
```

### Batch Size by GPU
| GPU | VRAM | Small Model | Medium Model |
|-----|------|-------------|--------------|
| T4  | 16GB | 8-12        | 2-4          |
| V100| 16GB | 12-16       | 4-6          |
| A100| 40GB | 16-24       | 8-12         |

The notebook automatically adjusts batch size based on detected GPU.

### Training Parameters
```python
CONFIG = {
    "num_epochs": 10,                  # Number of training epochs
    "learning_rate": 5e-4,             # AdamW learning rate
    "weight_decay": 0.1,               # L2 regularization
    "gradient_accumulation_steps": 4,  # Effective batch size multiplier
    "save_every_n_epochs": 2,          # Checkpoint frequency
    "context_length": 512,             # Sequence length (small: 512, medium: 1024)
}
```

## 📁 Output Files

After training, the following files will be saved to Google Drive:

### Location
```
Google Drive → MyDrive → Titan_Optimal_Checkpoints/
```

### Files
1. **Best model**: `titan_optimal_small_best.pth`
2. **Periodic checkpoints**: `titan_optimal_small_epoch_N.pth`
3. **Training metrics**: `training_results_small.json`
4. **Visualization**: `training_progress.png`

### Checkpoint Contents
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'train_loss': float,
    'val_loss': float,
    'perplexity': float,
}
```

## 🔧 Troubleshooting

### Issue: GPU Not Available
**Error**: `GPU not available, using CPU`

**Solutions**:
1. Runtime → Change runtime type → GPU
2. Restart runtime: Runtime → Restart runtime
3. Check GPU quota (free tier has limits)
4. Try again later if quota exceeded

### Issue: Out of Memory
**Error**: `CUDA out of memory`

**Solutions**:
1. Reduce batch size in config:
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

### Issue: Import Errors
**Error**: `ModuleNotFoundError: No module named 'titan_gpt_v3'`

**Solutions**:
1. Ensure repository is cloned correctly
2. Check that the `titan_optimal` folder exists:
   ```python
   !ls -la /content/LLMs-from-scratch/titan_optimal/
   ```
3. Verify branch name is correct (Colabnotebook)
4. Check that paths are added to sys.path

### Issue: Repository Clone Failed
**Error**: `fatal: Remote branch Colabnotebook not found`

**Solutions**:
1. Verify branch exists:
   ```bash
   git branch -a
   ```
2. Create and push the branch:
   ```bash
   git checkout -b Colabnotebook
   git push origin Colabnotebook
   ```
3. Update repository URL in notebook

### Issue: Runtime Disconnected
**Problem**: Training interrupted by disconnection

**Solutions**:
1. **Prevention**:
   - Use Colab Pro for longer runtimes
   - Keep browser tab active
   - Run during off-peak hours

2. **Recovery**:
   - Checkpoints are saved to Drive
   - Load last checkpoint:
   ```python
   checkpoint = torch.load('/content/drive/MyDrive/Titan_Optimal_Checkpoints/titan_optimal_small_epoch_6.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   start_epoch = checkpoint['epoch']
   # Continue training...
   ```

## 📊 Monitoring Training

### Real-time Monitoring
```python
# In a separate cell, run:
import time
while True:
    !nvidia-smi
    time.sleep(30)  # Update every 30 seconds
```

### Check Training Progress
The notebook displays:
- Epoch progress with tqdm bars
- Train/validation loss
- Perplexity metrics
- Checkpoint saves
- Inference times

### GPU Utilization
```bash
!nvidia-smi
```
Should show:
- GPU usage: >80% utilization
- Memory: Near capacity but not exceeded

## 🎓 Best Practices

### 1. Start Small
- Train small model for 2-3 epochs first
- Verify everything works
- Then scale up to full training

### 2. Monitor Resources
- Check GPU usage regularly
- Ensure GPU utilization is high (>80%)
- Watch for memory warnings

### 3. Save Frequently
```python
CONFIG["save_every_n_epochs"] = 1  # For long training
```

### 4. Use Mixed Precision (Optional)
For faster training and lower memory:
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

## 📈 Expected Training Times

| Model  | GPU  | Epochs | Estimated Time |
|--------|------|--------|----------------|
| Small  | T4   | 10     | 30-45 min      |
| Small  | V100 | 10     | 15-20 min      |
| Medium | T4   | 10     | 1.5-2 hours    |
| Medium | V100 | 10     | 45-60 min      |
| Medium | A100 | 10     | 20-30 min      |

## 🔗 Additional Resources

### Documentation
- [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### Research Papers
- **Titans Architecture**: [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
- **Compute-Optimal Sampling**: [arXiv:2408.16737](https://arxiv.org/abs/2408.16737)

## 🆘 Getting Help

### Check First
1. This README
2. Error message carefully
3. Colab FAQ
4. GitHub Issues

### Where to Ask
- GitHub Discussions
- Stack Overflow (tags: google-colab, pytorch)
- Reddit: r/MachineLearning

## ✅ Success Checklist

- [ ] GPU enabled and verified
- [ ] Google Drive mounted
- [ ] Repository cloned (Colabnotebook branch)
- [ ] Folder name is `titan_optimal` (not `titan-optimal`)
- [ ] Training started successfully
- [ ] Checkpoints saving to Drive
- [ ] Training progress visible
- [ ] Final results saved
- [ ] Visualizations generated
- [ ] Checkpoints accessible in Drive

## 🎉 Summary

You now have a complete Google Colab training setup for Titan-Optimal models with:
- ✅ Colab-compatible folder names (no hyphens)
- ✅ Automatic GPU detection and configuration
- ✅ Google Drive integration for checkpoints
- ✅ Progress monitoring and visualization
- ✅ Error recovery and troubleshooting

Happy Training! 🚀
