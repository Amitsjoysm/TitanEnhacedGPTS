"""Memory-Efficient Training for Titan-GPT V3 Small

Optimized for systems with limited memory:
- Minimal batch sizes
- Aggressive gradient accumulation
- Memory monitoring
- Reduced evaluation frequency
"""

import os
import sys
import torch
import torch.nn as nn
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append("/app/ch04/01_main-chapter-code")

from models.titan_gpt_v3 import TitanGPTModelV3
from configs.model_configs_v3 import get_model_config_v3, get_training_config_v3
from gpt import create_dataloader_v1

print("Starting V3 Small Training - Memory Efficient Mode")
print("="*80)

# Setup
device = torch.device("cpu")  # Force CPU to avoid CUDA memory issues
torch.manual_seed(123)

# Load data
data_path = "/app/ch05/01_main-chapter-code/the-verdict.txt"
if not os.path.exists(data_path):
    print("Downloading training data...")
    import requests
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    response = requests.get(url, timeout=30)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(response.text)

with open(data_path, "r", encoding="utf-8") as f:
    text_data = f.read()

print(f"✅ Loaded {len(text_data)} characters")

# Split data
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

# Memory-efficient config
train_cfg = {
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "batch_size": 1,  # Minimal batch size
    "num_epochs": 5,  # Reduced epochs for memory
    "eval_freq": 100,
    "eval_iter": 5,
    "gradient_accumulation_steps": 16,  # High accumulation
    "diversity_loss_weight": 0.01,
}

print(f"Config: batch_size={train_cfg['batch_size']}, "
      f"epochs={train_cfg['num_epochs']}, "
      f"grad_accum={train_cfg['gradient_accumulation_steps']}")

# Create dataloaders
context_length = 256  # Reduced context for memory
train_loader = create_dataloader_v1(
    text_data[:split_idx],
    batch_size=train_cfg["batch_size"],
    max_length=context_length,
    stride=context_length,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    text_data[split_idx:],
    batch_size=train_cfg["batch_size"],
    max_length=context_length,
    stride=context_length,
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Create model with reduced memory config
v3_cfg = get_model_config_v3("small", use_memory=True)
v3_cfg["batch_size"] = train_cfg["batch_size"]
v3_cfg["context_length"] = context_length
# Reduce memory sizes for efficiency
v3_cfg["short_term_size"] = 64
v3_cfg["medium_term_size"] = 256
v3_cfg["long_term_size"] = 1024
v3_cfg["compression_ratio"] = 8  # Higher compression

print(f"\nCreating V3 Small Model (124M params)...")
print(f"Memory config: short={v3_cfg['short_term_size']}, "
      f"medium={v3_cfg['medium_term_size']}, "
      f"long={v3_cfg['long_term_size']}, "
      f"compression={v3_cfg['compression_ratio']}x")

model = TitanGPTModelV3(v3_cfg).to(device)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model created: {param_count:,} parameters")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg["learning_rate"],
    weight_decay=train_cfg["weight_decay"]
)

# Training loop
print(f"\n{'='*80}")
print("STARTING TRAINING")
print(f"{'='*80}\n")

model.train()
train_losses = []
val_losses = []
global_step = 0
tokens_seen = 0

for epoch in range(train_cfg["num_epochs"]):
    epoch_start = time.time()
    
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        # Forward pass
        logits, aux_losses = model(input_batch, targets=target_batch, update_memory=True, mode="train")
        
        # Main loss
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        
        # Add diversity loss
        if aux_losses:
            div_loss = sum(v for k, v in aux_losses.items() if 'diversity' in k and isinstance(v, torch.Tensor))
            if isinstance(div_loss, torch.Tensor) and div_loss.numel() > 0:
                loss = loss + train_cfg["diversity_loss_weight"] * div_loss
        
        # Backward
        loss = loss / train_cfg["gradient_accumulation_steps"]
        loss.backward()
        
        tokens_seen += input_batch.numel()
        
        # Update weights
        if (batch_idx + 1) % train_cfg["gradient_accumulation_steps"] == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Evaluation
            if global_step % train_cfg["eval_freq"] == 0:
                model.eval()
                
                # Train loss
                total_loss = 0
                with torch.no_grad():
                    for i, (inp, tgt) in enumerate(train_loader):
                        if i >= train_cfg["eval_iter"]:
                            break
                        inp, tgt = inp.to(device), tgt.to(device)
                        logits, _ = model(inp, update_memory=False, mode="inference")
                        total_loss += nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten()).item()
                train_loss = total_loss / min(train_cfg["eval_iter"], len(train_loader))
                
                # Val loss
                total_loss = 0
                with torch.no_grad():
                    for i, (inp, tgt) in enumerate(val_loader):
                        if i >= train_cfg["eval_iter"]:
                            break
                        inp, tgt = inp.to(device), tgt.to(device)
                        logits, _ = model(inp, update_memory=False, mode="inference")
                        total_loss += nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten()).item()
                val_loss = total_loss / min(train_cfg["eval_iter"], len(val_loader))
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                perplexity = torch.exp(torch.tensor(val_loss)).item()
                
                print(f"Epoch {epoch+1}/{train_cfg['num_epochs']} | "
                      f"Step {global_step} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"PPL: {perplexity:.2f}")
                
                model.train()
    
    epoch_time = time.time() - epoch_start
    print(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    # Reset memory
    model.reset_memory(level="short")

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"{'='*80}\n")

# Save model
os.makedirs("/app/titan_optimal/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "/app/titan_optimal/checkpoints/titan_v3_small.pth")
print("✅ Model saved to: /app/titan_optimal/checkpoints/titan_v3_small.pth")

# Save results
results = {
    "final_train_loss": train_losses[-1] if train_losses else None,
    "final_val_loss": val_losses[-1] if val_losses else None,
    "final_perplexity": torch.exp(torch.tensor(val_losses[-1])).item() if val_losses else None,
    "total_params": param_count,
    "epochs": train_cfg["num_epochs"],
    "batch_size": train_cfg["batch_size"],
    "context_length": context_length,
}

with open("/app/titan_optimal/v3_small_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Results saved to: /app/titan_optimal/v3_small_results.json")

print(f"\nFinal Results:")
print(f"  Train Loss: {results['final_train_loss']:.4f}")
print(f"  Val Loss: {results['final_val_loss']:.4f}")
print(f"  Perplexity: {results['final_perplexity']:.2f}")

print(f"\n⏱️  Sleeping for 900 seconds...")
time.sleep(900)

print(f"\n{'='*80}")
print("CHECKING LOGS AFTER SLEEP")
print(f"{'='*80}\n")

# Check logs
log_files = [
    "/var/log/supervisor/backend.err.log",
    "/var/log/supervisor/backend.out.log",
]

for log_file in log_files:
    if os.path.exists(log_file):
        print(f"\n📄 {log_file}:")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                for line in lines[-10:]:
                    print(line.rstrip())
            else:
                print("  (empty)")
    else:
        print(f"📄 {log_file}: Not found")

print(f"\n{'='*80}")
print("🎉 TRAINING AND EVALUATION COMPLETE!")
print(f"{'='*80}")
print("\nOutputs:")
print("  - Model: /app/titan_optimal/checkpoints/titan_v3_small.pth")
print("  - Results: /app/titan_optimal/v3_small_results.json")
print(f"{'='*80}")
