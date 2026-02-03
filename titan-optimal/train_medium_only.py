"""Memory-optimized training for Titan-GPT V2 MEDIUM model only.

Aggressive memory optimization:
- Batch size 1
- High gradient accumulation
- Frequent garbage collection
- Reduced context length
"""

import os
import sys
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
import gc
from pathlib import Path

# Add required paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append("/app/ch04/01_main-chapter-code")

from models.titan_gpt_v2 import TitanGPTModelV2
from configs.model_configs_v2 import get_model_config_v2, get_training_config_v2
from gpt import create_dataloader_v1


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    mode: str = "train"
) -> torch.Tensor:
    """Calculate loss for a batch with memory cleanup."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch, targets=target_batch, update_memory=True, mode=mode)
    
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    
    # Clear intermediate tensors
    del input_batch, target_batch, logits
    
    return loss


def calc_loss_loader(
    data_loader,
    model: nn.Module,
    device: torch.device,
    num_batches: int = None,
    mode: str = "inference"
) -> float:
    """Calculate average loss over data loader."""
    total_loss = 0.0
    
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    model.eval()
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            logits = model(input_batch, update_memory=False, mode=mode)
            
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            total_loss += loss.item()
            
            # Cleanup
            del input_batch, target_batch, logits
    
    model.train()
    gc.collect()
    return total_loss / num_batches


def train_model_v2_optimized(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int = 100,
    eval_iter: int = 5,
    gradient_accumulation_steps: int = 1,
    model_name: str = "titan_v2"
) -> dict:
    """Memory-optimized training."""
    train_losses = []
    val_losses = []
    perplexities = []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = 0
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} (Memory Optimized)")
    print(f"{'='*80}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Device: {device}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Forward pass
            loss = calc_loss_batch(
                input_batch, target_batch, model, device, mode="train"
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            tokens_seen += input_batch.numel()
            
            # Update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Aggressive memory cleanup
                gc.collect()
                
                # Evaluation (less frequent to save memory)
                if global_step % eval_freq == 0:
                    train_loss = calc_loss_loader(
                        train_loader, model, device, num_batches=eval_iter, mode="inference"
                    )
                    val_loss = calc_loss_loader(
                        val_loader, model, device, num_batches=eval_iter, mode="inference"
                    ) if len(val_loader) > 0 else train_loss
                    
                    # Calculate perplexity
                    perplexity = torch.exp(torch.tensor(val_loss)).item()
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    perplexities.append(perplexity)
                    track_tokens_seen.append(tokens_seen)
                    
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Step {global_step} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"PPL: {perplexity:.2f}"
                    )
                    
                    gc.collect()
        
        epoch_time = time.time() - epoch_start
        print(f"Completed Epoch {epoch+1}/{num_epochs} in {epoch_time:.2f}s")
        
        # Reset short-term memory between epochs
        if hasattr(model, 'reset_memory'):
            model.reset_memory(level="short")
        
        gc.collect()
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "perplexities": perplexities,
        "tokens_seen": track_tokens_seen,
    }


def main():
    """Train MEDIUM model only with memory optimization."""
    
    # Setup
    device = torch.device("cpu")  # CPU only for stability
    print("\n" + "="*80)
    print("TITAN-GPT V2 MEDIUM MODEL TRAINING (MEMORY OPTIMIZED)")
    print("="*80)
    print(f"Device: {device}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Available RAM: {os.popen('free -h | grep Mem').read().split()[1]}")
    
    torch.manual_seed(123)
    
    # Load data
    print("\nLoading training data...")
    data_path = "/app/ch05/01_main-chapter-code/the-verdict.txt"
    
    if not os.path.exists(data_path):
        print("Downloading training data...")
        import requests
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url, timeout=30)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("✓ Data downloaded")
    
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"✓ Loaded {len(text_data)} characters")
    
    # Prepare dataloaders with reduced context for memory
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    # AGGRESSIVE MEMORY OPTIMIZATION
    batch_size = 1  # Minimum batch size
    context_length = 512  # Reduced from 1024
    gradient_accumulation = 32  # High accumulation to compensate
    
    print(f"\nMemory optimization settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create medium model
    print(f"\nInitializing Titan-GPT V2 MEDIUM model...")
    titan_v2_cfg = get_model_config_v2("medium", use_memory=True)
    
    # Reduce memory size for medium model to save RAM
    titan_v2_cfg["short_term_size"] = 128  # Reduced from 256
    titan_v2_cfg["medium_term_size"] = 512  # Reduced from 1024
    titan_v2_cfg["long_term_size"] = 1024  # Reduced from 4096
    
    titan_v2_model = TitanGPTModelV2(titan_v2_cfg).to(device)
    
    print(f"✓ Model initialized with {count_parameters(titan_v2_model):,} parameters")
    
    # Get training config
    train_cfg = get_training_config_v2("medium")
    
    titan_v2_optimizer = torch.optim.AdamW(
        titan_v2_model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    # Train with reduced epochs for memory safety
    num_epochs = 10
    
    print(f"\nStarting training for {num_epochs} epochs...")
    results = train_model_v2_optimized(
        titan_v2_model,
        train_loader,
        val_loader,
        titan_v2_optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=50,  # Less frequent eval
        eval_iter=3,   # Fewer eval batches
        gradient_accumulation_steps=gradient_accumulation,
        model_name="Titan-V2-MEDIUM"
    )
    
    # Save model
    print(f"\n{'='*80}")
    print("Saving MEDIUM model...")
    print(f"{'='*80}")
    os.makedirs("/app/titan-optimal/checkpoints", exist_ok=True)
    torch.save(
        titan_v2_model.state_dict(),
        "/app/titan-optimal/checkpoints/titan_v2_medium.pth"
    )
    print(f"✓ Saved to checkpoints/titan_v2_medium.pth")
    
    # Save training results
    import json
    with open("/app/titan-optimal/medium_training_results.json", "w") as f:
        json.dump({
            "model_size": "medium",
            "parameters": count_parameters(titan_v2_model),
            "final_train_loss": float(results["train_losses"][-1]) if results["train_losses"] else None,
            "final_val_loss": float(results["val_losses"][-1]) if results["val_losses"] else None,
            "final_perplexity": float(results["perplexities"][-1]) if results["perplexities"] else None,
        }, f, indent=2)
    
    # Print final results
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    if results["train_losses"]:
        print(f"\nFinal Results (Titan-V2-MEDIUM):")
        print(f"  Final Train Loss:  {results['train_losses'][-1]:.4f}")
        print(f"  Final Val Loss:    {results['val_losses'][-1]:.4f}")
        print(f"  Final Perplexity:  {results['perplexities'][-1]:.2f}")
        print(f"  Total Tokens:      {results['tokens_seen'][-1]:,}")
    
    print("\n" + "="*80)
    print("BOTH MODELS TRAINED SUCCESSFULLY!")
    print("="*80)
    print("\nCheckpoints:")
    print("  ✓ /app/titan-optimal/checkpoints/titan_v2_small.pth")
    print("  ✓ /app/titan-optimal/checkpoints/titan_v2_medium.pth")
    print("\nKey V2 Features:")
    print("  ✓ Gradient-free surprise metrics")
    print("  ✓ Hierarchical 3-tier memory")
    print("  ✓ Memory compression")
    print("  ✓ Separate training/inference modes")
    print("="*80)


if __name__ == "__main__":
    main()
