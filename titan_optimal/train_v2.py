"""Enhanced training script for Titan-GPT v2 models.

Implements comprehensive training and evaluation with:
1. Gradient-free surprise metrics
2. Hierarchical memory system
3. Curriculum learning
4. Comprehensive evaluation metrics
5. Both Small (124M) and Medium (340M) model training
"""

import os
import sys
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Add required paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append("/app/ch04/01_main-chapter-code")

from models.titan_gpt_v2 import TitanGPTModelV2
from models.titan_gpt import TitanGPTModel  # Original for comparison
from configs.model_configs_v2 import get_model_config_v2, get_training_config_v2
from configs.model_configs import get_model_config

# Import from existing codebase
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
    """Calculate loss for a batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Check if model is v2
    is_v2 = hasattr(model, 'set_mode')
    
    if is_v2:
        logits = model(input_batch, targets=target_batch, update_memory=True, mode=mode)
    else:
        logits = model(input_batch, targets=target_batch, update_memory=True)
    
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    
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
    is_v2 = hasattr(model, 'set_mode')
    
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            if is_v2:
                logits = model(input_batch, update_memory=False, mode=mode)
            else:
                logits = model(input_batch, update_memory=False)
            
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def train_model_v2(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int = 50,
    eval_iter: int = 10,
    gradient_accumulation_steps: int = 1,
    model_name: str = "titan_v2",
    is_v2: bool = True
) -> dict:
    """Train model with enhanced evaluation."""
    train_losses = []
    val_losses = []
    perplexities = []
    track_tokens_seen = []
    inference_times = []
    tokens_seen = 0
    global_step = 0
    
    print(f"\nTraining {model_name}...")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model type: {'V2 (Enhanced)' if is_v2 else 'V1 (Original)'}")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Forward pass
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                mode="train" if is_v2 else None
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
                
                # Evaluation
                if global_step % eval_freq == 0:
                    train_loss = calc_loss_loader(
                        train_loader, model, device, num_batches=eval_iter,
                        mode="inference" if is_v2 else None
                    )
                    val_loss = calc_loss_loader(
                        val_loader, model, device, num_batches=eval_iter,
                        mode="inference" if is_v2 else None
                    )
                    
                    # Calculate perplexity
                    perplexity = torch.exp(torch.tensor(val_loss)).item()
                    
                    # Measure inference speed
                    start_time = time.time()
                    with torch.no_grad():
                        _ = calc_loss_loader(
                            val_loader, model, device, num_batches=5,
                            mode="inference" if is_v2 else None
                        )
                    inference_time = time.time() - start_time
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    perplexities.append(perplexity)
                    track_tokens_seen.append(tokens_seen)
                    inference_times.append(inference_time)
                    
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Step {global_step} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Perplexity: {perplexity:.2f} | "
                        f"Inference Time: {inference_time:.2f}s"
                    )
        
        epoch_time = time.time() - epoch_start
        print(f"Completed Epoch {epoch+1}/{num_epochs} in {epoch_time:.2f}s")
        
        # Reset short-term memory between epochs for v2 models
        if is_v2 and hasattr(model, 'reset_memory'):
            model.reset_memory(level="short")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "perplexities": perplexities,
        "tokens_seen": track_tokens_seen,
        "inference_times": inference_times,
    }


def plot_comprehensive_comparison(results: dict, save_path: str = None):
    """Plot comprehensive training comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training and Validation Loss
    ax1 = axes[0, 0]
    for model_name, data in results.items():
        steps = range(len(data["train_losses"]))
        ax1.plot(steps, data["train_losses"], label=f"{model_name} (train)", marker='o', markersize=3)
        ax1.plot(steps, data["val_losses"], label=f"{model_name} (val)", marker='s', markersize=3, linestyle='--')
    ax1.set_xlabel("Evaluation Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss Only (clearer)
    ax2 = axes[0, 1]
    for model_name, data in results.items():
        steps = range(len(data["val_losses"]))
        ax2.plot(steps, data["val_losses"], label=model_name, marker='o', markersize=4)
    ax2.set_xlabel("Evaluation Step")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Comparison")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Perplexity
    ax3 = axes[1, 0]
    for model_name, data in results.items():
        if "perplexities" in data and len(data["perplexities"]) > 0:
            steps = range(len(data["perplexities"]))
            ax3.plot(steps, data["perplexities"], label=model_name, marker='o', markersize=4)
    ax3.set_xlabel("Evaluation Step")
    ax3.set_ylabel("Perplexity")
    ax3.set_title("Perplexity Comparison (Lower is Better)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Inference Time
    ax4 = axes[1, 1]
    for model_name, data in results.items():
        if "inference_times" in data and len(data["inference_times"]) > 0:
            steps = range(len(data["inference_times"]))
            ax4.plot(steps, data["inference_times"], label=model_name, marker='o', markersize=4)
    ax4.set_xlabel("Evaluation Step")
    ax4.set_ylabel("Inference Time (seconds)")
    ax4.set_title("Inference Speed Comparison (Lower is Better)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.savefig("/app/titan_optimal/training_comparison_v2.png", dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: /app/titan_optimal/training_comparison_v2.png")


def main():
    """Main training pipeline for v2 models."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("TITAN-GPT V2 ENHANCED TRAINING PIPELINE")
    print("="*80)
    print(f"Using device: {device}")
    print(f"CPU cores available: {os.cpu_count()}")
    
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    
    # Load data
    print("\nLoading training data...")
    data_path = "/app/ch05/01_main-chapter-code/the-verdict.txt"
    
    if not os.path.exists(data_path):
        # Download if not exists
        import requests
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url, timeout=30)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"Loaded {len(text_data)} characters")
    
    # Prepare dataloaders
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    # Train multiple model sizes and variants
    results = {}
    model_sizes = ["small", "medium"]  # Both as requested
    
    for size in model_sizes:
        print("\n" + "="*80)
        print(f"TRAINING {size.upper()} MODELS (124M params)" if size == "small" else f"TRAINING {size.upper()} MODELS (340M params)")
        print("="*80)
        
        # Get configs
        train_cfg = get_training_config_v2(size)
        
        # Adjust batch size for CPU if needed
        if device.type == "cpu":
            train_cfg["batch_size"] = max(1, train_cfg["batch_size"] // 2)
            train_cfg["gradient_accumulation_steps"] *= 2
            print(f"CPU mode: Adjusted batch_size to {train_cfg['batch_size']}, gradient_accumulation to {train_cfg['gradient_accumulation_steps']}")
        
        # Create dataloaders
        context_length = 512 if size == "small" else 1024
        
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
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # 1. Baseline GPT (no memory) - V1 architecture
        print("\n" + "-"*80)
        print(f"Training Baseline GPT {size.upper()} (no memory - V1)")
        print("-"*80)
        
        baseline_cfg = get_model_config(size, use_memory=False)
        baseline_model = TitanGPTModel(baseline_cfg).to(device)
        baseline_optimizer = torch.optim.AdamW(
            baseline_model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )
        
        baseline_results = train_model_v2(
            baseline_model,
            train_loader,
            val_loader,
            baseline_optimizer,
            device,
            num_epochs=train_cfg["num_epochs"],
            eval_freq=train_cfg["eval_freq"],
            eval_iter=train_cfg["eval_iter"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            model_name=f"Baseline-{size}",
            is_v2=False
        )
        results[f"Baseline-{size}"] = baseline_results
        
        # 2. Titan-GPT V1 (original memory) - For comparison
        print("\n" + "-"*80)
        print(f"Training Titan-GPT V1 {size.upper()} (original memory)")
        print("-"*80)
        
        titan_v1_cfg = get_model_config(size, use_memory=True)
        titan_v1_model = TitanGPTModel(titan_v1_cfg).to(device)
        titan_v1_optimizer = torch.optim.AdamW(
            titan_v1_model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )
        
        titan_v1_results = train_model_v2(
            titan_v1_model,
            train_loader,
            val_loader,
            titan_v1_optimizer,
            device,
            num_epochs=train_cfg["num_epochs"],
            eval_freq=train_cfg["eval_freq"],
            eval_iter=train_cfg["eval_iter"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            model_name=f"Titan-V1-{size}",
            is_v2=False
        )
        results[f"Titan-V1-{size}"] = titan_v1_results
        
        # 3. Titan-GPT V2 (enhanced memory)
        print("\n" + "-"*80)
        print(f"Training Titan-GPT V2 {size.upper()} (enhanced memory)")
        print("-"*80)
        
        titan_v2_cfg = get_model_config_v2(size, use_memory=True)
        titan_v2_model = TitanGPTModelV2(titan_v2_cfg).to(device)
        titan_v2_optimizer = torch.optim.AdamW(
            titan_v2_model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"]
        )
        
        titan_v2_results = train_model_v2(
            titan_v2_model,
            train_loader,
            val_loader,
            titan_v2_optimizer,
            device,
            num_epochs=train_cfg["num_epochs"],
            eval_freq=train_cfg["eval_freq"],
            eval_iter=train_cfg["eval_iter"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            model_name=f"Titan-V2-{size}",
            is_v2=True
        )
        results[f"Titan-V2-{size}"] = titan_v2_results
        
        # Save models for this size
        print(f"\nSaving {size} models...")
        os.makedirs("/app/titan_optimal/checkpoints", exist_ok=True)
        torch.save(baseline_model.state_dict(), f"/app/titan_optimal/checkpoints/baseline_{size}_v2.pth")
        torch.save(titan_v1_model.state_dict(), f"/app/titan_optimal/checkpoints/titan_v1_{size}.pth")
        torch.save(titan_v2_model.state_dict(), f"/app/titan_optimal/checkpoints/titan_v2_{size}.pth")
        print(f"Saved {size} models to checkpoints/")
    
    # Plot comprehensive results
    print("\n" + "="*80)
    print("Plotting comprehensive results...")
    print("="*80)
    plot_comprehensive_comparison(results)
    
    # Print final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    
    for model_name, data in results.items():
        final_train_loss = data["train_losses"][-1] if data["train_losses"] else float('nan')
        final_val_loss = data["val_losses"][-1] if data["val_losses"] else float('nan')
        final_perplexity = data["perplexities"][-1] if data["perplexities"] else float('nan')
        avg_inference_time = sum(data["inference_times"]) / len(data["inference_times"]) if data["inference_times"] else float('nan')
        
        print(f"\n{model_name}:")
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Final Perplexity: {final_perplexity:.2f}")
        print(f"  Avg Inference Time: {avg_inference_time:.2f}s")
    
    # Calculate improvements
    print("\n" + "="*80)
    print("V2 IMPROVEMENTS vs BASELINE")
    print("="*80)
    
    for size in model_sizes:
        baseline_key = f"Baseline-{size}"
        v2_key = f"Titan-V2-{size}"
        
        if baseline_key in results and v2_key in results:
            baseline_ppl = results[baseline_key]["perplexities"][-1] if results[baseline_key]["perplexities"] else float('nan')
            v2_ppl = results[v2_key]["perplexities"][-1] if results[v2_key]["perplexities"] else float('nan')
            
            improvement = ((baseline_ppl - v2_ppl) / baseline_ppl) * 100 if baseline_ppl != 0 else 0
            
            print(f"\n{size.upper()} Model:")
            print(f"  Baseline Perplexity: {baseline_ppl:.2f}")
            print(f"  Titan-V2 Perplexity: {v2_ppl:.2f}")
            print(f"  Improvement: {improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nKey Features Implemented:")
    print("✓ Gradient-free surprise metrics (entropy, rarity, attention)")
    print("✓ Hierarchical 3-tier memory (short/medium/long-term)")
    print("✓ Memory compression with clustering")
    print("✓ Separate training/inference modes")
    print("✓ Both Small (124M) and Medium (340M) model sizes")
    print("✓ Comprehensive evaluation (loss, perplexity, inference speed)")
    print("\nCheckpoints saved to: /app/titan_optimal/checkpoints/")
    print("Plots saved to: /app/titan_optimal/training_comparison_v2.png")
    print("="*80)


if __name__ == "__main__":
    main()
