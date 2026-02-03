"""Main training script for Titan-Optimal models.

Integrates:
1. Titan architecture with neural long-term memory
2. Compute-optimal sampling strategies
3. Both model sizes (124M and 340M)
"""

import os
import sys
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.titan_gpt import TitanGPTModel
from configs.model_configs import (
    get_model_config,
    get_training_config,
    get_strategy_config,
    BASELINE_GPT_124M
)
from training.compute_optimal import (
    SyntheticDataGenerator,
    DataQualityEvaluator,
    ComputeOptimalTrainer
)

# Import from existing codebase
sys.path.append("/app/ch04/01_main_chapter_code")
from gpt import GPTDatasetV1, create_dataloader_v1


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """Calculate loss for a batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
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
    num_batches: int = None
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
            
            # Temporarily disable memory updates during evaluation
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch, update_memory=False)
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int = 50,
    eval_iter: int = 10,
    gradient_accumulation_steps: int = 1,
    model_name: str = "titan"
) -> dict:
    """Train model with evaluation."""
    train_losses = []
    val_losses = []
    track_tokens_seen = []
    tokens_seen = 0
    global_step = 0
    
    print(f"\nTraining {model_name}...")
    print(f"Total parameters: {count_parameters(model):,}")
    
    model.train()
    
    for epoch in range(num_epochs):
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Forward pass
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
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
                        train_loader, model, device, num_batches=eval_iter
                    )
                    val_loss = calc_loss_loader(
                        val_loader, model, device, num_batches=eval_iter
                    )
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Step {global_step} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f}"
                    )
        
        # End of epoch
        print(f"Completed Epoch {epoch+1}/{num_epochs}")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": track_tokens_seen,
    }


def plot_comparison(
    results: dict,
    save_path: str = None
):
    """Plot training curves for multiple models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training losses
    for model_name, data in results.items():
        steps = range(len(data["train_losses"]))
        ax1.plot(steps, data["train_losses"], label=f"{model_name} (train)", marker='o')
        ax1.plot(steps, data["val_losses"], label=f"{model_name} (val)", marker='s', linestyle='--')
    
    ax1.set_xlabel("Evaluation Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot validation losses only for clarity
    for model_name, data in results.items():
        steps = range(len(data["val_losses"]))
        ax2.plot(steps, data["val_losses"], label=model_name, marker='o')
    
    ax2.set_xlabel("Evaluation Step")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.savefig("/app/titan-optimal/training_comparison.png", dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: /app/titan-optimal/training_comparison.png")


def main():
    """Main training pipeline."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    
    # Load data
    print("\nLoading training data...")
    data_path = "/app/ch05/01_main_chapter_code/the-verdict.txt"
    
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
    
    # Get configs
    train_cfg = get_training_config("small")
    
    # Create dataloaders
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=train_cfg["batch_size"],
        max_length=512,
        stride=512,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=train_cfg["batch_size"],
        max_length=512,
        stride=512,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Train multiple models for comparison
    results = {}
    
    # 1. Baseline GPT (no memory)
    print("\n" + "="*60)
    print("Training Baseline GPT (no memory)")
    print("="*60)
    
    baseline_cfg = get_model_config("small", use_memory=False)
    baseline_model = TitanGPTModel(baseline_cfg).to(device)
    baseline_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    baseline_results = train_model(
        baseline_model,
        train_loader,
        val_loader,
        baseline_optimizer,
        device,
        num_epochs=train_cfg["num_epochs"],
        eval_freq=train_cfg["eval_freq"],
        eval_iter=train_cfg["eval_iter"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        model_name="Baseline GPT"
    )
    results["Baseline GPT"] = baseline_results
    
    # 2. Titan-GPT with MAC (Memory as Context)
    print("\n" + "="*60)
    print("Training Titan-GPT with MAC (Memory as Context)")
    print("="*60)
    
    titan_mac_cfg = get_model_config("small", use_memory=True)
    titan_mac_cfg["memory_variant"] = "mac"
    titan_mac_model = TitanGPTModel(titan_mac_cfg).to(device)
    titan_mac_optimizer = torch.optim.AdamW(
        titan_mac_model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    titan_mac_results = train_model(
        titan_mac_model,
        train_loader,
        val_loader,
        titan_mac_optimizer,
        device,
        num_epochs=train_cfg["num_epochs"],
        eval_freq=train_cfg["eval_freq"],
        eval_iter=train_cfg["eval_iter"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        model_name="Titan-MAC"
    )
    results["Titan-MAC"] = titan_mac_results
    
    # 3. Titan-GPT with MAG (Memory as Gate)
    print("\n" + "="*60)
    print("Training Titan-GPT with MAG (Memory as Gate)")
    print("="*60)
    
    titan_mag_cfg = get_model_config("small", use_memory=True)
    titan_mag_cfg["memory_variant"] = "mag"
    titan_mag_model = TitanGPTModel(titan_mag_cfg).to(device)
    titan_mag_optimizer = torch.optim.AdamW(
        titan_mag_model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    titan_mag_results = train_model(
        titan_mag_model,
        train_loader,
        val_loader,
        titan_mag_optimizer,
        device,
        num_epochs=train_cfg["num_epochs"],
        eval_freq=train_cfg["eval_freq"],
        eval_iter=train_cfg["eval_iter"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        model_name="Titan-MAG"
    )
    results["Titan-MAG"] = titan_mag_results
    
    # Save models
    print("\n" + "="*60)
    print("Saving models...")
    print("="*60)
    
    os.makedirs("/app/titan-optimal/checkpoints", exist_ok=True)
    torch.save(baseline_model.state_dict(), "/app/titan-optimal/checkpoints/baseline_gpt.pth")
    torch.save(titan_mac_model.state_dict(), "/app/titan-optimal/checkpoints/titan_mac.pth")
    torch.save(titan_mag_model.state_dict(), "/app/titan-optimal/checkpoints/titan_mag.pth")
    
    print("Models saved to /app/titan-optimal/checkpoints/")
    
    # Plot results
    print("\n" + "="*60)
    print("Plotting results...")
    print("="*60)
    plot_comparison(results)
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for model_name, data in results.items():
        final_train_loss = data["train_losses"][-1]
        final_val_loss = data["val_losses"][-1]
        print(f"\n{model_name}:")
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
