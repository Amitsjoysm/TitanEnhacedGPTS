"""Training Pipeline for Titan-GPT V3 Small Model Only

🚀 Production-Ready Training with ALL Improvements:
- V3 Small (124M) model training
- All 10 V3 enhancements
- Dynamic batch size handling
- Memory budget management
- Auxiliary loss scaling
- Episodic boundary detection
- 900 second sleep then log checking
"""

import os
import sys
import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add required paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append("/app/ch04/01_main-chapter-code")

from models.titan_gpt_v3 import TitanGPTModelV3
from configs.model_configs_v3 import (
    get_model_config_v3,
    get_training_config_v3,
)

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
    mode: str = "train",
    diversity_weight: float = 0.01
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculate loss for a batch with auxiliary losses."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    metrics = {}
    
    logits, aux_losses = model(
        input_batch,
        targets=target_batch,
        update_memory=True,
        mode=mode
    )
    
    # Add auxiliary losses
    for k, v in aux_losses.items():
        if 'diversity' in k:
            metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
    
    # Main loss
    main_loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    
    # Total loss with diversity regularization
    total_loss = main_loss
    if aux_losses:
        diversity_loss = sum(v for k, v in aux_losses.items() if 'diversity' in k)
        if isinstance(diversity_loss, torch.Tensor) and diversity_loss.numel() > 0:
            total_loss = total_loss + diversity_weight * diversity_loss
            metrics['total_diversity_loss'] = diversity_loss.item()
    
    metrics['main_loss'] = main_loss.item()
    
    return total_loss, metrics


def calc_loss_loader(
    data_loader,
    model: nn.Module,
    device: torch.device,
    num_batches: int = None,
    mode: str = "inference"
) -> Tuple[float, Dict[str, float]]:
    """Calculate average loss over data loader."""
    total_loss = 0.0
    all_metrics = {}
    
    if len(data_loader) == 0:
        return float("nan"), {}
    
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
            
            logits, aux_losses = model(
                input_batch,
                update_memory=False,
                mode=mode
            )
            
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            total_loss += loss.item()
    
    model.train()
    avg_loss = total_loss / num_batches
    
    return avg_loss, all_metrics


def train_model_v3(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int = 50,
    eval_iter: int = 10,
    gradient_accumulation_steps: int = 1,
    model_name: str = "titan_v3_small",
    diversity_weight: float = 0.01
) -> Dict:
    """Train model with comprehensive evaluation."""
    train_losses = []
    val_losses = []
    perplexities = []
    track_tokens_seen = []
    inference_times = []
    diversity_losses = []
    memory_pressure_log = []
    tokens_seen = 0
    global_step = 0
    
    print(f"\n{'='*80}")
    print(f"Training {model_name}...")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model type: V3 with ALL enhancements")
    print(f"{'='*80}\n")
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_diversity_loss = 0.0
        epoch_steps = 0
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # Forward pass with auxiliary losses
            loss, metrics = calc_loss_batch(
                input_batch, target_batch, model, device,
                mode="train",
                diversity_weight=diversity_weight
            )
            
            # Track diversity loss
            if 'total_diversity_loss' in metrics:
                epoch_diversity_loss += metrics['total_diversity_loss']
                epoch_steps += 1
            
            # Normalize for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            tokens_seen += input_batch.numel()
            
            # Update weights
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Evaluation
                if global_step % eval_freq == 0:
                    train_loss, _ = calc_loss_loader(
                        train_loader, model, device, num_batches=eval_iter,
                        mode="inference"
                    )
                    val_loss, _ = calc_loss_loader(
                        val_loader, model, device, num_batches=eval_iter,
                        mode="inference"
                    )
                    
                    perplexity = torch.exp(torch.tensor(val_loss)).item()
                    
                    # Measure inference speed
                    start_time = time.time()
                    with torch.no_grad():
                        _ = calc_loss_loader(
                            val_loader, model, device, num_batches=5,
                            mode="inference"
                        )
                    inference_time = time.time() - start_time
                    
                    # Get memory pressure
                    memory_pressure = {}
                    for block in model.trf_blocks:
                        if hasattr(block, 'neural_memory'):
                            pressure = block.neural_memory.hierarchical_memory.get_memory_pressure()
                            memory_pressure = pressure
                            break
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    perplexities.append(perplexity)
                    track_tokens_seen.append(tokens_seen)
                    inference_times.append(inference_time)
                    memory_pressure_log.append(memory_pressure)
                    
                    # Average diversity loss
                    if epoch_steps > 0:
                        avg_div_loss = epoch_diversity_loss / epoch_steps
                        diversity_losses.append(avg_div_loss)
                    
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Step {global_step} | "
                        f"Train: {train_loss:.4f} | "
                        f"Val: {val_loss:.4f} | "
                        f"PPL: {perplexity:.2f} | "
                        f"Inf: {inference_time:.2f}s | "
                        f"Mem: {memory_pressure.get('overall', 0.0):.2%}"
                    )
        
        epoch_time = time.time() - epoch_start
        print(f"Completed Epoch {epoch+1}/{num_epochs} in {epoch_time:.2f}s")
        
        # Reset short-term memory between epochs
        if hasattr(model, 'reset_memory'):
            model.reset_memory(level="short")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "perplexities": perplexities,
        "tokens_seen": track_tokens_seen,
        "inference_times": inference_times,
        "diversity_losses": diversity_losses,
        "memory_pressure": memory_pressure_log,
    }


def plot_training_results(results: Dict, save_path: str = None):
    """Plot comprehensive training results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    steps = range(len(results["train_losses"]))
    ax1.plot(steps, results["train_losses"], label='Train Loss', marker='o', markersize=3)
    ax1.set_xlabel("Evaluation Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    ax2.plot(steps, results["val_losses"], label='Val Loss', marker='o', markersize=4, color='orange')
    ax2.set_xlabel("Evaluation Step")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Perplexity
    ax3 = axes[0, 2]
    ax3.plot(steps, results["perplexities"], label='Perplexity', marker='o', markersize=4, color='green')
    ax3.set_xlabel("Evaluation Step")
    ax3.set_ylabel("Perplexity")
    ax3.set_title("Perplexity (Lower is Better)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Inference Time
    ax4 = axes[1, 0]
    ax4.plot(steps, results["inference_times"], label='Inference Time', marker='o', markersize=4, color='red')
    ax4.set_xlabel("Evaluation Step")
    ax4.set_ylabel("Time (seconds)")
    ax4.set_title("Inference Speed")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Diversity Loss
    ax5 = axes[1, 1]
    if results.get("diversity_losses"):
        div_steps = range(len(results["diversity_losses"]))
        ax5.plot(div_steps, results["diversity_losses"], label='Diversity Loss', marker='o', markersize=4, color='purple')
    ax5.set_xlabel("Evaluation Step")
    ax5.set_ylabel("Diversity Loss")
    ax5.set_title("Memory Diversity Loss")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Memory Pressure
    ax6 = axes[1, 2]
    if results.get("memory_pressure"):
        pressures = [p.get('overall', 0.0) for p in results["memory_pressure"]]
        ax6.plot(range(len(pressures)), pressures, label='Memory Pressure', marker='o', markersize=4, color='brown')
    ax6.set_xlabel("Evaluation Step")
    ax6.set_ylabel("Memory Pressure")
    ax6.set_title("Memory Budget Usage")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.savefig("/app/titan_optimal/training_v3_small.png", dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: /app/titan_optimal/training_v3_small.png")


def main():
    """Main V3 Small training pipeline."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("TITAN-GPT V3 SMALL - PRODUCTION TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"CPU cores: {os.cpu_count()}")
    print("\n🚀 ALL Improvements Implemented:")
    print("  ✅ Batch Size Flexibility (dynamic resize)")
    print("  ✅ Auxiliary Loss Scaling (by n_layers)")
    print("  ✅ Memory Budget Management (pressure monitoring)")
    print("  ✅ Episodic Boundary Detection (working)")
    print("\n🎯 V3 Enhancements:")
    print("  1. GPU Tensor Buffers")
    print("  2. Surprise-Driven Consolidation")
    print("  3. Early Stopping Retrieval")
    print("  4. Integrated Compression")
    print("  5. Per-Batch Buffers")
    print("  6. Adaptive Schedule")
    print("  7. Importance Decay")
    print("  8. Cross-Attention")
    print("  9. Diversity Loss")
    print("  10. Episodic Boundaries")
    print("="*80)
    
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    
    # Load data
    print("\nLoading training data...")
    data_path = "/app/ch05/01_main-chapter-code/the-verdict.txt"
    
    if not os.path.exists(data_path):
        import requests
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url, timeout=30)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"Loaded {len(text_data)} characters")
    
    # Split data
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    # Get configs for small model
    train_cfg = get_training_config_v3("small")
    
    # Adjust for CPU if needed
    if device.type == "cpu":
        train_cfg["batch_size"] = max(1, train_cfg["batch_size"] // 2)
        train_cfg["gradient_accumulation_steps"] *= 2
        print(f"CPU mode: batch_size={train_cfg['batch_size']}, "
              f"grad_accum={train_cfg['gradient_accumulation_steps']}")
    
    # Create dataloaders
    context_length = 512
    
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
    
    # Train V3 Small Model
    print("\n" + "="*80)
    print("Training Titan-GPT V3 SMALL (124M parameters)")
    print("="*80)
    
    v3_cfg = get_model_config_v3("small", use_memory=True)
    v3_cfg["batch_size"] = train_cfg["batch_size"]
    v3_model = TitanGPTModelV3(v3_cfg).to(device)
    v3_optimizer = torch.optim.AdamW(
        v3_model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    # Train model
    results = train_model_v3(
        v3_model, train_loader, val_loader, v3_optimizer,
        device, train_cfg["num_epochs"], train_cfg["eval_freq"],
        train_cfg["eval_iter"], train_cfg["gradient_accumulation_steps"],
        "Titan-V3-Small", train_cfg["diversity_loss_weight"]
    )
    
    # Save model
    print(f"\nSaving model...")
    os.makedirs("/app/titan_optimal/checkpoints", exist_ok=True)
    torch.save(v3_model.state_dict(), "/app/titan_optimal/checkpoints/titan_v3_small.pth")
    print(f"✅ Model saved to: /app/titan_optimal/checkpoints/titan_v3_small.pth")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_results(results)
    
    # Save results to JSON
    print("\nSaving results...")
    results_summary = {
        "final_train_loss": results["train_losses"][-1] if results["train_losses"] else None,
        "final_val_loss": results["val_losses"][-1] if results["val_losses"] else None,
        "final_perplexity": results["perplexities"][-1] if results["perplexities"] else None,
        "avg_inference_time": sum(results["inference_times"]) / len(results["inference_times"]) if results["inference_times"] else None,
        "final_memory_pressure": results["memory_pressure"][-1] if results["memory_pressure"] else None,
        "has_diversity_loss": len(results.get("diversity_losses", [])) > 0,
        "total_params": count_parameters(v3_model)
    }
    
    with open("/app/titan_optimal/v3_small_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print("✅ Results saved to: /app/titan_optimal/v3_small_results.json")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL RESULTS")
    print("="*80)
    print(f"Model: Titan-GPT V3 Small (124M parameters)")
    print(f"Train Loss: {results['train_losses'][-1]:.4f}")
    print(f"Val Loss: {results['val_losses'][-1]:.4f}")
    print(f"Perplexity: {results['perplexities'][-1]:.2f}")
    print(f"Avg Inference: {sum(results['inference_times']) / len(results['inference_times']):.2f}s")
    if results['memory_pressure']:
        final_pressure = results['memory_pressure'][-1]
        print(f"Final Memory Pressure:")
        print(f"  - Short-term: {final_pressure.get('short_term', 0.0):.2%}")
        print(f"  - Medium-term: {final_pressure.get('medium_term', 0.0):.2%}")
        print(f"  - Long-term: {final_pressure.get('long_term', 0.0):.2%}")
        print(f"  - Overall: {final_pressure.get('overall', 0.0):.2%}")
    print("="*80)
    
    print("\n⏱️  Sleeping for 900 seconds before checking logs...")
    time.sleep(900)
    
    print("\n" + "="*80)
    print("CHECKING LOGS AFTER 900 SECOND SLEEP")
    print("="*80)
    
    # Check if there are any supervisor logs
    log_files = [
        "/var/log/supervisor/backend.err.log",
        "/var/log/supervisor/backend.out.log",
        "/var/log/supervisor/frontend.err.log",
        "/var/log/supervisor/frontend.out.log"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n📄 {log_file}:")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("Last 20 lines:")
                    for line in lines[-20:]:
                        print(line.rstrip())
                else:
                    print("  (empty)")
        else:
            print(f"\n📄 {log_file}: Not found")
    
    print("\n" + "="*80)
    print("🎉 ALL TASKS COMPLETE!")
    print("="*80)
    print("\n✅ V3 Small Model Trained")
    print("✅ All Improvements Implemented")
    print("✅ Evaluation Complete")
    print("✅ Logs Checked")
    print("\nOutputs:")
    print("  - Model: /app/titan_optimal/checkpoints/titan_v3_small.pth")
    print("  - Plot: /app/titan_optimal/training_v3_small.png")
    print("  - Results: /app/titan_optimal/v3_small_results.json")
    print("="*80)


if __name__ == "__main__":
    main()
