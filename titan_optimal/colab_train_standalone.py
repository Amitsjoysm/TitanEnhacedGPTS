"""
Standalone Google Colab Training Script for Titan-GPT V3
Can be run directly on Colab without the full repository

Usage in Colab:
1. Upload this file to Colab
2. Mount Google Drive
3. Run: python colab_train_standalone.py
"""

import os
import sys
import torch
import torch.nn as nn
import time
import json
from tqdm import tqdm

# Configuration
CONFIG = {
    "model_size": "small",  # "small" or "medium"
    "num_epochs": 10,
    "batch_size": 8,  # Adjust based on GPU memory
    "learning_rate": 5e-4,
    "weight_decay": 0.1,
    "gradient_accumulation_steps": 4,
    "save_every_n_epochs": 2,
    "context_length": 512,
    "eval_freq": 50,
    "checkpoint_dir": "/content/drive/MyDrive/Titan_V3_Checkpoints"
}

def setup_environment():
    """Setup Google Drive and verify GPU."""
    print("="*80)
    print("SETUP: Mounting Google Drive and Checking GPU")
    print("="*80)
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except:
        print("⚠️ Not running on Colab or Drive already mounted")
    
    # Create checkpoint directory
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    print(f"✅ Checkpoint directory: {CONFIG['checkpoint_dir']}")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ GPU not available, using CPU (will be slow!)")
    
    return device

def install_dependencies():
    """Install required packages."""
    print("\n" + "="*80)
    print("SETUP: Installing Dependencies")
    print("="*80)
    
    packages = ["torch", "tiktoken", "matplotlib", "tqdm"]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            os.system(f"pip install -q {package}")
            print(f"✅ {package} installed")

def clone_repo():
    """Clone the LLMs-from-scratch repository."""
    print("\n" + "="*80)
    print("SETUP: Cloning Repository")
    print("="*80)
    
    repo_path = "/content/LLMs-from-scratch"
    
    if not os.path.exists(repo_path):
        print("📥 Cloning repository...")
        os.system("git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git /content/LLMs-from-scratch")
        print("✅ Repository cloned")
    else:
        print("✅ Repository already exists")
    
    # Add to path
    sys.path.append(repo_path)
    sys.path.append(f"{repo_path}/ch04/01_main-chapter-code")
    sys.path.append(f"{repo_path}/titan_optimal")
    
    return repo_path

def download_training_data():
    """Download training data."""
    data_path = "/content/LLMs-from-scratch/ch05/01_main-chapter-code/the-verdict.txt"
    
    if not os.path.exists(data_path):
        print("📥 Downloading training data...")
        import requests
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        response = requests.get(url, timeout=30)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print("✅ Training data downloaded")
    
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    print(f"✅ Loaded {len(text_data)} characters")
    return text_data

def create_simple_dataloaders(text_data, batch_size, context_length):
    """Create simple dataloaders without external dependencies."""
    import tiktoken
    
    # Split data
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_text = text_data[:split_idx]
    val_text = text_data[split_idx:]
    
    # Tokenize
    tokenizer = tiktoken.get_encoding("gpt2")
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    
    print(f"✅ Train tokens: {len(train_tokens)}")
    print(f"✅ Val tokens: {len(val_tokens)}")
    
    # Import from repo
    from gpt import create_dataloader_v1
    
    train_loader = create_dataloader_v1(
        train_text,
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_text,
        batch_size=batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, optimizer, device, config):
    """Training loop."""
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80)
    
    train_losses = []
    val_losses = []
    perplexities = []
    best_val_loss = float('inf')
    
    model.train()
    
    for epoch in range(config["num_epochs"]):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{config['num_epochs']}")
        print(f"{'='*80}")
        
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Training with progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (input_batch, target_batch) in enumerate(pbar):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            try:
                logits, aux_losses = model(
                    input_batch,
                    targets=target_batch,
                    update_memory=True,
                    mode="train"
                )
            except:
                # Fallback for models without V3 features
                logits = model(input_batch)
                aux_losses = {}
            
            # Loss
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            
            # Backward
            loss = loss / config["gradient_accumulation_steps"]
            loss.backward()
            
            epoch_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)
                
                try:
                    logits, _ = model(
                        input_batch,
                        update_memory=False,
                        mode="inference"
                    )
                except:
                    logits = model(input_batch)
                
                val_loss += nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    target_batch.flatten()
                ).item()
        
        val_loss /= len(val_loader)
        train_loss = epoch_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        perplexities.append(perplexity)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % config["save_every_n_epochs"] == 0 or val_loss < best_val_loss:
            checkpoint_path = os.path.join(
                config["checkpoint_dir"],
                f"titan_v3_{config['model_size']}_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'perplexity': perplexity,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(
                    config["checkpoint_dir"],
                    f"titan_v3_{config['model_size']}_best.pth"
                )
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                }, best_checkpoint_path)
                print(f"  ⭐ Best model saved: {best_checkpoint_path}")
        
        model.train()
        
        # Reset short-term memory
        if hasattr(model, 'reset_memory'):
            model.reset_memory(level="short")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'perplexities': perplexities,
        'best_val_loss': best_val_loss
    }

def visualize_results(results, config):
    """Create training visualizations."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training Loss
    axes[0].plot(results['train_losses'], marker='o', label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[1].plot(results['val_losses'], marker='o', color='orange', label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Perplexity
    axes[2].plot(results['perplexities'], marker='o', color='green', label='Perplexity')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title('Perplexity (Lower is Better)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = f"{config['checkpoint_dir']}/training_progress.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {plot_path}")
    
    plt.show()

def main():
    """Main training pipeline."""
    print("="*80)
    print("TITAN-GPT V3 GOOGLE COLAB TRAINING")
    print("="*80)
    
    # Setup
    install_dependencies()
    device = setup_environment()
    repo_path = clone_repo()
    
    # Load data
    text_data = download_training_data()
    
    # Create dataloaders
    print("\n" + "="*80)
    print("CREATING DATALOADERS")
    print("="*80)
    train_loader, val_loader = create_simple_dataloaders(
        text_data,
        CONFIG["batch_size"],
        CONFIG["context_length"]
    )
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    try:
        # Try to import V3 model
        from models.titan_gpt_v3 import TitanGPTModelV3
        from configs.model_configs_v3 import get_model_config_v3
        
        v3_cfg = get_model_config_v3(CONFIG["model_size"], use_memory=True)
        v3_cfg["batch_size"] = CONFIG["batch_size"]
        model = TitanGPTModelV3(v3_cfg).to(device)
        print(f"✅ Using Titan-GPT V3 {CONFIG['model_size'].upper()} model")
    except:
        print("⚠️ V3 model not available, using fallback")
        # Fallback to basic GPT
        from gpt import GPTModel
        
        gpt_config = {
            "vocab_size": 50257,
            "context_length": CONFIG["context_length"],
            "emb_dim": 768 if CONFIG["model_size"] == "small" else 1024,
            "n_heads": 12 if CONFIG["model_size"] == "small" else 16,
            "n_layers": 12 if CONFIG["model_size"] == "small" else 24,
            "drop_rate": 0.1,
            "qkv_bias": False
        }
        model = GPTModel(gpt_config).to(device)
        print(f"✅ Using baseline GPT {CONFIG['model_size'].upper()} model")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Train
    results = train_model(model, train_loader, val_loader, optimizer, device, CONFIG)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_summary = {
        "model_size": CONFIG["model_size"],
        "total_params": total_params,
        "num_epochs": CONFIG["num_epochs"],
        "batch_size": CONFIG["batch_size"],
        "final_train_loss": results['train_losses'][-1],
        "final_val_loss": results['val_losses'][-1],
        "final_perplexity": results['perplexities'][-1],
        "best_val_loss": results['best_val_loss'],
        "train_losses": results['train_losses'],
        "val_losses": results['val_losses'],
        "perplexities": results['perplexities'],
    }
    
    results_path = os.path.join(CONFIG["checkpoint_dir"], f"training_results_{CONFIG['model_size']}.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ Results saved: {results_path}")
    
    # Visualize
    visualize_results(results, CONFIG)
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Model: Titan-GPT V3 {CONFIG['model_size'].upper()}")
    print(f"Parameters: {total_params:,}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Final Perplexity: {results['perplexities'][-1]:.2f}")
    print(f"\n📁 All checkpoints saved to: {CONFIG['checkpoint_dir']}")
    print("Access from: Google Drive → MyDrive → Titan_V3_Checkpoints")
    print("="*80)

if __name__ == "__main__":
    main()
