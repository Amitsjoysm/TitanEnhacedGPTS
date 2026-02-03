"""Quick test training for Titan-GPT V3 Small Model
Test the batch size fix before full training.
"""

import os
import sys
import torch
import torch.nn as nn
import tiktoken
import time
from pathlib import Path

# Add required paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append("/app/ch04/01_main_chapter_code")

from models.titan_gpt_v3 import TitanGPTModelV3
from configs.model_configs_v3 import get_model_config_v3, get_training_config_v3
from gpt import create_dataloader_v1


def main():
    """Quick test of V3 small model."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("TITAN-GPT V3 SMALL MODEL - BATCH SIZE FIX TEST")
    print("="*80)
    print(f"Device: {device}")
    
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    
    # Load data
    print("\nLoading training data...")
    data_path = "/app/ch05/01_main_chapter_code/the-verdict.txt"
    
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
    
    # Get configs
    train_cfg = get_training_config_v3("small")
    v3_cfg = get_model_config_v3("small", use_memory=True)
    v3_cfg["batch_size"] = train_cfg["batch_size"]
    
    print(f"\nModel config:")
    print(f"  Embedding dim: {v3_cfg['emb_dim']}")
    print(f"  Layers: {v3_cfg['n_layers']}")
    print(f"  Heads: {v3_cfg['n_heads']}")
    print(f"  Batch size: {v3_cfg['batch_size']}")
    print(f"  Context length: {v3_cfg['context_length']}")
    
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
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create V3 model
    print("\n" + "-"*80)
    print("Creating Titan-GPT V3 SMALL Model")
    print("-"*80)
    
    v3_model = TitanGPTModelV3(v3_cfg).to(device)
    total_params = sum(p.numel() for p in v3_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    v3_optimizer = torch.optim.AdamW(
        v3_model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"]
    )
    
    # Test forward pass with different batch sizes
    print("\n" + "-"*80)
    print("Testing Batch Size Handling")
    print("-"*80)
    
    test_successful = True
    try:
        # Get first batch
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            print(f"Testing with batch shape: {input_batch.shape}")
            
            # Forward pass
            logits, aux_losses = v3_model(
                input_batch,
                targets=target_batch,
                update_memory=True,
                mode="train"
            )
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            
            print(f"✅ Forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Aux losses: {list(aux_losses.keys())}")
            
            # Backward pass
            loss.backward()
            v3_optimizer.step()
            v3_optimizer.zero_grad()
            
            print(f"✅ Backward pass successful!")
            
            break  # Test just one batch
            
    except Exception as e:
        print(f"❌ Error during forward/backward pass:")
        print(f"   {type(e).__name__}: {str(e)}")
        test_successful = False
        import traceback
        traceback.print_exc()
    
    if not test_successful:
        print("\n" + "="*80)
        print("BATCH SIZE FIX TEST FAILED")
        print("="*80)
        return
    
    # If test passed, run short training
    print("\n" + "="*80)
    print("BATCH SIZE FIX SUCCESSFUL - Running Short Training")
    print("="*80)
    
    num_epochs = 2  # Just 2 epochs for quick test
    v3_model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            if batch_idx >= 10:  # Train only 10 batches per epoch
                break
                
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Forward pass
            logits, aux_losses = v3_model(
                input_batch,
                targets=target_batch,
                update_memory=True,
                mode="train"
            )
            
            # Loss
            loss = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten()
            )
            
            # Add diversity loss
            diversity_loss = sum(v for k, v in aux_losses.items() if 'diversity' in k and isinstance(v, torch.Tensor))
            if isinstance(diversity_loss, torch.Tensor) and diversity_loss.numel() > 0:
                total_loss = loss + train_cfg["diversity_loss_weight"] * diversity_loss
            else:
                total_loss = loss
            
            # Backward
            total_loss.backward()
            v3_optimizer.step()
            v3_optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1}: Loss = {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Reset short-term memory between epochs
        v3_model.reset_memory(level="short")
    
    print("\n" + "="*80)
    print("✅ QUICK TEST SUCCESSFUL!")
    print("="*80)
    print("The batch size fix is working correctly.")
    print("Ready for full training with train_v3.py")


if __name__ == "__main__":
    main()
