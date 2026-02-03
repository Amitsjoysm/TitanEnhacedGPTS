"""Quick training test with minimal resources."""

import os
import sys
import torch
import torch.nn as nn
import tiktoken
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append("/app/ch04/01_main-chapter-code")

from models.titan_gpt import TitanGPTModel
from configs.model_configs import get_model_config
from gpt import create_dataloader_v1


def quick_training_test():
    """Test training loop with minimal data."""
    
    print("="*70)
    print("QUICK TRAINING TEST")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Use very small sample text for quick test
    text_data = """The quick brown fox jumps over the lazy dog. """ * 20
    
    print(f"\nTest data size: {len(text_data)} characters")
    
    # Create minimal config
    config = get_model_config("small", use_memory=True)
    config["context_length"] = 64  # Very short for speed
    config["n_layers"] = 2  # Fewer layers
    config["memory_size"] = 64  # Smaller memory
    
    print(f"\nCreating Titan model...")
    model = TitanGPTModel(config).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Create dataloader
    print(f"\nCreating dataloader...")
    train_loader = create_dataloader_v1(
        text_data,
        batch_size=2,
        max_length=64,
        stride=64,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Batches: {len(train_loader)}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.1
    )
    
    # Training loop
    print(f"\nTraining for 3 steps...")
    model.train()
    
    for step, (input_batch, target_batch) in enumerate(train_loader):
        if step >= 3:  # Only 3 steps
            break
        
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        # Forward pass
        logits = model(input_batch, targets=target_batch, update_memory=True)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten()
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    print(f"\n✓ Training test completed successfully!")
    
    # Test generation
    print(f"\nTesting generation...")
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "The quick"
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=1.0
        )
    
    generated = tokenizer.decode(output_ids[0].tolist())
    print(f"  Prompt: '{prompt}'")
    print(f"  Generated: '{generated}'")
    
    print(f"\n✓ Generation test completed successfully!")
    
    # Test memory reset
    print(f"\nTesting memory reset...")
    model.reset_memory()
    print(f"✓ Memory reset successful!")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nThe implementation is working correctly.")
    print("You can now run the full training with: python train.py")


if __name__ == "__main__":
    quick_training_test()
