"""Demonstration script for Titan-Optimal architecture.

Shows key features:
1. Model initialization (small and medium sizes)
2. Memory operations
3. Text generation
4. Synthetic data generation
5. Compute-optimal training metrics
"""

import sys
import torch
import tiktoken
from pathlib import Path

# Add required paths
sys.path.append(str(Path(__file__).parent))
sys.path.append("/app/ch04/01_main-chapter-code")

from models.titan_gpt import TitanGPTModel
from configs.model_configs import get_model_config, get_training_config
from synthetic_data.reasoning_generator import create_reasoning_dataset
from training.compute_optimal import SyntheticDataGenerator, DataQualityEvaluator
from evaluation.benchmarks import MemoryEfficiencyAnalyzer


def demo_model_initialization():
    """Demonstrate model initialization with different configs."""
    print("="*70)
    print("DEMO 1: Model Initialization")
    print("="*70)
    
    # Small model
    print("\n1. Creating Small Titan-GPT (124M params)...")
    small_config = get_model_config("small", use_memory=True)
    small_model = TitanGPTModel(small_config)
    
    analyzer = MemoryEfficiencyAnalyzer()
    small_params = analyzer.count_parameters(small_model)
    small_memory = analyzer.estimate_memory_footprint(small_model)
    
    print(f"   Total parameters: {small_params['total']:,}")
    print(f"   Trainable: {small_params['trainable']:,}")
    print(f"   Estimated memory: {small_memory['total_estimate_mb']:.2f} MB")
    
    # Medium model
    print("\n2. Creating Medium Titan-GPT (340M params)...")
    medium_config = get_model_config("medium", use_memory=True)
    medium_model = TitanGPTModel(medium_config)
    
    medium_params = analyzer.count_parameters(medium_model)
    medium_memory = analyzer.estimate_memory_footprint(medium_model)
    
    print(f"   Total parameters: {medium_params['total']:,}")
    print(f"   Trainable: {medium_params['trainable']:,}")
    print(f"   Estimated memory: {medium_memory['total_estimate_mb']:.2f} MB")
    
    # Baseline (no memory)
    print("\n3. Creating Baseline GPT (no memory)...")
    baseline_config = get_model_config("small", use_memory=False)
    baseline_model = TitanGPTModel(baseline_config)
    
    baseline_params = analyzer.count_parameters(baseline_model)
    
    print(f"   Total parameters: {baseline_params['total']:,}")
    print(f"   Parameter difference vs Titan: {small_params['total'] - baseline_params['total']:,}")
    
    return small_model, medium_model, baseline_model


def demo_memory_operations(model):
    """Demonstrate memory operations."""
    print("\n" + "="*70)
    print("DEMO 2: Neural Memory Operations")
    print("="*70)
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    dummy_input = torch.randint(0, 50257, (batch_size, seq_len))
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass with memory updates
    print("\n1. Forward pass with memory updates...")
    model.train()
    with torch.no_grad():
        output1 = model(dummy_input, update_memory=True)
    print(f"   Output shape: {output1.shape}")
    print(f"   Memory initialized: {model.trf_blocks[0].neural_memory.memory_keys is not None}")
    
    # Second pass - memory should be populated
    print("\n2. Second forward pass (memory populated)...")
    with torch.no_grad():
        output2 = model(dummy_input, update_memory=True)
    print(f"   Output shape: {output2.shape}")
    print(f"   Memory size: {model.trf_blocks[0].neural_memory.memory_size}")
    
    # Reset memory
    print("\n3. Resetting memory...")
    model.reset_memory()
    print(f"   Memory cleared: {model.trf_blocks[0].neural_memory.memory_keys is None}")
    
    return output1, output2


def demo_text_generation(model):
    """Demonstrate text generation."""
    print("\n" + "="*70)
    print("DEMO 3: Text Generation")
    print("="*70)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "The key to success is",
    ]
    
    model.eval()
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: \"{prompt}\"")
        
        # Encode prompt
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50
            )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(f"   Generated: \"{generated_text}\"")


def demo_synthetic_data():
    """Demonstrate synthetic data generation."""
    print("\n" + "="*70)
    print("DEMO 4: Synthetic Reasoning Data Generation")
    print("="*70)
    
    print("\nGenerating 20 reasoning problems...")
    dataset = create_reasoning_dataset(num_problems=20)
    
    print(f"\nGenerated {len(dataset)} problems")
    print("\nExamples:")
    
    for i, (problem, solution) in enumerate(dataset[:5], 1):
        print(f"\n{i}. Problem: {problem}")
        print(f"   Solution: {solution}")
    
    return dataset


def demo_data_quality_metrics(dataset):
    """Demonstrate data quality evaluation."""
    print("\n" + "="*70)
    print("DEMO 5: Data Quality Metrics")
    print("="*70)
    
    # Create evaluator
    evaluator = DataQualityEvaluator()
    
    # Convert to format expected by evaluator
    eval_dataset = [
        (problem, [solution])  # Single solution per problem for demo
        for problem, solution in dataset
    ]
    
    print("\nEvaluating synthetic data quality...")
    
    coverage = evaluator.compute_coverage(eval_dataset)
    diversity = evaluator.compute_diversity(eval_dataset)
    fpr = evaluator.compute_false_positive_rate(eval_dataset)
    
    print(f"\nMetrics:")
    print(f"  Coverage: {coverage:.2%} (problems with at least one solution)")
    print(f"  Diversity: {diversity:.2f} (unique solutions per problem)")
    print(f"  False Positive Rate: {fpr:.2%} (incorrect solutions)")


def demo_memory_variants():
    """Demonstrate different memory integration variants."""
    print("\n" + "="*70)
    print("DEMO 6: Memory Integration Variants")
    print("="*70)
    
    variants = ["mac", "mag", "hybrid"]
    
    for variant in variants:
        print(f"\n{variant.upper()} (Memory as {'Context' if variant == 'mac' else 'Gate' if variant == 'mag' else 'Context+Gate'}):")
        
        config = get_model_config("small", use_memory=True)
        config["memory_variant"] = variant
        model = TitanGPTModel(config)
        
        # Test forward pass
        dummy_input = torch.randint(0, 50257, (1, 16))
        with torch.no_grad():
            output = model(dummy_input, update_memory=False)
        
        print(f"  Model created successfully")
        print(f"  Output shape: {output.shape}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("TITAN-OPTIMAL ARCHITECTURE DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases the key features of Titan-Optimal:")
    print("1. Neural long-term memory modules")
    print("2. Multiple model sizes (124M, 340M, 760M params)")
    print("3. Text generation capabilities")
    print("4. Synthetic data generation")
    print("5. Compute-optimal training metrics")
    print("6. Different memory integration variants")
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    
    # Run demos
    small_model, medium_model, baseline_model = demo_model_initialization()
    
    demo_memory_operations(small_model)
    
    demo_text_generation(small_model)
    
    dataset = demo_synthetic_data()
    
    demo_data_quality_metrics(dataset)
    
    demo_memory_variants()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run full training: python train.py")
    print("2. Check README.md for detailed usage")
    print("3. Explore notebooks/ for interactive examples")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
