"""Evaluation and benchmarking utilities for Titan-Optimal models."""

import torch
import torch.nn as nn
import tiktoken
import time
from typing import List, Dict, Tuple
import numpy as np


class PerplexityEvaluator:
    """Evaluate model perplexity on test data."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def calculate_perplexity(
        self,
        data_loader,
        max_batches: int = None
    ) -> float:
        """Calculate perplexity on data.
        
        Args:
            data_loader: DataLoader with test data
            max_batches: Maximum batches to evaluate
            
        Returns:
            Perplexity score
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        num_batches = len(data_loader) if max_batches is None else min(max_batches, len(data_loader))
        
        with torch.no_grad():
            for i, (input_batch, target_batch) in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                logits = self.model(input_batch, update_memory=False)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_batch.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += target_batch.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity


class InferenceSpeedBenchmark:
    """Benchmark inference speed and throughput."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def benchmark_generation(
        self,
        prompt: str,
        tokenizer,
        num_tokens: int = 100,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """Benchmark text generation speed.
        
        Args:
            prompt: Input prompt
            tokenizer: Tokenizer
            num_tokens: Number of tokens to generate
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing metrics
        """
        self.model.eval()
        
        # Encode prompt
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded).unsqueeze(0).to(self.device)
        
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    temperature=1.0
                )
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        tokens_per_sec = num_tokens / avg_time
        
        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": num_tokens,
        }


class MemoryEfficiencyAnalyzer:
    """Analyze memory usage of different model variants."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count total and trainable parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
        }
    
    @staticmethod
    def estimate_memory_footprint(
        model: nn.Module,
        batch_size: int = 1,
        seq_length: int = 512
    ) -> Dict[str, float]:
        """Estimate memory footprint in MB.
        
        Returns:
            Dictionary with memory estimates
        """
        param_count = sum(p.numel() for p in model.parameters())
        
        # Parameter memory (in MB)
        # Assuming float32 (4 bytes per parameter)
        param_memory = (param_count * 4) / (1024 ** 2)
        
        # Activation memory (rough estimate)
        # Depends on batch size and sequence length
        activation_memory = (batch_size * seq_length * 4 * 1024) / (1024 ** 2)
        
        return {
            "parameter_memory_mb": param_memory,
            "activation_memory_mb": activation_memory,
            "total_estimate_mb": param_memory + activation_memory,
        }


class ContextWindowEvaluator:
    """Evaluate performance on different context window sizes."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    def needle_in_haystack(
        self,
        context: str,
        needle: str,
        tokenizer,
        question: str = "What is the hidden fact?"
    ) -> Tuple[bool, str]:
        """Test if model can retrieve information from long context.
        
        Args:
            context: Long context with hidden information
            needle: The hidden information to retrieve
            tokenizer: Tokenizer
            question: Question to ask
            
        Returns:
            (success, generated_answer) tuple
        """
        self.model.eval()
        
        # Insert needle in middle of context
        words = context.split()
        mid_point = len(words) // 2
        context_with_needle = " ".join(
            words[:mid_point] + [needle] + words[mid_point:]
        )
        
        # Create prompt
        prompt = f"{context_with_needle}\n\nQuestion: {question}\nAnswer:"
        
        # Generate answer
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7
            )
        
        # Decode answer
        answer = tokenizer.decode(
            output_ids[0].tolist()[len(encoded):]
        )
        
        # Check if needle info is in answer
        success = needle.lower() in answer.lower()
        
        return success, answer


def run_comprehensive_evaluation(
    models: Dict[str, nn.Module],
    test_loader,
    device: torch.device,
    tokenizer
) -> Dict[str, Dict]:
    """Run comprehensive evaluation on multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        test_loader: Test data loader
        device: Device
        tokenizer: Tokenizer
        
    Returns:
        Dictionary of results per model
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Perplexity
        perplexity_eval = PerplexityEvaluator(model, device)
        perplexity = perplexity_eval.calculate_perplexity(test_loader, max_batches=50)
        
        # Inference speed
        speed_bench = InferenceSpeedBenchmark(model, device)
        speed_metrics = speed_bench.benchmark_generation(
            prompt="The quick brown fox",
            tokenizer=tokenizer,
            num_tokens=100,
            num_runs=3
        )
        
        # Memory analysis
        param_counts = MemoryEfficiencyAnalyzer.count_parameters(model)
        memory_est = MemoryEfficiencyAnalyzer.estimate_memory_footprint(model)
        
        results[model_name] = {
            "perplexity": perplexity,
            "speed": speed_metrics,
            "parameters": param_counts,
            "memory": memory_est,
        }
        
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Speed: {speed_metrics['tokens_per_sec']:.2f} tokens/sec")
        print(f"  Parameters: {param_counts['total']:,}")
    
    return results
