"""Compute-Optimal Sampling Training Pipeline

Implements training strategies from:
'Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling'
(arXiv:2408.16737)

Key features:
- Multi-model synthetic data generation (WC vs SE)
- Coverage, diversity, and false positive metrics
- Knowledge distillation, self-improvement, and weak-to-strong training
- Compute-optimal resource allocation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from collections import Counter


@dataclass
class SyntheticDataMetrics:
    """Metrics for evaluating synthetic training data quality."""
    coverage: float  # Proportion of unique problems solved
    diversity: float  # Average solutions per problem
    false_positive_rate: float  # Rate of incorrect solutions
    best_at_k: float  # Best performance among k samples
    worst_at_k: float  # Worst performance among k samples
    maj_at_k: float  # Majority voting performance


class SyntheticDataGenerator:
    """Generates synthetic training data using weaker/stronger models."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        temperature: float = 0.7,
        top_k: int = 50
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
    
    def generate_solutions(
        self,
        prompt: str,
        num_samples: int = 3,
        max_length: int = 256
    ) -> List[str]:
        """Generate multiple solutions for a single prompt.
        
        Args:
            prompt: Input prompt/problem
            num_samples: Number of solutions to generate (k)
            max_length: Maximum generation length
            
        Returns:
            List of generated solutions
        """
        self.model.eval()
        solutions = []
        
        # Encode prompt
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded).unsqueeze(0).to(self.device)
        
        for _ in range(num_samples):
            with torch.no_grad():
                # Generate with sampling
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=self.temperature,
                    top_k=self.top_k
                )
            
            # Decode solution
            solution = self.tokenizer.decode(
                output_ids[0].tolist()[len(encoded):]
            )
            solutions.append(solution)
        
        return solutions
    
    def generate_dataset(
        self,
        prompts: List[str],
        samples_per_prompt: int = 3,
        max_length: int = 256
    ) -> List[Tuple[str, List[str]]]:
        """Generate synthetic dataset from multiple prompts.
        
        Args:
            prompts: List of input prompts
            samples_per_prompt: Number of solutions per prompt
            max_length: Maximum generation length
            
        Returns:
            List of (prompt, solutions) tuples
        """
        dataset = []
        
        for prompt in prompts:
            solutions = self.generate_solutions(
                prompt,
                num_samples=samples_per_prompt,
                max_length=max_length
            )
            dataset.append((prompt, solutions))
        
        return dataset


class DataQualityEvaluator:
    """Evaluates quality of synthetic training data."""
    
    def __init__(
        self,
        correctness_fn: Optional[Callable[[str, str], bool]] = None
    ):
        """Initialize evaluator.
        
        Args:
            correctness_fn: Function to check if solution is correct
                          Takes (prompt, solution) -> bool
                          If None, uses simple heuristics
        """
        self.correctness_fn = correctness_fn
    
    def _default_correctness(self, prompt: str, solution: str) -> bool:
        """Default correctness heuristic (placeholder)."""
        # Simple heuristic: solution should be non-empty and different from prompt
        return len(solution.strip()) > 0 and solution != prompt
    
    def compute_coverage(
        self,
        dataset: List[Tuple[str, List[str]]]
    ) -> float:
        """Compute coverage: proportion of problems with at least one solution.
        
        Args:
            dataset: List of (prompt, solutions) tuples
            
        Returns:
            Coverage score [0, 1]
        """
        correctness_fn = self.correctness_fn or self._default_correctness
        
        num_solved = 0
        for prompt, solutions in dataset:
            # Check if any solution is correct
            if any(correctness_fn(prompt, sol) for sol in solutions):
                num_solved += 1
        
        return num_solved / len(dataset) if dataset else 0.0
    
    def compute_diversity(
        self,
        dataset: List[Tuple[str, List[str]]]
    ) -> float:
        """Compute diversity: average unique correct solutions per problem.
        
        Args:
            dataset: List of (prompt, solutions) tuples
            
        Returns:
            Average diversity score
        """
        correctness_fn = self.correctness_fn or self._default_correctness
        
        diversities = []
        for prompt, solutions in dataset:
            # Count unique correct solutions
            correct_solutions = set(
                sol for sol in solutions
                if correctness_fn(prompt, sol)
            )
            diversities.append(len(correct_solutions))
        
        return np.mean(diversities) if diversities else 0.0
    
    def compute_false_positive_rate(
        self,
        dataset: List[Tuple[str, List[str]]]
    ) -> float:
        """Compute false positive rate: proportion of incorrect solutions.
        
        Args:
            dataset: List of (prompt, solutions) tuples
            
        Returns:
            False positive rate [0, 1]
        """
        correctness_fn = self.correctness_fn or self._default_correctness
        
        total_solutions = 0
        incorrect_solutions = 0
        
        for prompt, solutions in dataset:
            total_solutions += len(solutions)
            for sol in solutions:
                if not correctness_fn(prompt, sol):
                    incorrect_solutions += 1
        
        return incorrect_solutions / total_solutions if total_solutions > 0 else 0.0
    
    def compute_maj_at_k(
        self,
        dataset: List[Tuple[str, List[str]]],
        ground_truth: List[str]
    ) -> float:
        """Compute majority voting accuracy (maj@k).
        
        Args:
            dataset: List of (prompt, solutions) tuples
            ground_truth: List of ground truth answers
            
        Returns:
            Majority voting accuracy [0, 1]
        """
        correct = 0
        
        for (prompt, solutions), truth in zip(dataset, ground_truth):
            # Get most common solution
            solution_counts = Counter(solutions)
            maj_solution = solution_counts.most_common(1)[0][0]
            
            # Check if matches ground truth
            if maj_solution == truth:
                correct += 1
        
        return correct / len(dataset) if dataset else 0.0
    
    def evaluate_dataset(
        self,
        dataset: List[Tuple[str, List[str]]],
        ground_truth: Optional[List[str]] = None
    ) -> SyntheticDataMetrics:
        """Comprehensive evaluation of synthetic dataset.
        
        Args:
            dataset: List of (prompt, solutions) tuples
            ground_truth: Optional ground truth for maj@k
            
        Returns:
            Complete metrics
        """
        coverage = self.compute_coverage(dataset)
        diversity = self.compute_diversity(dataset)
        fpr = self.compute_false_positive_rate(dataset)
        
        # Placeholder for best/worst at k (requires more complex evaluation)
        best_at_k = coverage  # Upper bound
        worst_at_k = 0.0  # Lower bound
        
        maj_at_k = 0.0
        if ground_truth is not None:
            maj_at_k = self.compute_maj_at_k(dataset, ground_truth)
        
        return SyntheticDataMetrics(
            coverage=coverage,
            diversity=diversity,
            false_positive_rate=fpr,
            best_at_k=best_at_k,
            worst_at_k=worst_at_k,
            maj_at_k=maj_at_k
        )


class ComputeOptimalTrainer:
    """Trainer implementing compute-optimal sampling strategies."""
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        device: torch.device = torch.device('cpu'),
        strategy: str = "self_improvement"
    ):
        """Initialize trainer.
        
        Args:
            student_model: Model to train
            teacher_model: Optional teacher for distillation/weak-to-strong
            device: Training device
            strategy: Training strategy:
                - "self_improvement": Train on own generations
                - "distillation": Learn from teacher
                - "weak_to_strong": Weaker teacher teaches stronger student
        """
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.device = device
        self.strategy = strategy
        
        if strategy in ["distillation", "weak_to_strong"] and teacher_model is None:
            raise ValueError(f"Strategy '{strategy}' requires a teacher model")
    
    def generate_training_data(
        self,
        prompts: List[str],
        tokenizer,
        samples_per_prompt: int = 3,
        use_teacher: bool = False
    ) -> List[Tuple[str, List[str]]]:
        """Generate training data using appropriate model.
        
        Args:
            prompts: Input prompts
            tokenizer: Tokenizer for encoding/decoding
            samples_per_prompt: Number of samples per prompt
            use_teacher: Whether to use teacher model
            
        Returns:
            Generated dataset
        """
        # Select model for generation
        if use_teacher and self.teacher_model is not None:
            gen_model = self.teacher_model
        else:
            gen_model = self.student_model
        
        # Generate data
        generator = SyntheticDataGenerator(
            model=gen_model,
            tokenizer=tokenizer,
            device=self.device
        )
        
        dataset = generator.generate_dataset(
            prompts=prompts,
            samples_per_prompt=samples_per_prompt
        )
        
        return dataset
    
    def compute_optimal_allocation(
        self,
        total_flops: float,
        weak_model_flops: float,
        strong_model_flops: float
    ) -> Tuple[int, int]:
        """Compute optimal sample allocation given FLOP budget.
        
        Args:
            total_flops: Total available FLOPs
            weak_model_flops: FLOPs per sample for weak model
            strong_model_flops: FLOPs per sample for strong model
            
        Returns:
            (weak_samples, strong_samples) tuple
        """
        # Under compute-optimal sampling, we prefer weak model
        # Allocate majority of budget to weak model
        
        weak_budget = total_flops * 0.8  # 80% to weak model
        strong_budget = total_flops * 0.2  # 20% to strong model
        
        weak_samples = int(weak_budget / weak_model_flops)
        strong_samples = int(strong_budget / strong_model_flops)
        
        return weak_samples, strong_samples
