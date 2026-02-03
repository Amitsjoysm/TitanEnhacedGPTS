"""Synthetic reasoning dataset generation for compute-optimal training.

Generates diverse reasoning problems and solutions for training.
"""

import random
from typing import List, Tuple, Dict
import json


class SyntheticReasoningDataset:
    """Generates synthetic reasoning problems for training."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.problem_templates = self._create_templates()
    
    def _create_templates(self) -> List[Dict]:
        """Create problem templates for different reasoning types."""
        return [
            # Arithmetic reasoning
            {
                "type": "arithmetic",
                "template": "What is {a} {op} {b}?",
                "generator": self._gen_arithmetic,
            },
            # Logical reasoning
            {
                "type": "logical",
                "template": "If {premise}, then what can we conclude?",
                "generator": self._gen_logical,
            },
            # Pattern recognition
            {
                "type": "pattern",
                "template": "What comes next in the sequence: {sequence}?",
                "generator": self._gen_pattern,
            },
            # Word problems
            {
                "type": "word_problem",
                "template": "{scenario}. How many {item} are there?",
                "generator": self._gen_word_problem,
            },
            # Comparison
            {
                "type": "comparison",
                "template": "Which is {comparative}: {a} or {b}?",
                "generator": self._gen_comparison,
            },
        ]
    
    def _gen_arithmetic(self) -> Tuple[str, str]:
        """Generate arithmetic problem."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        ops = [
            ("+", a + b),
            ("-", a - b),
            ("*", a * b),
        ]
        op, answer = random.choice(ops)
        
        problem = f"What is {a} {op} {b}?"
        solution = f"The answer is {answer}."
        
        return problem, solution
    
    def _gen_logical(self) -> Tuple[str, str]:
        """Generate logical reasoning problem."""
        premises = [
            ("all birds can fly", "sparrows are birds", "sparrows can fly"),
            ("all cats are mammals", "tigers are cats", "tigers are mammals"),
            ("some flowers are red", "roses are flowers", "some roses might be red"),
        ]
        
        p1, p2, conclusion = random.choice(premises)
        problem = f"If {p1} and {p2}, what can we conclude?"
        solution = f"We can conclude that {conclusion}."
        
        return problem, solution
    
    def _gen_pattern(self) -> Tuple[str, str]:
        """Generate pattern recognition problem."""
        patterns = [
            ([2, 4, 6, 8], 10, "even numbers"),
            ([1, 3, 5, 7], 9, "odd numbers"),
            ([1, 2, 4, 8], 16, "powers of 2"),
            ([1, 4, 9, 16], 25, "perfect squares"),
        ]
        
        seq, next_val, pattern_type = random.choice(patterns)
        problem = f"What comes next in the sequence: {', '.join(map(str, seq))}?"
        solution = f"The next number is {next_val} (this is a sequence of {pattern_type})."
        
        return problem, solution
    
    def _gen_word_problem(self) -> Tuple[str, str]:
        """Generate word problem."""
        scenarios = [
            ("apples", "John has {a} apples. Mary gives him {b} more apples", "{total}"),
            ("books", "A library has {a} books on one shelf and {b} books on another shelf", "{total}"),
            ("students", "There are {a} students in class A and {b} students in class B", "{total}"),
        ]
        
        item, scenario_template, answer_template = random.choice(scenarios)
        a = random.randint(5, 50)
        b = random.randint(5, 50)
        total = a + b
        
        scenario = scenario_template.format(a=a, b=b)
        problem = f"{scenario}. How many {item} are there in total?"
        solution = f"There are {total} {item} in total."
        
        return problem, solution
    
    def _gen_comparison(self) -> Tuple[str, str]:
        """Generate comparison problem."""
        comparisons = [
            ("larger", "an elephant", "a mouse", "an elephant"),
            ("faster", "a cheetah", "a turtle", "a cheetah"),
            ("heavier", "a truck", "a bicycle", "a truck"),
            ("taller", "a giraffe", "a dog", "a giraffe"),
        ]
        
        comparative, a, b, answer = random.choice(comparisons)
        problem = f"Which is {comparative}: {a} or {b}?"
        solution = f"{answer.capitalize()} is {comparative}."
        
        return problem, solution
    
    def generate_dataset(
        self,
        num_problems: int = 100,
        balance_types: bool = True
    ) -> List[Tuple[str, str]]:
        """Generate a dataset of reasoning problems.
        
        Args:
            num_problems: Number of problems to generate
            balance_types: Whether to balance problem types
            
        Returns:
            List of (problem, solution) tuples
        """
        dataset = []
        
        if balance_types:
            # Generate equal number of each type
            problems_per_type = num_problems // len(self.problem_templates)
            
            for template in self.problem_templates:
                for _ in range(problems_per_type):
                    problem, solution = template["generator"]()
                    dataset.append((problem, solution))
        else:
            # Random types
            for _ in range(num_problems):
                template = random.choice(self.problem_templates)
                problem, solution = template["generator"]()
                dataset.append((problem, solution))
        
        # Shuffle
        random.shuffle(dataset)
        
        return dataset
    
    def save_dataset(
        self,
        dataset: List[Tuple[str, str]],
        filepath: str
    ):
        """Save dataset to JSON file."""
        data = [
            {"problem": problem, "solution": solution}
            for problem, solution in dataset
        ]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dataset(self, filepath: str) -> List[Tuple[str, str]]:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [(item["problem"], item["solution"]) for item in data]


def create_reasoning_dataset(num_problems: int = 100, save_path: str = None) -> List[Tuple[str, str]]:
    """Convenience function to create reasoning dataset.
    
    Args:
        num_problems: Number of problems to generate
        save_path: Optional path to save dataset
        
    Returns:
        Generated dataset
    """
    generator = SyntheticReasoningDataset()
    dataset = generator.generate_dataset(num_problems=num_problems)
    
    if save_path:
        generator.save_dataset(dataset, save_path)
    
    return dataset


if __name__ == "__main__":
    # Example usage
    dataset = create_reasoning_dataset(
        num_problems=100,
        save_path="/app/titan-optimal/data/synthetic_reasoning_100.json"
    )
    
    print(f"Generated {len(dataset)} problems")
    print("\nExample problems:")
    for i, (problem, solution) in enumerate(dataset[:5]):
        print(f"\n{i+1}. {problem}")
        print(f"   Answer: {solution}")
