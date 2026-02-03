# Titan-Optimal: Next-Generation LLM Architecture

## Overview

Titan-Optimal is a novel LLM architecture that combines two cutting-edge research innovations:

1. **Titans Architecture** (arXiv:2501.00663): Neural long-term memory modules for extended context processing
2. **Compute-Optimal Sampling** (arXiv:2408.16737): Training with weaker models for better reasoning performance

## Key Features

### Architecture Innovations

- **Dual Memory System**:
  - **Short-term memory**: Multi-head attention for accurate local dependencies
  - **Long-term memory**: Neural memory modules for persistent historical context
  
- **Surprise-Based Learning**: Selective memory updates based on gradient-driven surprise metrics

- **Adaptive Forgetting**: Intelligent memory management that retains important information

- **Extended Context Windows**: Capable of processing 2M+ tokens efficiently

### Training Innovations

- **Compute-Optimal Data Generation**: Uses weaker models to generate diverse training data
- **Multiple Training Strategies**:
  - Self-improvement
  - Knowledge distillation  
  - Weak-to-strong learning

- **Quality Metrics**: Coverage, diversity, and false positive rate evaluation

## Project Structure

```
titan_optimal/
├── models/
│   ├── neural_memory.py       # Neural long-term memory module
│   └── titan_gpt.py          # Complete Titan-GPT architecture
├── training/
│   └── compute_optimal.py    # Compute-optimal training pipeline
├── synthetic_data/
│   └── reasoning_generator.py # Synthetic dataset generation
├── evaluation/
│   └── benchmarks.py         # Evaluation and benchmarking
├── configs/
│   └── model_configs.py      # Model and training configurations
├── data/                     # Generated datasets
├── checkpoints/              # Saved model checkpoints
├── notebooks/                # Jupyter notebooks for exploration
└── train.py                  # Main training script
```

## Model Sizes

Three model sizes are supported:

1. **Small (124M params)**: Fast iteration, CPU-friendly
2. **Medium (340M params)**: Balance of performance and efficiency
3. **Large (760M params)**: Maximum capability

Each size is available with or without neural memory for comparison.

## Quick Start

### Installation

```bash
cd /app/titan_optimal

# Install additional dependencies (if needed)
pip install matplotlib numpy requests
```

### Training

```bash
# Train all model variants (Baseline GPT, Titan-MAC, Titan-MAG)
python train.py

# Or import and customize:
python -c "from train import main; main()"
```

### Generating Synthetic Data

```python
from synthetic_data.reasoning_generator import create_reasoning_dataset

# Generate reasoning problems
dataset = create_reasoning_dataset(
    num_problems=100,
    save_path="data/reasoning_100.json"
)
```

### Evaluation

```python
from evaluation.benchmarks import run_comprehensive_evaluation
import torch
import tiktoken

# Load models
models = {
    "Baseline": baseline_model,
    "Titan-MAC": titan_mac_model,
}

# Run evaluation
results = run_comprehensive_evaluation(
    models=models,
    test_loader=test_loader,
    device=device,
    tokenizer=tiktoken.get_encoding("gpt2")
)
```

## Architecture Details

### Neural Memory Module

The neural memory module implements:

1. **Key-Value Storage**: Projects inputs to keys and values
2. **Surprise Metric**: Computes novelty using loss gradients
3. **Selective Updates**: Updates memory based on surprise scores
4. **Adaptive Forgetting**: Manages memory capacity with decay
5. **Efficient Retrieval**: Query-based memory access via MLP

### Memory Variants

Three integration variants:

- **MAC (Memory as Context)**: Adds retrieved memory to attention output
- **MAG (Memory as Gate)**: Uses memory to gate attention output  
- **Hybrid**: Combines both context and gating

### Compute-Optimal Training

Key principles:

1. **Weaker models generate more samples** per FLOP than stronger models
2. **Higher coverage and diversity** lead to better trained models
3. **False positives are acceptable** if compensated by volume
4. **Sample allocation** should favor weaker generators

## Performance Expectations

Based on research papers:

- **Titans vs Transformers**: Better performance on long contexts (2K-16K+ tokens)
- **Compute-Optimal vs Standard**: Up to 8% improvement in reasoning tasks
- **Memory Efficiency**: Linear scaling vs quadratic for standard attention

## Configuration

Edit `configs/model_configs.py` to customize:

```python
TITAN_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "memory_size": 512,          # Size of long-term memory
    "memory_variant": "mac",     # mac, mag, or hybrid
    "surprise_momentum": 0.9,    # Momentum for surprise metric
    "forget_decay": 0.01,        # Forgetting rate
}
```

## Research Papers

### Titans: Learning to Memorize at Test Time
- **arXiv**: 2501.00663
- **Authors**: Ali Behrouz, Peilin Zhong, Vahab Mirrokni
- **Key Innovation**: Neural long-term memory for transformers

### Compute-Optimal Sampling
- **arXiv**: 2408.16737  
- **Authors**: Hritik Bansal et al.
- **Key Innovation**: Training with weaker models for better reasoning

## Extending the Code

### Adding Custom Memory Variants

```python
class CustomTransformerBlock(TitanTransformerBlock):
    def forward(self, x, loss_grad=None, update_memory=True):
        # Implement custom memory integration
        pass
```

### Custom Training Strategies

```python
from training.compute_optimal import ComputeOptimalTrainer

trainer = ComputeOptimalTrainer(
    student_model=model,
    strategy="custom_strategy"
)
```

## CPU vs GPU Optimization

The code automatically detects and uses available hardware:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For CPU optimization:
- Use smaller batch sizes (2-4)
- Enable gradient accumulation
- Reduce model size (124M variant)
- Shorter context lengths (256-512)

For GPU optimization:
- Larger batch sizes (8-16)
- Longer context lengths (1024-2048)
- Enable mixed precision training (optional)

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Reduce context length
- Use smaller model variant

### Slow Training
- Enable GPU if available
- Reduce memory size
- Disable 1D convolutions
- Use fewer memory layers

### Poor Performance
- Increase training epochs
- Adjust learning rate
- Try different memory variants
- Generate more synthetic data

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}

@article{bansal2024smaller,
  title={Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling},
  author={Bansal, Hritik and Hosseini, Arian and Agarwal, Rishabh and Tran, Vinh Q and Kazemi, Mehran},
  journal={arXiv preprint arXiv:2408.16737},
  year={2024}
}
```

## License

This implementation is for educational and research purposes, building upon the concepts from the cited papers and the "Build a Large Language Model (From Scratch)" book by Sebastian Raschka.

## Contact

For questions or issues, please open an issue in the repository.
