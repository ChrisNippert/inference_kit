# Inference Kit

A collection of utility functions for working with causal language models using PyTorch and Transformers.

## Installation

**Important:** Install PyTorch first based on your system (CPU/CUDA/ROCm):

```bash
# For CPU (example)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (example)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (example)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# See https://pytorch.org/get-started/locally/ for your specific configuration
```

Then install inference_kit:

```bash
pip install -e .
# or
pip install transformers pandas
```

## Usage

```python
from inference_kit import init_model, generate

# Initialize model
tokenizer, model = init_model("meta-llama/Llama-3.1-8B-Instruct")

# Generate response
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I take the derivative of a cubic function?"}
]
response, full_response = generate(tokenizer, model, messages)
print(response)
```

## Features

- **Model initialization**: Load pretrained models with automatic device mapping
- **Text generation**: Generate responses from chat-formatted messages
- **Probability analysis**: Calculate log probabilities for sequences
- **Token prediction**: Get top-k next token predictions
- **Prompt branching**: Generate branched prompts with beam search
- **Evaluation**: Score model outputs
- **Embeddings**: Calculate sequence embedding similarity

## Functions

- `init_model(model_name)` - Initialize tokenizer and model
- `generate(tokenizer, model, messages, **kwargs)` - Generate text from chat messages
- `sum_seq_log_prob(tokenizer, model, string)` - Sum log probability of a sequence
- `mean_seq_log_prob(tokenizer, model, string)` - Mean log probability per token
- `get_k_next_tokens(tokenizer, model, prompt, k)` - Get top-k next tokens
- `branch_prompts(tokenizer, model, meta_prompt, k, n)` - Generate branched prompts
- `evaluate(tokenizer, model, prompt, outputs)` - Evaluate outputs for a prompt
- `sequence_embedding_similarity(tokenizer, model, seq1, seq2)` - Cosine similarity between sequences
