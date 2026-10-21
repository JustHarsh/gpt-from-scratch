# My GPT — Built from Scratch

## Project Structure

```
model/          Attention, Transformer, GPT architecture
  attention.py        Self-attention head
  multi_head_attention.py   Multi-headed attention
  transformer.py      Transformer block
  gpt.py              GPT model
  normalization.py    Layer normalization
  embeddings.py       Word embeddings
  positional_encoding.py  Positional encoding

data/           Data pipeline
  tokenizer.py        BPE tokenizer
  vocab.py            Character-level vocabulary
  loader.py           Batched training data loader
  dataset.py          GPT dataset preparation
  nlp_preprocessing.py    NLP preprocessing

train.py        GPT training loop
generate.py     Text generation

foundations/    Neural network primitives built from scratch
  neuron.py, backprop.py, mlp.py, activations.py, loss.py,
  training_loop.py, ...
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python generate.py
```

This project comprises of:
- Math Foundations (gradient descent, activations, loss functions)
- Neural Networks from scratch (neuron, backprop, MLP)
- PyTorch fundamentals
- NLP pipeline (embeddings, tokenization, attention)
- Transformer architecture
- GPT model + text generation
