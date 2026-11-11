# Mini Transformer from Scratch

A small Decoder-only Transformer (Mini GPT) implemented
with PyTorch to understand the internal architecture of
Transformer models.

## Objectives

Build a Mini GPT from scratch and study the fundamental
components of modern Transformer architectures.

Main topics:

- Tokenization and Token Embedding
- Positional Embedding
- Scaled Dot-Product Attention
- Query, Key and Value
- Multi-Head Self-Attention
- Causal Masking
- Feed Forward Network
- Residual Connections
- Layer Normalization
- Autoregressive Text Generation

## Dataset

Tiny Shakespeare

A small text corpus containing Shakespeare's plays.

The project uses character-level language modeling.

Objective:

Predict the next character given previous characters
and generate new text autoregressively.

## Model

Decoder-only Mini GPT.

Initial configuration:

| Hyperparameter | Value |
|---|---:|
| Context Length | 64 |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Blocks | 2 |
| Feed Forward Dimension | 512 |
| Dropout | 0.1 |
| Batch Size | 16–32 |
| Optimizer | AdamW |

All configurations can be modified during experiments.

## Project Structure

```text
Mini-Transformer-from-Scratch/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_tokenization_and_batching.ipynb
│   ├── 03_scaled_dot_product_attention.ipynb
│   ├── 04_multi_head_attention.ipynb
│   ├── 05_bigram_baseline.ipynb
│   ├── 06_train_mini_transformer.ipynb
│   └── 07_text_generation_and_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── attention.py
│   ├── transformer_block.py
│   ├── model.py
│   ├── training.py
│   ├── evaluation.py
│   └── generation.py
│
├── checkpoints/
│
├── reports/
│   ├── figures/
│   └── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Launch Jupyter:

```powershell
jupyter notebook
```

## Workflow

1. Explore and download Tiny Shakespeare.
2. Build character-level tokenization.
3. Implement Scaled Dot-Product Attention.
4. Implement Multi-Head Attention and Causal Masking.
5. Build a Bigram Language Model as baseline.
6. Implement and train Mini GPT.
7. Generate text and visualize Attention.

## Experiments

Compare different configurations:

- Bigram vs Mini GPT
- Different numbers of Attention Heads
- Different numbers of Transformer Blocks
- Different Context Lengths
- Adam vs AdamW

## Evaluation

- Training Loss
- Validation Loss
- Perplexity
- Number of Parameters
- Training Time
- Generated Text Quality

## Notes

- The project is designed to run on CPU.
- Tiny Shakespeare will be downloaded into data/raw/.
- Raw datasets and model checkpoints are excluded from Git.
- The model is character-level and is intended for educational
  purposes, not general-purpose conversational AI.