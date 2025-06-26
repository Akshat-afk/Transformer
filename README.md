# Transformer

A from-scratch implementation of the Transformer architecture, built for educational and self-learning purposes. This repository walks through every component of the Transformer model, originally proposed in the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

## 🚀 Purpose

This project aims to:

- Deepen understanding of self-attention, positional encoding, and multi-head attention
- Explore how sequence modeling works without recurrence
- Serve as a learning reference for others studying transformers or deep learning

## 🧠 Key Features

- Encoder and Decoder blocks implemented step-by-step
- Scaled Dot-Product Attention and Multi-Head Attention
- Positional Encoding (sinusoidal)
- Masking for autoregressive decoding
- Cross-attention between encoder and decoder
- Full forward pass for training/inference
- Minimal dependencies, clean and readable code

## 📁 Structure

```
transformer/
├── model/
│   ├── attention.py         # Scaled dot-product & multi-head attention
│   ├── encoder.py           # Encoder block
│   ├── decoder.py           # Decoder block
│   ├── transformer.py       # Full Transformer architecture
│   └── positional_encoding.py
├── utils/
│   └── masks.py             # Padding and look-ahead masks
├── train.py                 # Example training loop (optional)
├── vocab.py                 # Simple tokenizer/vocab utilities
├── config.py                # Hyperparameter config
└── README.md
```

## 🛠️ Dependencies

- Python 3.8+
- PyTorch >= 1.10
- NumPy

Install using:

```bash
pip install -r requirements.txt
```

## 🧪 Example Usage

```python
from model.transformer import Transformer

model = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    dff=2048,
    input_vocab_size=8000,
    target_vocab_size=8000,
    max_pos_encoding=10000
)
```

## 📝 Status

✅ Core architecture  
🟡 Training loop / tokenizer  
🔲 Dataset integration  

## 🧭 Roadmap

- [ ] Add training script on toy data (e.g., Copy task, Translation)
- [ ] Integrate positional encoding visualizations
- [ ] Add support for BPE/tokenizers

## 🙌 Acknowledgements

Inspired by:

- Vaswani et al., 2017 — *Attention Is All You Need*
- The Annotated Transformer by Harvard NLP
- TensorFlow/NLP & PyTorch tutorials

## 📜 License

MIT License
