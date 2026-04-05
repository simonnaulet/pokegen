# PokeGen — Pokémon Name Generator

A character-level **Transformer decoder** built from scratch in PyTorch, trained on all ~1,350 official Pokémon names to generate new, plausible-sounding names.

## Examples

```
Temperature 0.5 (conservative)     Temperature 1.0 (creative)
  Corombe                             Togembr
  Spoliwkoo                           Labliti
  Minior                              Coshier
  Pynash                              Fealettr
  Deranin                             Zhora
```

## Architecture

The model is a **decoder-only Transformer** (same family as GPT), implemented entirely from scratch — no `nn.TransformerDecoder`:

- **Token embedding** scaled by √d_model
- **Sinusoidal positional encoding**
- **Multi-head causal self-attention** (8 heads, 256 dims)
- **Feed-forward network** with ReLU (256 → 1024 → 256)
- **3 decoder blocks** with residual connections, LayerNorm, and dropout

Training uses the Adam optimizer with the warmup schedule from *Attention Is All You Need*, gradient clipping, and early stopping.

## Project Structure

```
pokegen/
├── app.py                  # Gradio demo (web interface)
├── notebook.ipynb          # Step-by-step walkthrough notebook
├── best_model.pt           # Trained model weights
├── src/pokegen/
│   ├── __init__.py
│   ├── model.py            # Transformer architecture
│   ├── data.py             # Data loading & tokenization
│   ├── generate.py         # Name generation (autoregressive sampling)
│   └── train.py            # Training loop
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/pokegen.git
cd pokegen

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Interactive Demo

```bash
python app.py
```

Opens a web interface at `http://localhost:7860` where you can generate names and adjust the temperature.

### Notebook

```bash
jupyter notebook notebook.ipynb
```

The notebook walks through the entire pipeline: data collection, tokenization, model architecture, training, and generation — with explanations at each step.

### From Python

```python
import torch
from pokegen import Transformer, load_pokemon_names, build_vocab, generate_name

# Load vocab and model
names = load_pokemon_names()
idx_to_char, char_to_idx = build_vocab(names)

model = Transformer(d_model=256, vocab_size=len(idx_to_char), max_len=30, n_heads=8, n_layers=3)
model.load_state_dict(torch.load("best_model.pt", weights_only=True))

# Generate a name
print(generate_name(model, char_to_idx, idx_to_char, temperature=0.8))
```

### Train from Scratch

```python
from pokegen import Transformer, load_pokemon_names, build_vocab, encode_names
from pokegen.train import train

names = load_pokemon_names()
idx_to_char, char_to_idx = build_vocab(names)
inputs, targets, max_len = encode_names(names, char_to_idx)

model = Transformer(d_model=256, vocab_size=len(idx_to_char), max_len=max_len, n_heads=8, n_layers=3)
history = train(model, inputs, targets, n_epochs=200)
```

## How It Works

1. **Data**: All Pokémon names are fetched from the [PokéAPI](https://pokeapi.co/)
2. **Tokenization**: Each name is split into characters with `<S>`tart and `<E>`nd tokens, then padded to equal length
3. **Training**: The model learns to predict the next character given all previous characters (autoregressive language modeling)
4. **Generation**: Starting from `<S>`, the model samples one character at a time until it produces `<E>` or reaches the max length

## Tech Stack

- **PyTorch** — model and training
- **Gradio** — interactive web demo
- **httpx** — API requests
- **scikit-learn** — train/val split

## License

MIT
