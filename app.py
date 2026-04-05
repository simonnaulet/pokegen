"""Gradio demo for the Pokémon name generator."""

from pathlib import Path

import gradio as gr
import torch

from pokegen.data import build_vocab, load_pokemon_names
from pokegen.generate import generate_name
from pokegen.model import Transformer

MODEL_PATH = Path(__file__).parent / "best_model.pt"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> tuple[Transformer, dict[str, int], dict[int, str]]:
    """Load vocabulary and trained model weights."""
    names = load_pokemon_names()
    idx_to_char, char_to_idx = build_vocab(names)

    model = Transformer(
        d_model=256,
        vocab_size=len(idx_to_char),
        max_len=29,
        n_heads=8,
        n_layers=3,
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, weights_only=True, map_location=device)
    )
    model.eval()
    return model, char_to_idx, idx_to_char


model, char_to_idx, idx_to_char = load_model()


def generate(count: int, temperature: float) -> str:
    """Generate multiple Pokémon names."""
    names = set()
    attempts = 0
    while len(names) < count and attempts < count * 3:
        name = generate_name(model, char_to_idx, idx_to_char, temperature=temperature)
        if name:
            names.add(name)
        attempts += 1
    return "\n".join(f"• {name.capitalize()}" for name in sorted(names))


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Slider(1, 20, value=5, step=1, label="Number of names"),
        gr.Slider(0.3, 1.5, value=0.8, step=0.1, label="Temperature",
                  info="Lower = more conservative, Higher = more creative"),
    ],
    outputs=gr.Textbox(label="Generated Pokémon names", lines=10),
    title="PokeGen — Pokémon Name Generator",
    description=(
        "Generate new Pokémon names using a **Transformer decoder** trained from scratch "
        "on all ~1,350 official Pokémon names. "
        "Adjust the temperature to control creativity."
    ),
    examples=[[5, 0.5], [10, 0.8], [5, 1.2]],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
