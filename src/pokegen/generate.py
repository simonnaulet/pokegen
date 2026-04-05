"""Name generation using a trained Transformer model."""

import torch
import torch.nn.functional as F

from pokegen.model import Transformer


def generate_name(
    model: Transformer,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    max_len: int = 25,
    temperature: float = 1.0,
) -> str:
    """Generate a single Pokémon name using autoregressive sampling.

    Args:
        model: Trained Transformer model.
        char_to_idx: Character-to-index mapping.
        idx_to_char: Index-to-character mapping.
        max_len: Maximum number of characters to generate.
        temperature: Sampling temperature (lower = more conservative).

    Returns:
        A generated Pokémon name string.
    """
    model.eval()
    device = next(model.parameters()).device
    input_seq = torch.tensor([[char_to_idx["<S>"]]], device=device)
    generated: list[str] = []

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_seq)[0, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == char_to_idx["<E>"]:
                break

            generated.append(idx_to_char[next_idx])
            input_seq = torch.cat(
                [input_seq, torch.tensor([[next_idx]], device=device)], dim=1
            )

    return "".join(generated)
