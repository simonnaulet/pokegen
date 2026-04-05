"""Data loading and preprocessing for Pokémon name generation."""

import httpx
import torch
from torch.nn.utils.rnn import pad_sequence

POKEAPI_URL = "https://pokeapi.co/api/v2/pokemon/?limit=2000"

START_TOKEN = "<S>"
END_TOKEN = "<E>"
PAD_TOKEN = "<P>"


def load_pokemon_names() -> list[str]:
    """Fetch all Pokémon names from the PokéAPI."""
    response = httpx.get(POKEAPI_URL)
    response.raise_for_status()
    return [p["name"] for p in response.json()["results"]]


def build_vocab(names: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """Build character-level vocabulary from a list of names.

    Returns:
        idx_to_char: Mapping from index to character.
        char_to_idx: Mapping from character to index.
    """
    chars = sorted(set().union(*names))
    alphabet = [START_TOKEN, END_TOKEN, PAD_TOKEN] + chars
    idx_to_char = dict(enumerate(alphabet))
    char_to_idx = {v: k for k, v in idx_to_char.items()}
    return idx_to_char, char_to_idx


def encode_names(
    names: list[str], char_to_idx: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize and pad names into input/target tensor pairs.

    Returns:
        inputs: Tensor of shape (N, max_len - 1) — all tokens except the last.
        targets: Tensor of shape (N, max_len - 1) — all tokens except the first.
        max_len: Maximum sequence length (including start/end tokens).
    """
    tokenized = [[START_TOKEN] + list(name) + [END_TOKEN] for name in names]
    max_len = max(map(len, tokenized))
    encoded = [torch.tensor([char_to_idx[t] for t in tokens]) for tokens in tokenized]
    batch = pad_sequence(encoded, batch_first=True, padding_value=char_to_idx[PAD_TOKEN])
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return inputs, targets, max_len
