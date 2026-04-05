"""Training loop for the Pokémon name generator."""

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from pokegen.model import Transformer


def train(
    model: Transformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 20,
    save_path: str = "best_model.pt",
) -> dict:
    """Train the model with early stopping on validation loss.

    Args:
        model: Transformer model to train.
        inputs: Input token sequences.
        targets: Target token sequences.
        n_epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        patience: Early stopping patience (epochs without improvement).
        save_path: Path to save the best model weights.

    Returns:
        Dictionary with training history (train_losses, val_losses).
    """
    vocab_size = model.head.out_features
    device = next(model.parameters()).device

    train_in, val_in, train_tgt, val_tgt = train_test_split(
        inputs, targets, test_size=0.15
    )
    train_loader = DataLoader(
        TensorDataset(train_in.to(device), train_tgt.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_in.to(device), val_tgt.to(device)),
        batch_size=batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) ** -0.5, (step + 1) * 4000 ** -1.5)
        * 512**0.5,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    no_improve = 0
    history = {"train_losses": [], "val_losses": []}

    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for batch_input, batch_target in train_loader:
            logits = model(batch_input).view(-1, vocab_size)
            loss = loss_fn(logits, batch_target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                logits = model(batch_input).view(-1, vocab_size)
                val_loss += loss_fn(logits, batch_target.view(-1)).item()

        train_avg = total_loss / len(train_loader)
        val_avg = val_loss / len(val_loader)
        history["train_losses"].append(train_avg)
        history["val_losses"].append(val_avg)

        print(f"Epoch {epoch}/{n_epochs} — Train: {train_avg:.4f} — Val: {val_avg:.4f}")

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return history
