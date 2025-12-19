"""Task definitions for spatial brain experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class ThreeSlotBatch:
    """Container for a batch from the ThreeSlotWMTask."""

    inputs: torch.Tensor  # (time, batch, num_columns, content_dim) after embedding
    targets: torch.Tensor  # (batch, sequence_length)
    contexts: torch.Tensor  # (time, batch, context_dim)
    item_indices: torch.Tensor  # (batch, sequence_length)


class ThreeSlotWMTask:
    """Three-slot ordered working-memory task.

    Slot identity is represented in the context vector; the control network
    learns to map these slots to spatial patches via beta stencils.
    """

    def __init__(self, num_items: int, sequence_length: int, num_columns: int, content_dim: int) -> None:
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.num_columns = num_columns
        self.content_dim = content_dim
        # context_dim: one-hot for sequence_length slots + one retrieval flag.
        self.context_dim = sequence_length + 1

        self.item_embeddings = torch.nn.Embedding(num_items, content_dim)

    def sample_batch(self, batch_size: int) -> ThreeSlotBatch:
        """Sample a batch of sequences and contexts."""
        device = next(self.item_embeddings.parameters()).device
        items = torch.randint(low=0, high=self.num_items, size=(batch_size, self.sequence_length), device=device)

        # Trial length: encode each slot, then a retrieval step.
        trial_steps = self.sequence_length + 1
        contexts = torch.zeros((trial_steps, batch_size, self.context_dim), device=device)
        inputs = torch.zeros((trial_steps, batch_size, self.num_columns, self.content_dim), device=device)

        # Encoding steps: present item embedding, set slot one-hot in context.
        for step in range(self.sequence_length):
            contexts[step, :, step] = 1.0
            emb = self.item_embeddings(items[:, step])  # (batch, content_dim)
            inputs[step] = emb.unsqueeze(1).expand(-1, self.num_columns, -1)

        # Retrieval context: final element in context vector.
        contexts[-1, :, -1] = 1.0  # retrieval cue
        targets = items.clone()
        return ThreeSlotBatch(inputs=inputs, targets=targets, contexts=contexts, item_indices=items)

    def vocab(self) -> Dict[int, str]:
        """Return a simple mapping of item ids to string labels."""
        return {i: str(i) for i in range(self.num_items)}
