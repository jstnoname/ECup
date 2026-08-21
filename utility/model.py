from __future__ import annotations

import json

import torch


def product_text(name, category, attributes, attr_cap: int = 1500) -> str:
    """
    Build product text like org baseline: Name: ... Category: ... Attributes: ...
    Attributes string is capped to `attr_cap` chars.
    """
    try:
        attrs = json.loads(attributes)
        attr_text = " ".join(f"{k}: {v}" for k, v in attrs.items())[:attr_cap]
    except (TypeError, json.JSONDecodeError):
        attr_text = ""
    return f"Name: {name} Category: {category} Attributes: {attr_text}"


class CrossEncoder(torch.nn.Module):
    """Bare encoder + [CLS] pooling + Dropout + Linear(1). Raw logits out (for BCEWithLogitsLoss)."""

    def __init__(self, encoder, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(dropout)
        self.clf = torch.nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, **kwargs):
        outputs = self.encoder(**kwargs)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.clf(self.dropout(cls)).squeeze(-1)

    def get_active_params(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param
