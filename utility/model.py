import json
import os
import shutil
from pathlib import Path

import mlflow
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedConfig, AutoModel


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
        cls = self.encoder(**kwargs).last_hidden_state[:, 0, :]
        return self.clf(self.dropout(cls)).squeeze(-1)

    def get_active_params(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param


class HFCrossEncoder(PreTrainedModel):
    _supports_sdpa = True

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.encoder = AutoModel.from_config(config)
        self.dropout = torch.nn.Dropout(getattr(config, "dropout", 0.1))
        self.clf = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, **kwargs):
        cls = self.encoder(**kwargs).last_hidden_state[:, 0, :]
        return self.clf(self.dropout(cls)).squeeze(-1)


def download_model_from_mlflow(run_id: str, save_path: str | Path, artifact_path: str) -> None:
    """
    Download a pickled CrossEncoder artifact from MLflow and repack it as a fp16
    state_dict + encoder config into `<save_path>/model`.
    Temp files go to `<save_path>/temp` and are removed afterwards.
    """
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    save_path = Path(save_path)
    model_dir = save_path / "model"
    temp_dir = save_path / "temp"
    model_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=str(temp_dir),
        )
        print(f"downloaded: {local}")

        model = torch.load(Path(local) / "data" / "model.pth", map_location="cpu", weights_only=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"unpickled: {type(model).__name__}, {n_params / 1e6:.0f}M params")

        state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        out = model_dir / "student_state.pt"
        torch.save(state, out)
        print(f"state_dict: {out} ({out.stat().st_size / 1e6:.0f} MB), {len(state)} tensors")
        model.encoder.config.save_pretrained(model_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def download_tokenizer_from_hf(name: str, save_path: str | Path) -> None:
    """
    Download a tokenizer from the HF Hub and save it flat into `<save_path>/tokenizer`.
    Temp files go to `<save_path>/temp` and are removed afterwards.
    """
    save_path = Path(save_path)
    tokenizer_dir = save_path / "tokenizer"
    temp_dir = save_path / "temp"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=str(temp_dir))
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"tokenizer saved: {sorted(p.name for p in tokenizer_dir.iterdir())}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
