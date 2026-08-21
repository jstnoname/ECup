import argparse
import json
import os
import time
from typing import Literal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("RAYON_NUM_THREADS", "20")

from pathlib import Path

import polars as pl
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TOKENIZER_DIR = BASE_DIR / "tokenizer"

MAX_LEN = 256
BATCH_SIZE = 512
ATTR_CAP = 1500
DEVICE = torch.device("cuda")
DTYPE = torch.float16


def parse_args() -> tuple[Literal['items_path'], Literal['matches_path'], Literal['output_path']]:
    parser = argparse.ArgumentParser(description="Score product duplicate pairs.")
    parser.add_argument("--output_path", type=str, help="output file")
    parser.add_argument("--items_path", type=str, default=None, help="test items data path")
    parser.add_argument("--matches_path", type=str, default=None, help="test matches data path")

    args = parser.parse_args()
    return args.items_path, args.matches_path, args.output_path


def product_text(name, category, attributes, attr_cap=2000) -> str:
    try:
        attrs = json.loads(attributes)
        attr_text = " ".join(f"{k}: {v}" for k, v in attrs.items())[:attr_cap]
    except (TypeError, json.JSONDecodeError):
        attr_text = ""
    return f"Name: {name} Category: {category} Attributes: {attr_text}"


class CrossEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(0.1)
        self.clf = torch.nn.Linear(encoder.config.hidden_size, 1)

    def forward(self, **kwargs):
        cls = self.encoder(**kwargs).last_hidden_state[:, 0, :]
        return self.clf(self.dropout(cls)).squeeze(-1)


def build_pairs(items_path: str, matches_path: str) -> pl.DataFrame:
    items = pl.scan_parquet(items_path)
    matches = pl.read_parquet(matches_path)

    needed = pl.DataFrame({"id": pl.concat([matches["id1"], matches["id2"]]).unique()})
    texts = items \
        .join(needed, on="id", how="semi") \
        .unique(subset=["id"], keep="any") \
        .select(
            "id",
            pl.struct(["name", "category", "attributes"]).map_elements(
                lambda r: product_text(r["name"], r["category"], r["attributes"], attr_cap=ATTR_CAP),
                return_dtype=pl.String
            ).alias("text"),
        ).collect()
    pairs = matches \
        .join(texts, left_on="id1", right_on="id") \
        .rename({"text": "text1"}) \
        .join(texts, left_on="id2", right_on="id") \
        .rename({"text": "text2"}) \
        .select("id1", "id2", "text1", "text2") \
        .with_columns(pl.col("text1").fill_null(""), pl.col("text2").fill_null(""))

    if pairs.height != matches.height:
        print(f"WARNING: pair count changed: {matches.height} -> {pairs.height}")
    print(f"pairs: {pairs.height}, unique ids: {needed.height}")
    return pairs


def load_model():
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = CrossEncoder(AutoModel.from_config(config)).to(device=DEVICE, dtype=DTYPE)
    state = torch.load(MODEL_DIR / "student_state.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"model loaded on {DEVICE} ({DTYPE})")
    return model


def predict(model, tokenizer, text1: list[str], text2: list[str]) -> list[float]:
    n = len(text1)
    lengths = [len(a) + len(b) for a, b in zip(text1, text2)]
    order = sorted(range(n), key=lengths.__getitem__)
    scores = [0.0] * n
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            inputs = tokenizer(
                [text1[i] for i in idx],
                [text2[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            probs = model(**inputs).float().sigmoid().cpu().tolist()
            for pos, i in enumerate(idx):
                scores[i] = probs[pos]
            if (start // BATCH_SIZE) % 50 == 0:
                done = min(start + BATCH_SIZE, n)
                rate = done / (time.time() - t0)
                print(f"{done}/{n} ({rate:.0f} pairs/s)")
    print(f"inference: {time.time() - t0:.1f}s ({n / (time.time() - t0):.0f} pairs/s avg)")
    return scores


def main():
    time_start = time.time()
    args = parse_args()
    pairs = build_pairs(args.items_path, args.matches_path)
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    scores = predict(model, tokenizer, pairs["text1"].to_list(), pairs["text2"].to_list())
    out = pl.DataFrame({"id1": pairs["id1"], "id2": pairs["id2"], "predict": scores})
    out.write_csv(args.output_path)
    print(f"done: {out.height} rows -> {args.output_path}, total {time.time() - time_start:.1f}s")


if __name__ == "__main__":
    main()
