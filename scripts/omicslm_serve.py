#!/usr/bin/env python3
"""OmicsLM inference server — FastAPI endpoint with omics vector injection.

Loads a trained OmicsLM checkpoint (HF safetensors format) and serves
a chat-style API that accepts omics vectors alongside text prompts.

Usage:
    python scripts/omicslm_serve.py \
        --model_path /tmp/omicslm_hf_roundtrip \
        --norm_stats /tmp/omics_norm_stats.npz \
        --port 8000

    # Client:
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "What lineage is this cell line? <omics>",
            "omics_vector": [0.12, 4.63, ...],
            "max_new_tokens": 32
        }'
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


OMICS_TOKEN = "<omics>"


class OmicsLMForCausalLM(nn.Module):
    """Qwen3 backbone + omics projection layer."""

    def __init__(self, model_path: str):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = AutoConfig.from_pretrained(model_path)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.backbone = self.backbone.to(torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.omics_token_id = self.tokenizer.convert_tokens_to_ids(OMICS_TOKEN)

        from safetensors import safe_open
        with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
            if "model.omics_projection.weight" in f.keys():
                weight = f.get_tensor("model.omics_projection.weight")
                bias = f.get_tensor("model.omics_projection.bias")
                self.omics_dim = weight.shape[0]
                self.hidden_size = weight.shape[1]
                self.omics_projection = nn.Linear(self.omics_dim, self.hidden_size, bias=True)
                self.omics_projection.weight.data = weight.T.contiguous()
                self.omics_projection.bias.data = bias
                dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                self.omics_projection = self.omics_projection.to(dtype=dtype, device=self.device)
                print(f"Loaded omics projection: [{self.omics_dim}] -> [{self.hidden_size}]")
            else:
                self.omics_projection = None
                self.omics_dim = None
                print("WARNING: No omics projection found in checkpoint")

    @torch.no_grad()
    def generate(self, prompt: str, omics_vector: np.ndarray | None = None, max_new_tokens: int = 32) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        if omics_vector is not None and self.omics_projection is not None:
            embed_layer = self.backbone.model.embed_tokens
            inputs_embeds = embed_layer(input_ids)

            dtype = inputs_embeds.dtype
            omics_tensor = torch.tensor(omics_vector, dtype=dtype, device=self.device).unsqueeze(0)
            projected = self.omics_projection(omics_tensor.to(self.omics_projection.weight.dtype))
            projected = projected.to(dtype)

            omics_mask = (input_ids[0] == self.omics_token_id)
            positions = omics_mask.nonzero(as_tuple=True)[0]
            for i, pos in enumerate(positions):
                if i < projected.shape[0]:
                    inputs_embeds[0, pos] = projected[i]

            output = self.backbone.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            output = self.backbone.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def load_and_normalize_omics(path: str, index: int, norm_stats_path: str | None = None) -> np.ndarray:
    from array_record.python.array_record_module import ArrayRecordReader
    import tensorflow as tf

    reader = ArrayRecordReader(path)
    records = reader.read([index])
    example = tf.train.Example()
    example.ParseFromString(records[0])
    f = example.features.feature
    omics_key = "omics" if "omics" in f else "omics_inputs"
    feat = f[omics_key]
    if feat.bytes_list.value:
        vec = np.frombuffer(feat.bytes_list.value[0], dtype=np.float32).copy()
    elif feat.float_list.value:
        vec = np.array(feat.float_list.value, dtype=np.float32)
    else:
        raise ValueError("Empty omics field")

    if norm_stats_path:
        stats = np.load(norm_stats_path)
        vec = (np.log1p(vec) - stats["gene_mean"]) / max(float(stats["global_std"]), 1e-8)
    return vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--norm_stats", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--test", action="store_true", help="Run quick test instead of server")
    parser.add_argument("--test_record", default=None, help="ArrayRecord path for test")
    parser.add_argument("--test_indices", default="0,1,2", help="Comma-separated record indices")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = OmicsLMForCausalLM(args.model_path)

    if args.test:
        indices = [int(x) for x in args.test_indices.split(",")]
        for idx in indices:
            if args.test_record:
                from array_record.python.array_record_module import ArrayRecordReader
                import tensorflow as tf
                reader = ArrayRecordReader(args.test_record)
                records = reader.read([idx])
                example = tf.train.Example()
                example.ParseFromString(records[0])
                prompt = example.features.feature["prompt"].bytes_list.value[0].decode()
                expected = example.features.feature["completion"].bytes_list.value[0].decode()
                vec = load_and_normalize_omics(args.test_record, idx, args.norm_stats)
            else:
                prompt = "What lineage is this cell line? <omics>"
                expected = "?"
                vec = None

            # With omics
            output_with = model.generate(prompt, omics_vector=vec, max_new_tokens=16)
            # Without omics (baseline)
            prompt_no_omics = prompt.replace(" <omics>", "").replace("<omics> ", "").replace("<omics>", "")
            output_without = model.generate(prompt_no_omics, max_new_tokens=16)

            print(f"\n--- Record {idx} ---")
            print(f"Prompt:   {prompt}")
            print(f"Expected: {expected}")
            print(f"With omics:    {output_with}")
            print(f"Without omics: {output_without}")
        return

    # Server mode
    from fastapi import FastAPI
    from pydantic import BaseModel as PydanticBaseModel
    import uvicorn

    app = FastAPI(title="OmicsLM")

    class GenerateRequest(PydanticBaseModel):
        prompt: str
        omics_vector: list[float] | None = None
        max_new_tokens: int = 32

    class GenerateResponse(PydanticBaseModel):
        prompt: str
        output: str
        used_omics: bool

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        vec = np.array(req.omics_vector, dtype=np.float32) if req.omics_vector else None
        output = model.generate(req.prompt, omics_vector=vec, max_new_tokens=req.max_new_tokens)
        return GenerateResponse(prompt=req.prompt, output=output, used_omics=vec is not None)

    @app.get("/health")
    def health():
        return {"status": "ok", "omics_dim": model.omics_dim if model.omics_projection else None}

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
