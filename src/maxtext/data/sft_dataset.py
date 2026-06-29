"""JSONL SFT dataset utilities for Tunix-free MaxText SFT training.

This module provides a small, dependency-light input pipeline that reads JSONL
records and converts them into causal LM training features:

- decoder_input_tokens
- decoder_target_tokens
- decoder_loss_weights
- decoder_positions
- decoder_segment_ids

Supported JSONL record formats:
1) {"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}
2) {"prompt": "...", "response": "..."}
3) {"text": "already formatted text"}

Notes:
- This intentionally avoids Tunix and keeps logic local to MaxText.
- The tokenizer is expected to be a Hugging Face tokenizer-like object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SFTExample:
  """Tokenized SFT example ready for batching."""

  decoder_input_tokens: np.ndarray
  decoder_target_tokens: np.ndarray
  decoder_loss_weights: np.ndarray
  decoder_positions: np.ndarray
  decoder_segment_ids: np.ndarray


def _read_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
  """Yields parsed JSON records from a JSONL file.

  Args:
    path: Local filesystem path to a JSONL file.

  Yields:
    Parsed dict records.
  """
  with Path(path).open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
      line = line.strip()
      if not line:
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON at line {i} in {path}: {e}") from e
      if not isinstance(record, dict):
        raise ValueError(f"Expected JSON object at line {i} in {path}, got {type(record)}")
      yield record


def _format_messages(messages: Sequence[Dict[str, Any]]) -> str:
  """Simple fallback chat formatter.

  If your tokenizer supports `apply_chat_template`, prefer that at call site.
  """
  parts: List[str] = []
  for m in messages:
    role = str(m.get("role", "user")).strip().lower()
    content = str(m.get("content", "")).strip()
    if not content:
      continue
    parts.append(f"<{role}>\n{content}")
  return "\n\n".join(parts)


def _extract_text_pair(
    record: Dict[str, Any],
    messages_field: str,
    prompt_field: str,
    response_field: str,
    text_field: str,
    tokenizer: Any,
    use_chat_template: bool,
) -> Tuple[str, Optional[str]]:
  """Extracts (full_text, response_text_or_none) from a record.

  Returns:
    full_text: Entire model input text.
    response_text_or_none: Assistant response if separable (used for masking).
  """
  if messages_field in record:
    messages = record[messages_field]
    if not isinstance(messages, list):
      raise ValueError(f"Expected list in field '{messages_field}', got {type(messages)}")

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
      # Build full conversation text.
      full_text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=False,
      )
      # Best effort: identify last assistant turn as response span.
      response = None
      for m in reversed(messages):
        if str(m.get("role", "")).lower() == "assistant":
          response = str(m.get("content", ""))
          break
      return full_text, response

    full_text = _format_messages(messages)
    response = None
    for m in reversed(messages):
      if str(m.get("role", "")).lower() == "assistant":
        response = str(m.get("content", ""))
        break
    return full_text, response

  if prompt_field in record and response_field in record:
    prompt = str(record[prompt_field])
    response = str(record[response_field])
    return f"{prompt}\n{response}", response

  if text_field in record:
    return str(record[text_field]), None

  raise ValueError(
      "Record did not match supported schemas. "
      f"Expected one of: '{messages_field}', ('{prompt_field}' + '{response_field}'), '{text_field}'."
  )


def _tokenize(tokenizer: Any, text: str, max_length: int) -> List[int]:
  """Tokenizes text to a fixed max length without adding extra framework deps."""
  encoded = tokenizer(
      text,
      truncation=True,
      max_length=max_length,
      add_special_tokens=True,
  )
  input_ids = encoded.get("input_ids")
  if not input_ids:
    # Ensure at least one token to avoid empty-array edge cases.
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
      eos = 0
    input_ids = [eos]
  return input_ids


def _build_features(
    token_ids: Sequence[int],
    max_length: int,
    pad_id: int,
    response_start_token_idx: Optional[int],
    train_on_prompt: bool,
) -> SFTExample:
  """Builds causal LM train features from token IDs."""
  # Pad/truncate to exact length.
  ids = list(token_ids[:max_length])
  if len(ids) < max_length:
    ids.extend([pad_id] * (max_length - len(ids)))

  ids_arr = np.asarray(ids, dtype=np.int32)

  decoder_input_tokens = ids_arr
  # Left-shift targets for next-token prediction.
  decoder_target_tokens = np.roll(ids_arr, -1)
  decoder_target_tokens[-1] = pad_id

  # Base loss mask: non-pad targets only.
  loss_weights = (decoder_target_tokens != pad_id).astype(np.float32)

  # Optional prompt masking.
  if (not train_on_prompt) and response_start_token_idx is not None:
    # Tokens before response start are prompt/context; mask them out.
    # Clamp to sequence range.
    start = int(max(0, min(max_length, response_start_token_idx)))
    loss_weights[:start] = 0.0

  decoder_positions = np.arange(max_length, dtype=np.int32)
  decoder_segment_ids = (decoder_input_tokens != pad_id).astype(np.int32)

  return SFTExample(
      decoder_input_tokens=decoder_input_tokens,
      decoder_target_tokens=decoder_target_tokens,
      decoder_loss_weights=loss_weights,
      decoder_positions=decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
  )


def iter_sft_examples(
    *,
    path: str,
    tokenizer: Any,
    max_length: int,
    pad_id: Optional[int] = None,
    messages_field: str = "messages",
    prompt_field: str = "prompt",
    response_field: str = "response",
    text_field: str = "text",
    use_chat_template: bool = True,
    train_on_prompt: bool = False,
) -> Generator[SFTExample, None, None]:
  """Yields tokenized SFT examples from JSONL.

  Args:
    path: JSONL file path.
    tokenizer: HuggingFace-like tokenizer.
    max_length: Sequence length.
    pad_id: Padding token id. If None, inferred from tokenizer.
    messages_field/prompt_field/response_field/text_field: Schema fields.
    use_chat_template: Use tokenizer.apply_chat_template when available.
    train_on_prompt: Whether to include prompt tokens in loss.
  """
  if pad_id is None:
    pad_id = getattr(tokenizer, "pad_token_id", None)
  if pad_id is None:
    pad_id = getattr(tokenizer, "eos_token_id", 0)

  for record in _read_jsonl(path):
    full_text, response_text = _extract_text_pair(
        record,
        messages_field=messages_field,
        prompt_field=prompt_field,
        response_field=response_field,
        text_field=text_field,
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
    )

    token_ids = _tokenize(tokenizer, full_text, max_length=max_length)

    response_start_token_idx: Optional[int] = None
    if response_text and not train_on_prompt:
      # Best-effort prompt masking by locating response text token boundary.
      # We tokenize prefix-only text and treat its length as response start.
      split_idx = full_text.rfind(response_text)
      if split_idx > 0:
        prefix = full_text[:split_idx]
        prefix_ids = _tokenize(tokenizer, prefix, max_length=max_length)
        response_start_token_idx = len(prefix_ids)

    yield _build_features(
      token_ids=token_ids,
      max_length=max_length,
      pad_id=pad_id,
      response_start_token_idx=response_start_token_idx,
      train_on_prompt=train_on_prompt,
    )


def batch_examples(examples: Iterable[SFTExample], batch_size: int) -> Generator[Dict[str, np.ndarray], None, None]:
  """Batches SFT examples into MaxText-style batch dicts."""
  cur: List[SFTExample] = []
  for ex in examples:
    cur.append(ex)
    if len(cur) < batch_size:
      continue

    yield {
        "decoder_input_tokens": np.stack([e.decoder_input_tokens for e in cur], axis=0),
        "decoder_target_tokens": np.stack([e.decoder_target_tokens for e in cur], axis=0),
        "decoder_loss_weights": np.stack([e.decoder_loss_weights for e in cur], axis=0),
        "decoder_positions": np.stack([e.decoder_positions for e in cur], axis=0),
        "decoder_segment_ids": np.stack([e.decoder_segment_ids for e in cur], axis=0),
    }
    cur = []

  if cur:
    # Drop remainder by default to keep shape static for JIT callers.
    return
