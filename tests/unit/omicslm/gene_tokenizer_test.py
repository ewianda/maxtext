from __future__ import annotations

import numpy as np

from omicslm.gene_tokenizer import augment_tokenizer_with_gene_tokens, extend_embedding_table


class FakeTokenizer:
  def __init__(self):
    self._vocab = {"BR": 0, "CA": 1, "1": 2, "TP": 3, "53": 4, "EXISTING": 5}

  def get_vocab(self):
    return dict(self._vocab)

  def encode(self, text, add_special_tokens=False):
    if text == "BRCA1":
      return [0, 1, 2]
    if text == "TP53":
      return [3, 4]
    if text == "EXISTING":
      return [5]
    raise KeyError(text)

  def add_tokens(self, tokens):
    start = len(self._vocab)
    for index, token in enumerate(tokens):
      self._vocab[token] = start + index
    return len(tokens)


def test_augment_tokenizer_with_gene_tokens_initializes_from_subword_means():
  tokenizer = FakeTokenizer()
  embedding_table = np.arange(18, dtype=np.float32).reshape(6, 3)

  augmentation = augment_tokenizer_with_gene_tokens(tokenizer, embedding_table, ["BRCA1", "TP53", "EXISTING"])
  extended = extend_embedding_table(embedding_table, augmentation.new_embeddings)

  np.testing.assert_allclose(augmentation.new_embeddings[0], embedding_table[[0, 1, 2]].mean(axis=0))
  np.testing.assert_allclose(augmentation.new_embeddings[1], embedding_table[[3, 4]].mean(axis=0))
  assert augmentation.added_tokens == ("BRCA1", "TP53")
  assert extended.shape == (8, 3)
  assert tokenizer.get_vocab()["BRCA1"] == 6
  assert tokenizer.get_vocab()["TP53"] == 7
