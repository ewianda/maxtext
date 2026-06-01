"""Gene panel extraction helpers for OmicsLM."""

from __future__ import annotations

from pathlib import Path

CANONICAL_GENE_TYPES = {"protein_coding", "Mt_rRNA", "Mt_tRNA"}


def _parse_attributes(raw_attributes: str) -> dict[str, str]:
  attributes: dict[str, str] = {}
  for field in raw_attributes.split(";"):
    field = field.strip()
    if not field:
      continue
    if " " not in field:
      continue
    key, value = field.split(" ", maxsplit=1)
    attributes[key] = value.strip().strip('"')
  return attributes


def extract_gene_panel(gtf_path: str | Path) -> list[str]:
  genes: list[str] = []
  seen: set[str] = set()
  with Path(gtf_path).open("r", encoding="utf-8") as handle:
    for line in handle:
      if not line or line.startswith("#"):
        continue
      fields = line.rstrip("\n").split("\t")
      if len(fields) < 9 or fields[2] != "gene":
        continue
      attributes = _parse_attributes(fields[8])
      gene_name = attributes.get("gene_name")
      gene_type = attributes.get("gene_type") or attributes.get("gene_biotype")
      if not gene_name or gene_name in seen:
        continue
      if gene_type in CANONICAL_GENE_TYPES:
        genes.append(gene_name)
        seen.add(gene_name)
  return genes


def write_gene_panel(genes: list[str], output_path: str | Path) -> None:
  Path(output_path).write_text("\n".join(genes) + "\n", encoding="utf-8")
