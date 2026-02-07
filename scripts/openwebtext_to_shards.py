#!/usr/bin/env python3
"""Convert OpenWebText parquet files into GPT token shard binaries.

Output format:
- `shard-XXXXXX.bin`: raw little-endian token ids (`uint16` or `uint32`)
- `meta.json`: manifest and tokenizer/config metadata
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import tiktoken


@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    source_glob: str
    text_column: str
    tokenizer: str
    tokens_per_shard: int
    max_files: int | None
    limit_docs: int | None
    append_eot: bool
    dtype: str
    overwrite: bool
    batch_size: int


class ShardWriter:
    def __init__(self, output_dir: Path, tokens_per_shard: int, dtype: np.dtype) -> None:
        self.output_dir = output_dir
        self.tokens_per_shard = tokens_per_shard
        self.dtype = np.dtype(dtype).newbyteorder("<")
        self.buf = np.empty(tokens_per_shard, dtype=self.dtype)
        self.buf_pos = 0
        self.shard_index = 0
        self.shards: list[dict[str, int | str]] = []

    def _flush(self, n_tokens: int) -> None:
        shard_name = f"shard-{self.shard_index:06d}.bin"
        shard_path = self.output_dir / shard_name
        self.buf[:n_tokens].tofile(shard_path)
        self.shards.append({"file": shard_name, "tokens": n_tokens})
        self.shard_index += 1

    def add(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        arr = np.asarray(token_ids, dtype=self.dtype)
        start = 0
        arr_len = arr.shape[0]
        while start < arr_len:
            space = self.tokens_per_shard - self.buf_pos
            take = min(space, arr_len - start)
            self.buf[self.buf_pos : self.buf_pos + take] = arr[start : start + take]
            self.buf_pos += take
            start += take
            if self.buf_pos == self.tokens_per_shard:
                self._flush(self.tokens_per_shard)
                self.buf_pos = 0

    def finish(self) -> None:
        if self.buf_pos > 0:
            self._flush(self.buf_pos)
            self.buf_pos = 0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Convert OpenWebText parquet to token shards for GPT-style LM training.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("~/data/lm/openwebtext/plain_text").expanduser(),
        help="Input directory containing parquet files (default: ~/data/lm/openwebtext/plain_text)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/data/lm/openwebtext_gpt2").expanduser(),
        help="Output directory for shard binaries and meta.json",
    )
    parser.add_argument(
        "--source-glob",
        default="train-*.parquet",
        help="Glob pattern for parquet source files",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Parquet text column name (default: text)",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="tiktoken encoding name (default: gpt2)",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=int,
        default=8_388_608,
        help="Token count per shard file",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional max parquet files to process",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Optional max documents to process (for smoke tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Row batch size when reading parquet",
    )
    parser.add_argument(
        "--no-eot",
        action="store_true",
        help="Do not append end-of-text token between documents",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "uint16", "uint32"],
        default="auto",
        help="Output token dtype",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory contents",
    )
    args = parser.parse_args()
    return Config(
        input_dir=args.input_dir.expanduser(),
        output_dir=args.output_dir.expanduser(),
        source_glob=args.source_glob,
        text_column=args.text_column,
        tokenizer=args.tokenizer,
        tokens_per_shard=args.tokens_per_shard,
        max_files=args.max_files,
        limit_docs=args.limit_docs,
        append_eot=not args.no_eot,
        dtype=args.dtype,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )


def discover_files(input_dir: Path, source_glob: str, max_files: int | None) -> list[Path]:
    files = sorted(input_dir.glob(source_glob))
    if max_files is not None:
        files = files[:max_files]
    return files


def resolve_dtype(config_dtype: str, vocab_size: int) -> np.dtype:
    if config_dtype == "uint16":
        return np.dtype("uint16")
    if config_dtype == "uint32":
        return np.dtype("uint32")
    if vocab_size <= 0xFFFF:
        return np.dtype("uint16")
    return np.dtype("uint32")


def iter_texts(parquet_file: Path, text_column: str, batch_size: int) -> Iterable[str]:
    pf = pq.ParquetFile(parquet_file)
    for batch in pf.iter_batches(columns=[text_column], batch_size=batch_size):
        arr = batch.column(0)
        for v in arr.to_pylist():
            if isinstance(v, str):
                yield v


def run(config: Config) -> dict[str, object]:
    files = discover_files(config.input_dir, config.source_glob, config.max_files)
    if not files:
        raise SystemExit(
            f"no parquet files matched in {config.input_dir} with {config.source_glob}",
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = config.output_dir / "meta.json"
    if meta_path.exists() and not config.overwrite:
        raise SystemExit(
            f"{meta_path} already exists. pass --overwrite to replace output",
        )
    if config.overwrite:
        for p in config.output_dir.glob("shard-*.bin"):
            p.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    enc = tiktoken.get_encoding(config.tokenizer)
    eot = enc.eot_token
    dtype = resolve_dtype(config.dtype, enc.n_vocab)
    max_token_id = int(np.iinfo(dtype).max)
    if enc.n_vocab - 1 > max_token_id:
        raise SystemExit(
            f"tokenizer vocab ({enc.n_vocab}) exceeds dtype capacity ({dtype})",
        )

    writer = ShardWriter(config.output_dir, config.tokens_per_shard, dtype)
    total_docs = 0
    total_tokens = 0
    processed_files = 0

    for i, parquet_path in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {parquet_path.name}")
        processed_files += 1
        for text in iter_texts(parquet_path, config.text_column, config.batch_size):
            token_ids = enc.encode_ordinary(text)
            if config.append_eot:
                token_ids.append(eot)
            writer.add(token_ids)
            total_docs += 1
            total_tokens += len(token_ids)
            if config.limit_docs is not None and total_docs >= config.limit_docs:
                break
        if config.limit_docs is not None and total_docs >= config.limit_docs:
            break

    writer.finish()
    manifest = {
        "dataset": "Skylion007/openwebtext",
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "source_glob": config.source_glob,
        "discovered_files": len(files),
        "processed_files": processed_files,
        "tokenizer": config.tokenizer,
        "n_vocab": enc.n_vocab,
        "eot_token": eot,
        "append_eot": config.append_eot,
        "dtype": str(dtype),
        "tokens_per_shard": config.tokens_per_shard,
        "total_documents": total_docs,
        "total_tokens": total_tokens,
        "num_shards": len(writer.shards),
        "shards": writer.shards,
    }
    meta_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "done: "
        + f"docs={total_docs} tokens={total_tokens} shards={len(writer.shards)} "
        + f"dtype={dtype} output={config.output_dir}",
    )
    return manifest


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
