from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
import openwebtext_to_shards as mod


def _make_parquet(path: Path, texts: list[str]) -> None:
    table = pa.Table.from_arrays([pa.array(texts, type=pa.string())], names=["text"])
    pq.write_table(table, path)


def test_smoke_single_shard(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _make_parquet(in_dir / "train-00000-of-00080.parquet", ["hello", "world"])
    cfg = mod.Config(
        input_dir=in_dir,
        output_dir=out_dir,
        source_glob="train-*.parquet",
        text_column="text",
        tokenizer="gpt2",
        tokens_per_shard=1024,
        max_files=None,
        limit_docs=None,
        append_eot=True,
        dtype="auto",
        overwrite=True,
        batch_size=32,
    )
    manifest = mod.run(cfg)

    assert manifest["total_documents"] == 2
    assert manifest["num_shards"] == 1
    shard_path = out_dir / "shard-000000.bin"
    assert shard_path.exists()

    enc = tiktoken.get_encoding("gpt2")
    expected = enc.encode_ordinary("hello") + [enc.eot_token]
    expected += enc.encode_ordinary("world") + [enc.eot_token]
    arr = np.fromfile(shard_path, dtype=np.uint16)
    assert arr.tolist() == expected

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["total_tokens"] == len(expected)
    assert meta["dtype"] == "uint16"


def test_split_multi_shards(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _make_parquet(
        in_dir / "train-00000-of-00080.parquet",
        ["a", "b", "c", "d", "e"],
    )
    cfg = mod.Config(
        input_dir=in_dir,
        output_dir=out_dir,
        source_glob="train-*.parquet",
        text_column="text",
        tokenizer="gpt2",
        tokens_per_shard=3,
        max_files=None,
        limit_docs=None,
        append_eot=True,
        dtype="auto",
        overwrite=True,
        batch_size=32,
    )
    manifest = mod.run(cfg)
    assert manifest["num_shards"] >= 2

    shard_files = sorted(out_dir.glob("shard-*.bin"))
    assert len(shard_files) == manifest["num_shards"]
    counts = [int(s["tokens"]) for s in manifest["shards"]]
    assert all(c <= 3 for c in counts)
