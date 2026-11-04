"""
Streaming UTF-8 Byte Dataset loader for FineWeb-Edu, TinyStories, or local text files.
No tokenizer required: text documents are converted directly into UTF-8 byte streams [0..255].
"""

from typing import Iterator, Optional
import torch
from torch.utils.data import IterableDataset, DataLoader


DATASET_ALIASES = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "tinystories": ("roneneldan/TinyStories", None, "text"),
}


class StreamingByteDataset(IterableDataset):
    """
    Streams documents from a HuggingFace dataset (or local text file), encodes each document
    into raw UTF-8 bytes separated by a newline/EOS byte (10), and yields fixed-length
    (seq_len,) input and target byte tensors.
    """

    def __init__(
        self,
        dataset_name: str = "fineweb-edu",
        seq_len: int = 1024,
        split: str = "train",
        local_file: Optional[str] = None,
        skip_docs: int = 0,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.split = split
        self.local_file = local_file
        self.skip_docs = skip_docs

    def _byte_stream(self) -> Iterator[int]:
        if self.local_file is not None:
            with open(self.local_file, "rb") as f:
                data = f.read()
            while True:
                for b in data:
                    yield b

        from datasets import load_dataset
        import time

        if self.dataset_name in DATASET_ALIASES:
            hf_path, hf_name, text_col = DATASET_ALIASES[self.dataset_name]
        else:
            hf_path, hf_name, text_col = self.dataset_name, None, "text"

        docs_seen = self.skip_docs
        backoff = 5.0
        shard_offset = docs_seen // 100000
        local_skip = docs_seen % 5000
        while True:
            try:
                base_ds = load_dataset(hf_path, name=hf_name, split=self.split, streaming=True)
                n_shards = getattr(base_ds, "n_shards", 1) or 1
                if n_shards > 1:
                    ds = base_ds.shard(num_shards=n_shards, index=shard_offset % n_shards)
                else:
                    ds = base_ds
                if local_skip > 0:
                    ds = ds.skip(local_skip)

                for row in ds:
                    docs_seen += 1
                    local_skip += 1
                    backoff = 5.0
                    text = row.get(text_col, "")
                    if not text:
                        continue
                    raw = text.encode("utf-8", errors="ignore")
                    for b in raw:
                        yield b
                    yield 10  # '\n' document separator byte

                # Advance to next shard when current shard finishes
                shard_offset += 1
                local_skip = 0
            except Exception as e:
                print(f"[StreamingByteDataset] Network/stream error after {docs_seen} docs: {e!r}. Retrying in {backoff:.0f}s...", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 60.0)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buffer = bytearray()
        chunk_bytes = self.seq_len + 1
        for b in self._byte_stream():
            buffer.append(b)
            if len(buffer) == chunk_bytes:
                tensor = torch.tensor(list(buffer), dtype=torch.long)
                yield tensor[:-1], tensor[1:]
                buffer.clear()


def build_byte_dataloader(
    dataset_name: str = "fineweb-edu",
    seq_len: int = 1024,
    batch_size: int = 16,
    local_file: Optional[str] = None,
    skip_docs: int = 0,
) -> DataLoader:
    ds = StreamingByteDataset(
        dataset_name=dataset_name,
        seq_len=seq_len,
        local_file=local_file,
        skip_docs=skip_docs,
    )
    use_workers = local_file is None and torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        num_workers=1 if use_workers else 0,
        prefetch_factor=8 if use_workers else None,
    )
