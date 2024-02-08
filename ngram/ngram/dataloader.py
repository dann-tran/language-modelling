import os
from typing import List, Optional

import pyarrow.parquet as pq
import pyarrow as pa

from ngram.tokenize import Tokenizer


class DataLoader:
    def __init__(
        self,
        parquet_dir: str | os.PathLike[str],
        vocab_file: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.parquet_dir = parquet_dir
        self.filenames = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
        self.filenames.sort()
        self.cache_dir = os.path.join(parquet_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if use_cache:
            vocab_file_cache = os.path.join(self.cache_dir, "vocab.json")
            tokenizer = Tokenizer.load(vocab_file_cache)
        else:
            assert vocab_file is not None, "vocab_file is required"
            tokenizer = Tokenizer.load(vocab_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> List[List[int]]:
        filename = self.filenames[idx]
        cache_filepath = os.path.join(self.cache_dir, filename)
        if os.path.exists(cache_filepath):
            table = pq.read_table(cache_filepath, columns=["input_ids"])
            data = table["input_ids"].as_py()
        else:
            filepath = os.path.join(self.parquet_dir, filename)
            table = pq.read_table(filepath, columns=["id", "text"])
            data = [self.tokenizer.tokenize(text.as_py()) for text in table["text"]]
            table_out = pa.Table.from_arrays(
                [table["id"], pa.array(data)], names=["id", "input_ids"]
            )
            pq.write_table(table_out, cache_filepath)
        return data

    def __iter__(self):
        for i in len(self):
            yield self[i]
        raise StopIteration
