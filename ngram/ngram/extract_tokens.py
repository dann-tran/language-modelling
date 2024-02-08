from argparse import ArgumentParser
import os
from multiprocessing import Pool
from typing import Tuple

import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

from ngram.tokenize import tokenize


def save_tokens(filepath_pair: Tuple[str, str]):
    filepath_in, filepath_out = filepath_pair

    table_in = pq.read_table(filepath_in, columns=["id", "text"])
    tokens = []
    for text in table_in["text"]:
        text = text.as_py()
        tokens.append(tokenize(text))

    tokens = pa.array(tokens)
    table_out = pa.Table.from_arrays([table_in["id"], tokens], names=["id", "tokens"])
    pq.write_table(table_out, filepath_out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir_in", type=str, required=True)
    parser.add_argument("--dir_out", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.dir_out, exist_ok=True)
    filenames_exclude = os.listdir(args.dir_out)
    filenames = [
        f
        for f in os.listdir(args.dir_in)
        if (f.endswith(".parquet") and f not in filenames_exclude)
    ]
    filepaths_in = [os.path.join(args.dir_in, f) for f in filenames]
    filepaths_out = [os.path.join(args.dir_out, f) for f in filenames]

    with Pool(args.n_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(save_tokens, zip(filepaths_in, filepaths_out)),
            total=len(filenames),
        ):
            pass
