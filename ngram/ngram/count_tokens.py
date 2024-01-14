from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool
import os

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm


def count_tokens(filepath: str):
    token2freq = defaultdict(lambda: 0)
    table = pq.read_table(filepath, columns=["tokens"])

    for tokens in table["tokens"]:
        for t in tokens:
            token2freq[t.as_py()] += 1

    return dict(token2freq)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dir_in", type=str, required=True)
    parser.add_argument("--file_out", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    filenames = [f for f in os.listdir(args.dir_in) if f.endswith(".parquet")]
    filepaths = [os.path.join(args.dir_in, f) for f in filenames]

    agg = dict()
    with Pool(args.n_workers) as pool:
        for token2freq in tqdm(
            pool.imap_unordered(count_tokens, filepaths), total=len(filepaths)
        ):
            agg.update(token2freq)

    tokens = []
    freqs = []
    for tok, freq in agg.items():
        tokens.append(tok)
        freqs.append(freq)

    tokens = pa.array(tokens)
    freqs = pa.array(freqs)
    table = pa.Table.from_arrays([tokens, freqs], names=["token", "frequency"])

    os.makedirs(os.path.dirname(args.file_out), exist_ok=True)
    pq.write_table(table, args.file_out)
