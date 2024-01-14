from collections import defaultdict
import json
from multiprocessing import Pool
import pyarrow.parquet
import os

from tqdm import tqdm


DIRPATH = "/Users/danntran/Repos/language-modelling/data/wikpedia-20231101.en"
FILENAMES = [fn for fn in os.listdir(DIRPATH) if fn.endswith(".parquet")]
FILEPATHS = [os.path.join(DIRPATH, fn) for fn in FILENAMES]

TOKEN_FREQ_FILEPATH = os.path.join(DIRPATH, "token_freq.json")


def extract_token_freq(filepath: str):
    token2freq = defaultdict(lambda: 0)
    table = pyarrow.parquet.read_table(filepath, columns=["tokens"])

    for tokens in table["tokens"]:
        for t in tokens:
            token2freq[t.as_py()] += 1

    del table
    return dict(token2freq)


if __name__ == "__main__":
    agg = defaultdict(lambda: 0)
    with Pool(8) as pool:
        for token2freq in tqdm(
            pool.imap_unordered(extract_token_freq, FILEPATHS), total=len(FILEPATHS)
        ):
            agg.update(token2freq)

    agg = dict(sorted(agg.items()))
    with open(TOKEN_FREQ_FILEPATH, "w") as f:
        json.dump(agg, f, indent=4)
