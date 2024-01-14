import os
import pandas as pd
from nltk.tokenize.treebank import TreebankWordTokenizer
from multiprocessing import Pool
from tqdm import tqdm

tokenizer = TreebankWordTokenizer()


def save_tokens(filepath: str):
    df = pd.read_parquet(filepath)
    df["tokens"] = df["text"].str.lower().apply(tokenizer.tokenize)
    df.to_parquet(filepath)
    del df


DIRPATH = "/Users/danntran/Repos/language-modelling/data/wikpedia-20231101.en"
FILENAMES = [fn for fn in os.listdir(DIRPATH) if fn.endswith(".parquet")]
FILEPATHS = [os.path.join(DIRPATH, fn) for fn in FILENAMES]

if __name__ == "__main__":
    with Pool(8) as pool:
        for _ in tqdm(
            pool.imap_unordered(save_tokens, FILEPATHS), total=len(FILEPATHS)
        ):
            pass
