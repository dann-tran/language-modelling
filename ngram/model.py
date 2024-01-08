import json
import os
from typing import Iterable
import itertools

import numpy as np

from nltk.tokenize.treebank import TreebankWordTokenizer
import pandas as pd
from tqdm import tqdm


class NGramModel:
    CONFIG_FILENAME = "config.json"
    VOCAB_FILENAME = "vocab.txt"
    MODEL_FILENAME = "model.npz"

    @classmethod
    def load(cls, input_dir: str | os.PathLike[str]):
        with open(os.path.join(input_dir, cls.CONFIG_FILENAME), "r") as f:
            n = json.load(f)["n"]

        with open(os.path.join(input_dir, cls.VOCAB_FILENAME), "r") as f:
            vocab = [x.strip() for x in f.readlines()]
            vocab = [x for x in vocab if x]

        prob_matrix = np.load(os.path.join(input_dir, cls.MODEL_FILENAME))

        return cls(vocab=vocab, prob_matrix=prob_matrix, n=n)

    def __init__(
        self, vocab: Iterable[str], prob_matrix: np.ndarray | None = None, n: int = 2
    ):
        if n < 2:
            raise ValueError("n must not be smaller than 2.")

        self.n = n
        self.vocab = tuple(vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}

        self.seqs = itertools.permutations(self.vocab, self.context_size)
        self.seq2idx = {seq: i for i, seq in enumerate(self.seqs)}

        self.freq_matrix = np.zeros((len(self.seqs), len(vocab)), dtype=np.uint32)
        if prob_matrix is not None:
            assert prob_matrix.shape == (len(self.seqs), len(self.vocab))
        self._prob_matrix = prob_matrix

    @property
    def prob_matrix(self):
        if self._prob_matrix is None:
            self._prob_matrix = self.freq_matrix / self.freq_matrix.sum(axis=1)[:, None]
        return self._prob_matrix

    def fit(self, documents: Iterable[Iterable[str]]):
        self._prob_matrix = None

        for doc in documents:
            for start in range(0, len(doc), self.n):
                ngram = doc[start : start + self.n]
                context_idx = self.seq2idx[ngram[:-1]]
                word_idx = self.word2idx[ngram[-1]]
                self.freq_matrix[context_idx, word_idx] += 1

    def geneate(self, input_tokens: Iterable[str]):
        pass

    def save(self, output_dir: str | os.PathLike[str]):
        with open(os.path.join(output_dir, self.CONFIG_FILENAME), "w") as f:
            json.dump(dict(model="ngram", n=self.n))

        with open(os.path.join(output_dir, self.VOCAB_FILENAME), "w") as f:
            f.write("\n".join(self.vocab))

        self.prob_matrix.savez(os.path.join(output_dir, self.MODEL_FILENAME))


class WikipediaDataLoader:
    def __init__(self, parquet_dir: str | os.PathLike[str]):
        filenames = os.listdir(parquet_dir)
        filenames.sort()

        self.parquet_filepaths = [
            os.path.join(parquet_dir, filename)
            for filename in filenames
            if filename.endswith("parquet")
        ]

    def __len__(self):
        return len(self.parquet_filepaths)

    def __getitem__(self, idx: int):
        filepath = self.parquet_filepaths[idx]
        df = pd.read_parquet(filepath)
        return df["text"]


class Trainer:
    def __init__(
        self,
        model: NGramModel,
        dataloader: WikipediaDataLoader,
        tokenizer: TreebankWordTokenizer = TreebankWordTokenizer(),
    ):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer

    def train(self):
        for docs in tqdm(self.dataloader):
            self.model.fit(docs.apply(self.tokenizer.tokenize))
