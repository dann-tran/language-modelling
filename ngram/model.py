from typing import Iterable

import numpy as np


class NGramTokenizer:
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<UNK>"
    SPECIAL_TOKENS = [BOS, EOS, UNK]

    def __init__(self, vocab: Iterable[str]):
        self.vocab = tuple(self.SPECIAL_TOKENS + list(vocab))
        self.w2id = {w: i for i, w in enumerate(self.vocab)}

    def _encode(self, word: str):
        if word not in self.w2id:
            word = self.UNK
        return self.w2id[word]

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [self._encode(w) for w in text.split()]
        if add_special_tokens:
            ids.insert(0, self.w2id[self.BOS])
            ids.append(self.w2id[self.EOS])
        return tuple(ids)

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False):
        words = [self.vocab[i] for i in ids]
        if skip_special_tokens:
            words = [w for w in words if w not in self.SPECIAL_TOKENS]
        return " ".join(words)


class NGramModel:
    def __init__(self, n: int = 2):
        if n < 2:
            raise ValueError("n must not be smaller than 2.")

        self.prob_matrix = np.array

    def fit(self, id_documents: np.ndarray):
        pass
