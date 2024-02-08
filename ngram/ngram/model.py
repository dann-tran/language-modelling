import dataclasses
import json
import os
from typing import Iterable, List, Optional

from ngram.trie import Trie
from ngram.tokenize import Tokenizer


class NGramModel:
    CONFIG_FILENAME = "config.json"
    TOKENIZER_FILENAME = "tokenizer.json"
    FREQ_MATRIX_FILENAME = "freq_matrix.json"

    @classmethod
    def load(cls, input_dir: str | os.PathLike[str]):
        with open(os.path.join(input_dir, cls.CONFIG_FILENAME), "r") as f:
            config = json.load(f)
        tokenizer = Tokenizer.load(os.path.join(input_dir, cls.TOKENIZER_FILENAME))
        with open(os.path.join(input_dir, cls.FREQ_MATRIX_FILENAME), "r") as f:
            freq_matrix = json.load(f)
        return cls(
            n=config["n"],
            tokenizer=tokenizer,
            freq_matrix=freq_matrix,
        )

    def __init__(
        self,
        n: int,
        tokenizer: Tokenizer,
        freq_matrix: Optional[Trie[int]] = None,
    ):
        if n < 2:
            raise ValueError("n must not be smaller than 2.")

        self.n = n
        self.tokenizer = tokenizer
        if freq_matrix is None:
            self.freq_matrix: Trie[int] = Trie.make(len(tokenizer))
        else:
            assert len(tokenizer) == freq_matrix.vocab_size
            self.freq_matrix = freq_matrix

    def fit(self, documents: Iterable[List[int]]):
        for doc in documents:
            for start in range(0, len(doc) - self.n + 1):
                ngram = doc[start : start + self.n]
                if ngram not in self.freq_matrix:
                    self.freq_matrix[ngram] = 1
                else:
                    self.freq_matrix[ngram] += 1

    def geneate(self, input_tokens: Iterable[str]):
        pass

    def save(self, output_dir: str | os.PathLike[str]):
        with open(os.path.join(output_dir, self.CONFIG_FILENAME), "w") as f:
            json.dump(dict(model="ngram", n=self.n), f, indent=4)
        self.tokenizer.save(os.path.join(output_dir, self.TOKENIZER_FILENAME))
        with open(os.path.join(output_dir, self.FREQ_MATRIX_FILENAME), "w") as f:
            json.dump(dataclasses.asdict(self.freq_matrix), f, indent=4)
