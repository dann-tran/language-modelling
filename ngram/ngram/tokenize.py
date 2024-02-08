import json
from typing import Dict, Iterable


PUNCTUATIONS = set('.?!,:;(){}[]"-–—@#$%&|=~></\\')  # hyphen, en-dash, em-dash
CONTRACTIONS = ["n't", "'ve", "'d", "'ll", "'s", "'re", "'m"]
POSSESSIVE_APOS = ["'s", "'"]
SUFFIX_TOKENS = CONTRACTIONS + POSSESSIVE_APOS


def tokenize(text: str):
    # replace non-ASCII characters with whitespaces
    text = "".join([c.lower() if c.isascii() else " " for c in text])
    tokens = []
    idx = 0

    while idx < len(text):
        if text[idx].isspace():
            if text[idx] == "\n":
                tokens.append("\n")
            idx += 1
            continue

        if text[idx] in PUNCTUATIONS:
            tokens.append(text[idx])
            idx += 1
            continue

        idx_forward = idx + 1

        if text[idx].isnumeric():
            while idx_forward < len(text) and text[idx_forward].isnumeric():
                idx_forward += 1
            tokens.append(text[idx:idx_forward])
            idx = idx_forward
            continue

        while (
            idx_forward < len(text)
            and not text[idx_forward].isspace()
            and not text[idx_forward] in PUNCTUATIONS
        ):
            idx_forward += 1
        # text[idx_forward]: either end of text, space, or punctuation

        found_suffix = False
        for suffix in SUFFIX_TOKENS:
            if text.endswith(suffix, idx, idx_forward):
                tok = text[idx : idx_forward - len(suffix)]
                if tok.startswith("'"):
                    tokens.append("'")
                    tok = tok[1:]
                if tok:
                    tokens.append(tok)
                tokens.append(suffix)
                idx = idx_forward
                found_suffix = True
                break

        if found_suffix:
            continue

        tok = text[idx:idx_forward]
        if tok.startswith("'"):
            tokens.append("'")
            tok = tok[1:]

        if tok.endswith("'"):
            tok = tok[:-1]
            if tok:
                tokens.append(tok)
            tokens.append("'")
        elif tok:
            tokens.append(tok)

        idx = idx_forward

    return tokens


class Tokenizer:
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<UNK>"
    SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(
        self, tok2id: Dict[str, int], bos_id: int = 0, eos_id: int = 1, unk_id: int = 2
    ):
        assert tok2id[self.BOS_TOKEN] == bos_id
        assert tok2id[self.EOS_TOKEN] == eos_id
        assert tok2id[self.UNK_TOKEN] == unk_id
        self.tok2id = tok2id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            tok2id = json.load(f)["vocab"]
        return cls(tok2id)

    @classmethod
    def from_nonspecial_tokens(cls, vocab: Iterable[str]):
        vocab = list(vocab)
        vocab.sort()
        tokens = cls.SPECIAL_TOKENS + vocab
        tok2id = {tok: i for i, tok in enumerate(tokens)}
        return cls(tok2id)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump({"vocab": self.tok2id}, f, indent=4)

    def __len__(self):
        return len(self.tok2id)

    def tokenize(self, text: str):
        ids = [self.bos_id]

        for tok in tokenize(text):
            ids.append(self.tok2id.get(tok, self.unk_id))

        ids.append(self.eos_id)

        return ids
