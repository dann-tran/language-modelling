from dataclasses import dataclass
from typing import Generic, Iterable, List, Optional, TypeVar

V = TypeVar("V")


@dataclass
class Trie(Generic[V]):
    vocab_size: int
    children: List[Optional["Trie[V]"]]
    value: Optional[V]

    def __init__(self, vocab_size: int, value: Optional[V] = None):
        self.vocab_size = vocab_size
        self.children = [None for _ in range(self.vocab_size)]
        self.value = value

    @classmethod
    def make(cls, vocab_size: int):
        return cls(vocab_size)

    def __getitem__(self, key: Iterable[int]):
        node = self
        for k in key:
            node = node.children[k]
            if node is None:
                raise IndexError(key)
        if node.value is None:
            raise IndexError(key)
        return node.value

    def __setitem__(self, key: Iterable[int], value: V):
        node = self
        for k in key:
            if node.children[k] is None:
                node.children[k]: Trie[V] = Trie.make(self.vocab_size)
            node = node.children[k]
        node.value = value

    def __contains__(self, key: Iterable[int]):
        node = self
        for k in key:
            node = node.children[k]
            if node is None:
                return False
        return node.value is not None
