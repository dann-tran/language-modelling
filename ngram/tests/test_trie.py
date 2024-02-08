import pytest

from ngram.trie import Trie


class TestTrie:
    def test_trie(self):
        trie: Trie[int] = Trie.make(5)
        assert (1,) not in trie
        trie[(1, 2, 3)] = 4
        assert (1,) not in trie
        assert (1, 2, 3) in trie
        assert trie[(1, 2, 3)] == 4
        with pytest.raises(IndexError):
            trie[(1, 2)]
        trie[(1, 2, 4, 2)] = 5
        assert (1, 2) not in trie
        assert (1, 2, 4) not in trie
        assert trie[(1, 2, 4, 2)] == 5
