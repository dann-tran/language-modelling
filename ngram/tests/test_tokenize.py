import pytest

from ngram.tokenize import tokenize


class TestTokenize:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                """Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.  Please buy me\ntwo of them.\nThanks.""",
                [
                    "Good",
                    "muffins",
                    "cost",
                    "$",
                    "3",
                    ".",
                    "88",
                    "(",
                    "roughly",
                    "3",
                    ",",
                    "36",
                    "euros",
                    ")",
                    "in",
                    "New",
                    "York",
                    ".",
                    "Please",
                    "buy",
                    "me",
                    "two",
                    "of",
                    "them",
                    ".",
                    "Thanks",
                    ".",
                ],
            ),
            (
                "hello, i can't feel; my feet! Help!! He said: Help, help?!",
                [
                    "hello",
                    ",",
                    "i",
                    "ca",
                    "n't",
                    "feel",
                    ";",
                    "my",
                    "feet",
                    "!",
                    "Help",
                    "!",
                    "!",
                    "He",
                    "said",
                    ":",
                    "Help",
                    ",",
                    "help",
                    "?",
                    "!",
                ],
            ),
            ("The dog's leash", ["The", "dog", "'s", "leash"]),
            ("The planets' atmospheres", ["The", "planets", "'", "atmospheres"]),
            (
                "Can I ask you somethin'?",
                ["Can", "I", "ask", "you", "somethin", "'", "?"],
            ),
            (
                '"\'Twas the night before Christmas," he said.',
                [
                    '"',
                    "'Twas",
                    "the",
                    "night",
                    "before",
                    "Christmas",
                    ",",
                    '"',
                    "he",
                    "said",
                    ".",
                ],
            ),
        ],
    )
    def test_tokenize(self, text, expected):
        assert tokenize(text) == expected
