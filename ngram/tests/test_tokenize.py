import pytest

from ngram.tokenize import tokenize


class TestTokenize:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                """Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.  Please buy me\ntwo of them.\nThanks.""",
                [
                    "good",
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
                    "\n",
                    "in",
                    "new",
                    "york",
                    ".",
                    "please",
                    "buy",
                    "me",
                    "\n",
                    "two",
                    "of",
                    "them",
                    ".",
                    "\n",
                    "thanks",
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
                    "help",
                    "!",
                    "!",
                    "he",
                    "said",
                    ":",
                    "help",
                    ",",
                    "help",
                    "?",
                    "!",
                ],
            ),
            ("The dog's leash", ["the", "dog", "'s", "leash"]),
            ("The planets' atmospheres", ["the", "planets", "'", "atmospheres"]),
            (
                "Can I ask you somethin'?",
                ["can", "i", "ask", "you", "somethin", "'", "?"],
            ),
            (
                '"\'Twas the night before Christmas," he said.',
                [
                    '"',
                    "'",
                    "twas",
                    "the",
                    "night",
                    "before",
                    "christmas",
                    ",",
                    '"',
                    "he",
                    "said",
                    ".",
                ],
            ),
            (
                "The first political philosopher to call himself an anarchist () was Pierre-Joseph Proudhon (1809-1865), marking the formal birth of anarchism in the mid-19th century.",
                [
                    "the",
                    "first",
                    "political",
                    "philosopher",
                    "to",
                    "call",
                    "himself",
                    "an",
                    "anarchist",
                    "(",
                    ")",
                    "was",
                    "pierre",
                    "-",
                    "joseph",
                    "proudhon",
                    "(",
                    "1809",
                    "-",
                    "1865",
                    ")",
                    ",",
                    "marking",
                    "the",
                    "formal",
                    "birth",
                    "of",
                    "anarchism",
                    "in",
                    "the",
                    "mid",
                    "-",
                    "19",
                    "th",
                    "century",
                    ".",
                ],
            ),
            (
                "what does 's' represent?",
                ["what", "does", "'", "s", "'", "represent", "?"],
            ),
            (
                "I will say, 'that was a good night,' or is it!",
                [
                    "i",
                    "will",
                    "say",
                    ",",
                    "'",
                    "that",
                    "was",
                    "a",
                    "good",
                    "night",
                    ",",
                    "'",
                    "or",
                    "is",
                    "it",
                    "!",
                ],
            ),
        ],
    )
    def test_tokenize(self, text, expected):
        assert tokenize(text) == expected
