PUNCTUATIONS = set('.?!,:;(){}[]"-–—@#$%&')  # hyphen, en-dash, em-dash
CONTRACTIONS = ["n't", "'ve", "'d", "'ll", "'s", "'re", "'m"]
POSSESSIVE_APOS = ["'s", "'"]
SUFFIX_TOKENS = CONTRACTIONS + POSSESSIVE_APOS


def tokenize(text: str):
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
                if tok:
                    tokens.append(tok)
                tokens.append(suffix)
                idx = idx_forward
                found_suffix = True
                break

        if found_suffix:
            continue

        tok = text[idx:idx_forward]
        if tok:
            tokens.append(tok)
        idx = idx_forward

    return tokens
