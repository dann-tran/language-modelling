from argparse import ArgumentParser
import json
import os

import pandas as pd


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_in", type=str, required=True)
    parser.add_argument("--file_out", type=str, required=True)
    parser.add_argument("--freq_cutoff", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.file_in, columns=["token", "frequency"], engine="pyarrow")
    restricted_df = df[df["frequency"] >= args.freq_cutoff]
    vocab = restricted_df["token"].to_list()
    vocab.sort()

    data = dict(
        statistics=dict(
            size=len(restricted_df),
            coverage=restricted_df["frequency"].sum() / df["frequency"].sum(),
        ),
        vocab=vocab,
    )

    os.makedirs(os.path.dirname(args.file_out), exist_ok=True)
    with open(args.file_out, "w") as f:
        json.dump(data, f, indent=4)
