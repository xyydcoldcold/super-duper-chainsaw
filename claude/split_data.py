"""
Split the released data into train / validation / test sets.

Every respondent (the `id` column) answered the survey three times, once for
each sport. We split by respondent rather than by row, so that the same
person never appears in two splits. Otherwise a model could memorise a
person's writing style in the training set and look better on the
validation/test sets than it would on genuinely new people (the real test
set is written by TAs and instructors).

The test split is created first and is never opened by explore.py or used
for any tuning. It is only used once, at the very end, to estimate the
accuracy of the final model.

Usage:
    python split_data.py
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/csc311_project_training_202609.csv"
SEED = 311
TEST_FRAC = 0.15
VAL_FRAC = 0.15


def main():
    df = pd.read_csv(DATA_PATH)

    ids = np.sort(df["id"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)

    n_test = round(TEST_FRAC * len(ids))
    n_val = round(VAL_FRAC * len(ids))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])
    train_ids = set(ids[n_test + n_val:])

    splits = {
        "train": df[df["id"].isin(train_ids)],
        "val": df[df["id"].isin(val_ids)],
        "test": df[df["id"].isin(test_ids)],
    }
    for name, part in splits.items():
        part.to_csv(f"data/{name}.csv", index=False)
        counts = part["label"].value_counts().to_dict()
        print(f"{name:5s}: {part['id'].nunique():3d} respondents, "
              f"{len(part):3d} rows, labels {counts}")

    # sanity check: no respondent appears in more than one split
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)


if __name__ == "__main__":
    main()
