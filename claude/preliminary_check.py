"""
Preliminary sanity check of the planned pipeline (NOT the final tuning).

Runs 5-fold cross-validation grouped by respondent on the TRAINING split only,
with untuned default hyperparameters. The FeatureBuilder is re-fit inside each
fold, so vocabularies and scaling statistics never see the held-out fold.

Its only purpose is to confirm that (1) the features carry signal well above
the 33% chance level and (2) the three planned model families are in a
sensible range, before we spend time on tuning. The validation and test
splits are not touched.

Usage:
    python preliminary_check.py
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from preprocess import FeatureBuilder, encode_labels

SEED = 311

MODELS = {
    "majority baseline": lambda: DummyClassifier(strategy="most_frequent"),
    "softmax regression": lambda: LogisticRegression(C=1.0, max_iter=2000),
    "random forest": lambda: RandomForestClassifier(n_estimators=300,
                                                    random_state=SEED),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3,
                                 max_iter=500, early_stopping=False,
                                 random_state=SEED),
    "kNN (k=15)": lambda: KNeighborsClassifier(n_neighbors=15),
}


def main():
    df = pd.read_csv("data/train.csv")
    y = encode_labels(df["label"])
    groups = df["id"].to_numpy()
    cv = GroupKFold(n_splits=5)

    rows = []
    for name, make in MODELS.items():
        accs, f1s = [], []
        for tr, va in cv.split(df, y, groups):
            fb = FeatureBuilder(min_df=2).fit(df.iloc[tr])
            Xtr, Xva = fb.transform(df.iloc[tr]), fb.transform(df.iloc[va])
            model = make().fit(Xtr, y[tr])
            pred = model.predict(Xva)
            accs.append(accuracy_score(y[va], pred))
            f1s.append(f1_score(y[va], pred, average="macro"))
        rows.append({"model": name,
                     "cv_accuracy_mean": np.mean(accs),
                     "cv_accuracy_std": np.std(accs),
                     "cv_macro_f1_mean": np.mean(f1s)})
    out = pd.DataFrame(rows)
    out.to_csv("tables/preliminary_cv.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
