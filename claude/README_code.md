# CSC311 project proposal: data exploration code

Run from this folder with Python 3.10+, numpy, pandas, matplotlib and scikit-learn.
Put `csc311_project_training_202609.csv` in `data/` first (the data is not included in this zip).

1. `python split_data.py`: splits the 290 respondents into train / validation / test
   (202 / 44 / 44 respondents, 70/15/15), grouped by `id`, with seed 311.
   Writes `data/train.csv`, `data/val.csv` and `data/test.csv`.
2. `python explore.py`: data exploration on `data/train.csv` only.
   Writes the plots to `figures/` and the tables to `tables/`.
3. `python preliminary_check.py`: untuned, grouped 5-fold CV on the training split.
   This is a sanity check of the pipeline before tuning.

`preprocess.py` holds all cleaning and feature-encoding code (missing values, price parsing,
Likert/multi-hot encoding, bag-of-words). It uses only numpy and pandas, so it can be reused in `pred.py`.
