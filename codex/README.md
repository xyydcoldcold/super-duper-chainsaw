# CSC311 proposal

Submit **proposal.pdf** and **code.zip** to the MarkUs `project_proposal` assignment.
The original official templates and `pred_example.py` have been preserved.

## What is complete

- The proposal contains two pages of completed Data Exploration, three pages of
  Planned Methodology, a title page, and one page of references/AI assistance.
- All reported feature statistics use the training respondents only.
- The source, preprocessing, full aggregate evidence, and verification code are
  included in `code.zip`. Raw survey data and the virtual environment are excluded.
- `models.py` implements the initial optional tuning plan. Full searches and
  validation/test scoring have **not** been run. The model compatibility checks
  use a fold within the training set and do not select a winner.
- `pred_example.py` is a separate official interface example, not this proposal's
  model implementation. No final restricted `pred.py` is claimed here.

## Reproduce the analysis

Tested with Python 3.13 and the exact package versions in `requirements.txt`.
From this directory (or after extracting `code.zip`):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/explore.py --data data/csc311_project_training_202609.csv --out analysis
.venv/bin/python code/verify.py --data data/csc311_project_training_202609.csv --out analysis/verification.json
```

Supply the official CSV locally at the path above, or replace `--data` with its
location. The full working folder contains a local data copy for convenience;
the ZIP intentionally does not contain the `data/` folder. No script downloads
data or uses network access.

The split is independent of row order: sort respondent IDs, shuffle with NumPy
`default_rng(311)`, take 44 test IDs, then 44 validation IDs, leaving 202 training
IDs. Exact assignments are saved in `analysis/split_manifest.csv`. Do not change
the seed after seeing model performance. Only identifiers and row counts are
recorded for validation/test; no feature or label summaries are produced for them.

## Optional later model work

```sh
.venv/bin/python code/models.py --data data/csc311_project_training_202609.csv --out model_results
```

This runs the initial 24 candidates x 5 respondent folds for each of three
families, then evaluates each tuned family on validation. It deliberately has no
test-scoring command. The planned manual review of boundary optima and any
extra ablations must happen using training folds before validation is opened;
edit/log any revised grids first. Do not repeatedly rerun validation while
changing representations. The PDF explains the selection rule and final
one-time test procedure. Prediction export is a later milestone.

## Files

- `code/data.py`: exact official CSV headers, input aliases and group split.
- `code/features.py`: explicit monetary parser and fold-local feature transformer.
- `code/explore.py`: all training-only summary tables, associations and figures.
- `code/models.py`: optional, planned grouped model search.
- `code/verify.py`: parser, leakage, unknown-input and three-model compatibility checks.
- `analysis/feature_summary.csv`: missingness and distributions for all 12 inputs.
- `analysis/numeric_by_class.csv`: observed numeric means, SDs, medians and counts.
- `analysis/multiselect_by_class.csv`: option selection rates for each class.
- `analysis/text_associations.csv`: term support, class presence and exploratory
  chi-square rankings; these are not corrected hypothesis-test conclusions.
- `analysis/distributions.json`, `all_feature_distributions.pdf`, and
  `numeric_correlations.pdf`: full training distribution evidence.
- `analysis/metrics.json`, `split_metadata.json`, `verification.json`: computed
  dimensions, source SHA-256, split sizes and passed behavioral checks.
- `proposal.tex` and `proposal_references.bib`: editable PDF sources.
- `REQUIREMENTS_CHECK.md`: map from official requirements to implementation.

## Build the PDF

Use the full working folder, which contains the LaTeX sources; the code-only ZIP
does not include those PDF sources.

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error proposal.tex
```

The source follows the official single-column, 12-point, one-inch-margin layout.
Latin Modern is used because the local BasicTeX installation lacks the template's
XCharter/newtx/inconsolata fonts. The section length limits remain unchanged.
The guiding-question and placeholder instructions have been removed from the
submitted report. AI assistance is disclosed in the PDF; no individual teammate
contribution has been invented. The group identifier is the observed MarkUs
repository identifier `group_62629`.

Nothing has been uploaded to MarkUs by this task.
