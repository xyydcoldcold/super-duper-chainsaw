"""
Data exploration for the CSC311 sports survey.

Reads ONLY data/train.csv (created by split_data.py). The validation and test
splits are never opened here, so nothing we learn while exploring can leak
into model selection or the final test estimate.

Outputs
  figures/*.pdf       plots used in the proposal
  tables/*.csv        the numbers behind every claim in the report
  stdout              a short text summary

Usage:
    python split_data.py      # once
    python explore.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import (LABELS, LIKERT_COLS, LIKERT_MAP, MULTI_COLS,
                        MULTI_OPTIONS, PRICE_CAP, Q_EASE, Q_PRICE, SHORT,
                        TEXT_COLS, FeatureBuilder, normalize_missing,
                        parse_price, tokenize)

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

COLORS = {"Baseball": "#1f77b4", "Basketball": "#ff7f0e", "Soccer": "#2ca02c"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def section(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def main():
    raw = pd.read_csv("data/train.csv")
    df = normalize_missing(raw)
    y = df["label"]

    # ------------------------------------------------------------------
    section("1. Size, respondents and class balance")
    print(f"rows = {len(df)}, respondents = {df['id'].nunique()}, "
          f"rows per respondent = {df.groupby('id').size().unique()}")
    print(y.value_counts().to_string())
    same_three = df.groupby("id")["label"].apply(lambda s: sorted(s) == LABELS).all()
    print("every respondent answered for all three sports:", same_three)

    # ------------------------------------------------------------------
    section("2. Feature types and missing values")
    kinds = {Q_EASE: "numeric (0-10)", Q_PRICE: "free text -> number"}
    kinds.update({c: "Likert (5 levels)" for c in LIKERT_COLS})
    kinds.update({c: "multi-select" for c in MULTI_COLS})
    kinds.update({c: "free text" for c in TEXT_COLS})
    rows = []
    for col, short in SHORT.items():
        miss = df[col].isna()
        rows.append({
            "feature": short, "type": kinds[col],
            "missing_%": 100 * miss.mean(),
            "n_unique_raw": raw[col].nunique(),
            **{f"missing_%_{c}": 100 * miss[y == c].mean() for c in LABELS},
        })
    summary = pd.DataFrame(rows)
    price = df[Q_PRICE].map(parse_price)
    unparsed = df[Q_PRICE].notna() & price.isna()
    summary.loc[summary.feature == "price", "unparseable_%"] = 100 * unparsed.mean()
    summary.to_csv("tables/feature_summary.csv", index=False)
    print(summary.round(1).to_string(index=False))
    all_empty = df[list(SHORT)].isna().all(axis=1)
    print(f"rows with every question empty: {all_empty.sum()}")

    # ------------------------------------------------------------------
    section("3. Numeric features: distributions, outliers, separation")
    num = pd.DataFrame({"ease": df[Q_EASE]})
    for col in LIKERT_COLS:
        num[SHORT[col]] = df[col].map(LIKERT_MAP)
    num["price"] = price
    print(num.describe().round(2).to_string())

    print("\nprice: unparseable answers ->", df.loc[unparsed, Q_PRICE].tolist())
    q1, q3 = price.quantile([0.25, 0.75])
    print(f"price IQR rule upper fence = {q3 + 1.5 * (q3 - q1):.0f}; "
          f"answers above it = {(price > q3 + 1.5 * (q3 - q1)).sum()}; "
          f"above cap {PRICE_CAP:.0f} = {(price > PRICE_CAP).sum()}; "
          f"skewness raw = {price.skew():.2f}, "
          f"log1p(capped) = {np.log1p(price.clip(upper=PRICE_CAP)).skew():.2f}")
    print("ease takes half values:", sorted(set(num['ease'].dropna()) - set(range(11))))

    # class-conditional mean, and a separation score:
    # (largest gap between two class means) / (average within-class std)
    num_sep = num.copy()
    num_sep["price"] = np.log1p(price.clip(upper=PRICE_CAP))
    stats = num_sep.groupby(y).agg(["mean", "std"])
    sep_rows = []
    for f in num_sep.columns:
        means = stats[(f, "mean")]
        sep_rows.append({"feature": f if f != "price" else "log_price",
                         **{c: means[c] for c in LABELS},
                         "within_std": stats[(f, "std")].mean(),
                         "separation": (means.max() - means.min()) / stats[(f, "std")].mean()})
    sep = pd.DataFrame(sep_rows).sort_values("separation", ascending=False)
    sep.to_csv("tables/numeric_separation.csv", index=False)
    print("\n", sep.round(2).to_string(index=False))

    corr = num_sep.corr()
    corr.to_csv("tables/numeric_correlation.csv")
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    print(f"\nlargest |correlation| between numeric features: {off.max().max():.2f}")

    fig, axes = plt.subplots(1, 5, figsize=(11, 1.9))
    titles = {"ease": "Ease (0-10)", "strategy": "Strategy (1-5)",
              "injury": "Injury risk (1-5)", "individual": "Individual (1-5)",
              "price": "log(1+price)"}
    for ax, f in zip(axes, num_sep.columns):
        data = [num_sep.loc[y == c, f].dropna() for c in LABELS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True,
                        flierprops=dict(marker=".", markersize=2, color="gray"),
                        meanprops=dict(marker="o", markerfacecolor="white",
                                       markeredgecolor="black", markersize=3))
        for patch, c in zip(bp["boxes"], LABELS):
            patch.set_facecolor(COLORS[c])
            patch.set_alpha(0.7)
        ax.set_xticks([1, 2, 3], ["BB", "BK", "SC"])
        ax.set_title(titles[f])
    fig.suptitle("Numeric features by class (BB = Baseball, BK = Basketball, "
                 "SC = Soccer; circle = mean)", y=1.04)
    fig.savefig("figures/numeric_by_class.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------
    section("4. Multi-select features: % of responses choosing each option")
    blocks = []
    for col in MULTI_COLS:
        chosen = df[col].fillna("").map(lambda s: [p.strip() for p in s.split(",")])
        for opt in MULTI_OPTIONS[col]:
            has = chosen.map(lambda xs, o=opt: o in xs)
            blocks.append({"question": SHORT[col], "option": opt,
                           **{c: 100 * has[y == c].mean() for c in LABELS}})
    multi = pd.DataFrame(blocks)
    multi["max_gap"] = multi[LABELS].max(axis=1) - multi[LABELS].min(axis=1)
    multi.to_csv("tables/multiselect_by_class.csv", index=False)
    print(multi.round(1).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.5, 2.3))
    mat = multi[LABELS].to_numpy().T
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_yticks(range(3), LABELS)
    short_opt = {"School or university facility": "School/univ.",
                 "Community recreation facility": "Community rec.",
                 "Private sports facility": "Private facility",
                 "Outdoor public space": "Outdoor public",
                 "Indoor facility": "Indoor", "At home or nearby": "At home",
                 "Electronic or dance": "Electronic", "Hip-hop or rap": "Hip-hop",
                 "Folk or traditional": "Folk", "Classical or jazz": "Classical",
                 "No particular type": "No type"}
    ax.set_xticks(range(len(multi)),
                  [short_opt.get(o, o) for o in multi["option"]], rotation=60,
                  ha="right")
    for i in range(3):
        for j in range(len(multi)):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                    fontsize=6.5, color="white" if mat[i, j] > 60 else "black")
    bounds = np.cumsum([len(MULTI_OPTIONS[c]) for c in MULTI_COLS])[:-1]
    for b in bounds:
        ax.axvline(b - 0.5, color="black", lw=1)
    ax.set_title("% of responses selecting each option: "
                 "Where (left) | Season (middle) | Music (right)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.savefig("figures/multiselect_by_class.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------
    section("5. Free-text features: answer length and most class-specific words")
    word_rows = []
    for col in TEXT_COLS:
        toks = df[col].map(tokenize)
        lens = toks.map(len)
        print(f"\n[{SHORT[col]}] words per answer: median {lens.median():.0f}, "
              f"max {lens.max()}, distinct raw answers {raw[col].nunique()}")
        counts = {c: {} for c in LABELS}
        for t, c in zip(toks, y):
            for w in set(t):
                counts[c][w] = counts[c].get(w, 0) + 1
        vocab = set().union(*counts.values())
        n = {c: (y == c).sum() for c in LABELS}
        for w in vocab:
            freq = {c: counts[c].get(w, 0) / n[c] for c in LABELS}
            total = sum(counts[c].get(w, 0) for c in LABELS)
            if total < 8:
                continue
            top = max(freq, key=freq.get)
            # share of the word's occurrences that fall in its top class
            purity = counts[top].get(w, 0) / total
            word_rows.append({"question": SHORT[col], "word": w, "count": total,
                              "top_class": top, "purity": purity,
                              **{f"%{c}": 100 * freq[c] for c in LABELS}})
    words = pd.DataFrame(word_rows).sort_values(["question", "count"],
                                                ascending=[True, False])
    words.to_csv("tables/text_words_by_class.csv", index=False)
    for q in [SHORT[c] for c in TEXT_COLS]:
        sub = words[(words.question == q) & (words.purity >= 0.7)]
        print(f"\n[{q}] frequent words (>=8 uses) with >=70% in one class:")
        print(sub.head(12)[["word", "count", "top_class", "purity"]]
              .round(2).to_string(index=False))

    # ------------------------------------------------------------------
    section("6. Encoded design matrix")
    for min_df in [1, 2, 3, 5]:
        fb = FeatureBuilder(min_df=min_df).fit(raw)
        print(f"min_df={min_df}: {len(fb.feature_names_)} features "
              f"(text vocab sizes "
              f"{[len(fb.vocab_[c]) for c in TEXT_COLS]}) for {len(raw)} rows")
    fb = FeatureBuilder(min_df=2).fit(raw)
    X = fb.transform(raw)
    binary = X[:, [i for i, n in enumerate(fb.feature_names_)
                   if ":" in n or "=" in n]]
    rare = (binary.mean(axis=0) < 0.01).sum()
    print(f"binary features present in <1% of rows: {rare} of {binary.shape[1]}")
    sparsity = (binary == 0).mean()
    print(f"fraction of zeros among binary features: {sparsity:.3f}")

    # ------------------------------------------------------------------
    section("7. Single-feature predictiveness (mutual information with label)")
    yy = y.to_numpy()

    def mutual_info(x):
        mi = 0.0
        for v in np.unique(x):
            pv = (x == v).mean()
            for c in LABELS:
                pvc = ((x == v) & (yy == c)).mean()
                if pvc > 0:
                    mi += pvc * np.log2(pvc / (pv * (yy == c).mean()))
        return mi

    names = fb.feature_names_
    mi_rows = []
    for j, name in enumerate(names):
        col = X[:, j]
        if name.startswith("miss_"):
            continue
        if len(np.unique(col)) > 10:            # continuous: bin into quintiles
            col = pd.qcut(col, 5, labels=False, duplicates="drop")
        mi_rows.append({"feature": name, "mutual_info_bits": mutual_info(col)})
    mi = pd.DataFrame(mi_rows).sort_values("mutual_info_bits", ascending=False)
    mi.to_csv("tables/mutual_information.csv", index=False)
    print(mi.head(25).round(3).to_string(index=False))

    top = mi.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    groups = top["feature"].str.extract(r"^([a-z_]+)")[0]
    palette = {"ease": "#7f7f7f", "strategy": "#7f7f7f", "injury": "#7f7f7f",
               "individual": "#7f7f7f", "log_price": "#7f7f7f",
               "where": "#9467bd", "season": "#9467bd", "music": "#9467bd",
               "food": "#d62728", "colour": "#d62728", "song": "#d62728",
               "feel": "#d62728"}
    ax.barh(top["feature"], top["mutual_info_bits"],
            color=[palette.get(g, "gray") for g in groups])
    ax.set_xlabel("mutual information with label (bits; max = 1.585)")
    ax.set_title("Top 20 single features\n(grey = numeric, purple = multi-select,"
                 " red = text word)")
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig("figures/top_mutual_information.pdf")
    plt.close(fig)

    print("\nDone. Figures in figures/, tables in tables/.")


if __name__ == "__main__":
    main()
