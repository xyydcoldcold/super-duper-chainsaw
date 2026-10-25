"""
Cleaning and feature extraction for the CSC311 sports survey.

Everything here uses only the standard library, numpy and pandas, so the same
code can later be copied into pred.py (which may not import sklearn).

The main entry point is FeatureBuilder:

    fb = FeatureBuilder(min_df=2).fit(train_df)   # learns medians, means,
    X_train = fb.transform(train_df)               # stds and vocabularies
    X_val = fb.transform(val_df)                   # from the TRAINING rows only

Anything that is learned from data (imputation medians, standardisation
statistics, text vocabularies) is learned in fit() and only reused in
transform(), so no information leaks from the validation or test sets.
"""

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column names (the raw CSV uses the full question text as the header)
# ---------------------------------------------------------------------------

Q_EASE = ("How easy do you think it would be for someone with no prior "
          "experience to start playing this sport recreationally? Rate it on "
          "a scale from 0 to 10, where 0 means extremely difficult and 10 "
          "means extremely easy.")
Q_STRATEGY = "Success in this sport depends heavily on strategy and planning."
Q_INJURY = "Players must be comfortable with a high risk of injury."
Q_INDIVIDUAL = "Individual brilliance can outweigh teamwork in this sport."
Q_WHERE = "Where might you play this sport?"
Q_SEASON = ("During which seasons would you be most likely to play this sport "
            "recreationally in Toronto?")
Q_MUSIC = "What types of music do you most associate with this sport?"
Q_FOOD = ("What foods and drinks do you associate with watching this sport in "
          "the place or culture you know best?")
Q_COLOUR = "What colour do you associate with this sport?"
Q_SONG = "Which song do you associate with this sport?"
Q_FEEL = "How do you feel when you watch this sport?"
Q_PRICE = ("How much would you be willing to pay for a decent seat at a "
           "professional competition of this sport?")

# short names used in features, tables and plots
SHORT = {
    Q_EASE: "ease", Q_STRATEGY: "strategy", Q_INJURY: "injury",
    Q_INDIVIDUAL: "individual", Q_WHERE: "where", Q_SEASON: "season",
    Q_MUSIC: "music", Q_FOOD: "food", Q_COLOUR: "colour", Q_SONG: "song",
    Q_FEEL: "feel", Q_PRICE: "price",
}

LIKERT_COLS = [Q_STRATEGY, Q_INJURY, Q_INDIVIDUAL]
LIKERT_MAP = {"Strongly disagree": 1, "Disagree": 2, "Neutral": 3,
              "Agree": 4, "Strongly agree": 5}

MULTI_COLS = [Q_WHERE, Q_SEASON, Q_MUSIC]
MULTI_OPTIONS = {
    Q_WHERE: ["Indoor facility", "Outdoor public space",
              "School or university facility", "Community recreation facility",
              "Private sports facility", "At home or nearby", "Others"],
    Q_SEASON: ["Winter", "Spring", "Summer", "Fall"],
    Q_MUSIC: ["Pop", "Rock", "Hip-hop or rap", "Electronic or dance", "Latin",
              "Country", "Folk or traditional", "Classical or jazz", "Other",
              "No particular type"],
}

TEXT_COLS = [Q_FOOD, Q_COLOUR, Q_SONG, Q_FEEL]

LABELS = ["Baseball", "Basketball", "Soccer"]

# Prices above this are treated as outliers and clipped (see report).
PRICE_CAP = 1000.0

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

# Strings that mean "no answer". Free-text answers such as "none" or
# "no particular song" are NOT in this list: they are real answers and are
# kept as words in the bag-of-words.
MISSING_STRINGS = {"", "nan", "n/a", "na", "-", "--", ".", "?"}


def normalize_missing(df):
    """Return a copy of df where every kind of empty answer is NaN."""
    out = df.copy()
    for col in SHORT:
        s = out[col].astype("string").str.replace("\xa0", " ").str.strip()
        s = s.mask(s.str.lower().isin(MISSING_STRINGS))
        out[col] = s
    out[Q_EASE] = pd.to_numeric(out[Q_EASE], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Numeric free text: the price question
# ---------------------------------------------------------------------------

ZERO_WORDS = re.compile(
    r"\b(nothing|none|zero|free|unwilling|not willing|wouldn'?t|would not|"
    r"won'?t|no i|no willing|not really willing|pay me|not interested)\b")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_price(text):
    """
    Turn a free-text price answer into a number of dollars.

    - "$1,000", "200 CAD", "about 100$"  -> 1000, 200, 100
    - ranges "100-200", "$150-$250"       -> midpoint (150, 200)
    - "1k"                                -> 1000
    - answers that only express refusal  -> 0  ("Nothing", "Unwilling")
    - anything else without a number      -> NaN (unparseable)
    """
    if text is None or (isinstance(text, float) and np.isnan(text)) \
            or text is pd.NA:
        return np.nan
    t = str(text).lower().replace(",", "")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", lambda m: str(float(m.group(1)) * 1000), t)
    nums = [float(x) for x in NUMBER.findall(t)]
    if not nums:
        return 0.0 if ZERO_WORDS.search(t) else np.nan
    if len(nums) >= 2 and re.search(r"\d\s*(?:-|–|to|~)\s*\$?\s*\d", t):
        return (nums[0] + nums[1]) / 2
    return nums[0]


# ---------------------------------------------------------------------------
# Free text: bag of words
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "if", "in", "is", "it", "its", "like", "me", "my", "of", "on", "or",
    "so", "that", "the", "this", "to", "was", "with", "would", "you", "your",
    "very", "really", "just", "also", "some", "when", "game", "games",
    "sport", "sports", "watch", "watching", "feel", "think", "associate",
}


def tokenize(text):
    """Lower-case, drop placeholders like "[this sport]", split into words,
    remove stop words and strip a trailing plural "s" (hotdogs -> hotdog)."""
    if text is None or text is pd.NA or (isinstance(text, float) and np.isnan(text)):
        return []
    t = str(text).lower()
    t = re.sub(r"\[[^\]]*\]", " ", t)          # anonymisation placeholders
    t = t.replace("hot dog", "hotdog")
    words = re.findall(r"[a-z]+", t)
    out = []
    for w in words:
        if w in STOP_WORDS or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def split_multi(text):
    """Split a multi-select answer into its options. Option names never
    contain commas, so splitting on commas is safe."""
    if text is None or text is pd.NA or (isinstance(text, float) and np.isnan(text)):
        return []
    return [p.strip() for p in str(text).split(",") if p.strip()]


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """
    Builds the design matrix X (one row per survey response).

    Feature groups
      numeric : ease (0-10), 3 Likert items (1-5), log price   -> standardised
      missing : one 0/1 indicator per original question
      multi   : multi-hot encoding of the 3 multi-select questions (0/1)
      text    : binary bag-of-words for the 4 free-text questions (0/1),
                one vocabulary per question, words kept if they appear in
                at least `min_df` training responses
    """

    def __init__(self, min_df=2):
        self.min_df = min_df

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _numeric_raw(df):
        num = pd.DataFrame(index=df.index)
        num["ease"] = df[Q_EASE].astype(float)
        for col in LIKERT_COLS:
            num[SHORT[col]] = df[col].map(LIKERT_MAP).astype(float)
        price = df[Q_PRICE].map(parse_price).astype(float)
        num["log_price"] = np.log1p(price.clip(lower=0, upper=PRICE_CAP))
        return num

    # -- fit / transform -------------------------------------------------
    def fit(self, df):
        df = normalize_missing(df)
        num = self._numeric_raw(df)
        self.medians_ = num.median()
        filled = num.fillna(self.medians_)
        self.means_ = filled.mean()
        self.stds_ = filled.std(ddof=0).replace(0, 1.0)

        self.vocab_ = {}
        for col in TEXT_COLS:
            doc_freq = {}
            for toks in df[col].map(tokenize):
                for w in set(toks):
                    doc_freq[w] = doc_freq.get(w, 0) + 1
            words = sorted(w for w, c in doc_freq.items() if c >= self.min_df)
            self.vocab_[col] = words
        self.feature_names_ = self._build(df).columns.tolist()
        return self

    def transform(self, df):
        return self._build(normalize_missing(df)).to_numpy(dtype=float)

    def _build(self, df):
        parts = []

        num = self._numeric_raw(df)
        missing = num.isna()
        num = (num.fillna(self.medians_) - self.means_) / self.stds_
        parts.append(num)

        miss = pd.DataFrame(index=df.index)
        miss["miss_price_or_unparsed"] = missing["log_price"].astype(float)
        for col in SHORT:
            if col == Q_PRICE:
                continue
            miss[f"miss_{SHORT[col]}"] = df[col].isna().astype(float)
        parts.append(miss)

        for col in MULTI_COLS:
            chosen = df[col].map(split_multi)
            block = pd.DataFrame(
                {f"{SHORT[col]}={opt}": chosen.map(lambda xs, o=opt: float(o in xs))
                 for opt in MULTI_OPTIONS[col]}, index=df.index)
            parts.append(block)

        for col in TEXT_COLS:
            vocab = self.vocab_[col]
            index = {w: j for j, w in enumerate(vocab)}
            mat = np.zeros((len(df), len(vocab)))
            for i, toks in enumerate(df[col].map(tokenize)):
                for w in toks:
                    j = index.get(w)   # words never seen in training are ignored
                    if j is not None:
                        mat[i, j] = 1.0
            parts.append(pd.DataFrame(
                mat, index=df.index,
                columns=[f"{SHORT[col]}:{w}" for w in vocab]))

        return pd.concat(parts, axis=1)


def encode_labels(labels):
    return np.array([LABELS.index(y) for y in labels])
