"""Explicit cleaning rules and fold-local fitted preprocessing.

No ID or target enters the feature matrix. Unseen text tokens/options are ignored.
"""
import re
import unicodedata
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from data import FEATURES, NUMERIC, MULTI, TEXT

MISSING = {'', 'n/a', 'na', 'nan', 'null', 'none', '-', '[]', 'unknown',
           'not sure', "i don't know", 'idk'}
LIKERT = {'strongly disagree': 1, 'disagree': 2, 'neutral': 3,
          'agree': 4, 'strongly agree': 5}

def normalize(value):
    if pd.isna(value): return ''
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value))).strip().casefold()

def is_missing(value):
    return normalize(value).rstrip('.') in MISSING

def clean_text(value):
    if is_missing(value): return ''
    # Staff-inserted redactions are not words originally supplied by the respondent.
    return re.sub(r'\[[^\]]*\]', ' ', normalize(value))

def options(value):
    if is_missing(value): return []
    return sorted(set(normalize(x) for x in str(value).split(',') if normalize(x)))

def parse_ease(value):
    try: number = float(normalize(value))
    except ValueError: return np.nan
    return number if np.isfinite(number) and 0 <= number <= 10 else np.nan

def parse_price(value):
    """Accept a whole monetary answer or one simple range; abstain on narratives.

    CAD/default unspecified dollars are one scale. No guessed FX conversion.
    Explicit unwillingness means zero; bare 'none' is missing, not a guessed zero.
    """
    s = normalize(value).rstrip('.!')
    if s in {'zero', 'nothing', 'unwilling', 'not willing', 'no willing'} or re.fullmatch(
            r"i (?:would not|wouldn't|wouldnt) pay(?: to watch(?: in person| this sport)?| for (?:a )?.+)?", s):
        return 0.0
    if is_missing(s) or re.search(r'(?<![a-z])(?:usd|eur|euro|pounds?|gbp)(?![a-z])|[€£]', s):
        return np.nan
    # Normalize grouped thousands, but preserve decimal points.
    s = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', s)
    s = re.sub(r'(?<=\d) (?=\d{3}(?:\D|$))', '', s)
    s = re.sub(r'^(?:around|about|at most|up to|under|no more than)\s*', '', s)
    s = re.sub(r'\s*(?:max|maximum)$', '', s)
    unit = r'(?:canadian\s+dollars?|dollars?|dollar|bucks?|cad|ca\$|\$)'
    s = re.sub(unit, '', s).strip()
    number = r'\d+(?:\.\d+)?'
    if re.fullmatch(number, s): return float(s)
    m = re.fullmatch(fr'({number})\s*(?:-|–|—|to)\s*({number})', s)
    if m:
        lo, hi = map(float, m.groups())
        return (lo+hi)/2 if lo <= hi else np.nan
    return np.nan

def numeric_frame(df):
    d = pd.DataFrame(index=df.index)
    d['ease'] = df.ease.map(parse_ease)
    for c in ['strategy', 'injury', 'individual']:
        d[c] = df[c].map(lambda x: LIKERT.get(normalize(x), np.nan))
    d['price'] = df.price.map(parse_price)
    return d.astype(float)

class SurveyPreprocessor(TransformerMixin, BaseEstimator):
    """Median/scale, 99th-percentile price cap, vocabularies fit only in fit()."""
    def __init__(self, min_df=2, max_features=500):
        self.min_df = min_df
        self.max_features = max_features

    def _numeric(self, X):
        n = numeric_frame(X)
        missing = n.isna().astype(float).to_numpy()
        n['price'] = np.log1p(n.price.clip(upper=self.price_cap_))
        a = n.to_numpy()
        a = np.where(np.isnan(a), self.medians_, a)
        return a, missing

    def fit(self, X, y=None):
        if not set(FEATURES).issubset(X.columns): raise ValueError('Missing input feature.')
        n = numeric_frame(X)
        valid_price = n.price.dropna()
        self.price_cap_ = float(valid_price.quantile(.99)) if len(valid_price) else 0.0
        n['price'] = np.log1p(n.price.clip(upper=self.price_cap_))
        self.medians_ = n.median().fillna(0).to_numpy()
        a, _ = self._numeric(X)
        self.scaler_ = StandardScaler().fit(a)
        self.options_ = {c: sorted({v for x in X[c] for v in options(x)}) for c in MULTI}
        self.vectorizers_ = {}
        for c in TEXT:
            vec = TfidfVectorizer(ngram_range=(1,2), min_df=self.min_df,
                    max_features=self.max_features, sublinear_tf=True,
                    strip_accents='unicode', norm='l2')
            docs = X[c].map(clean_text)
            try: vec.fit(docs)
            except ValueError as e:
                if 'empty vocabulary' not in str(e) and 'no terms remain' not in str(e).lower(): raise
                vec = None
            self.vectorizers_[c] = vec
        self.feature_names_ = ['num_'+c for c in NUMERIC]+['missing_'+c for c in NUMERIC]
        self.feature_names_ += ['missing_'+c for c in MULTI+TEXT]
        for c in MULTI: self.feature_names_ += [c+'='+s for s in self.options_[c]]
        for c, v in self.vectorizers_.items():
            if v is not None: self.feature_names_ += [c+'::'+s for s in v.get_feature_names_out()]
        return self

    def transform(self, X):
        a, missing = self._numeric(X)
        other_missing = np.column_stack([X[c].map(is_missing) for c in MULTI+TEXT]).astype(float)
        blocks = [sparse.csr_matrix(self.scaler_.transform(a)), sparse.csr_matrix(missing),
                  sparse.csr_matrix(other_missing)]
        for c in MULTI:
            lookup = {s:i for i,s in enumerate(self.options_[c])}
            rows, cols = [], []
            for i,x in enumerate(X[c]):
                for s in options(x):
                    if s in lookup: rows.append(i); cols.append(lookup[s])
            blocks.append(sparse.csr_matrix((np.ones(len(rows)), (rows,cols)),
                                           shape=(len(X),len(lookup))))
        for c,v in self.vectorizers_.items():
            if v is not None: blocks.append(v.transform(X[c].map(clean_text)))
        result = sparse.hstack(blocks, format='csr')
        assert result.shape[1] == len(self.feature_names_)
        if not np.isfinite(result.data).all(): raise ValueError('Non-finite processed features.')
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
