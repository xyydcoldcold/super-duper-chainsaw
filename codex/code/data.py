"""Schema validation and deterministic respondent-level splits (no learned transforms)."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 311
CLASSES = ['Baseball', 'Basketball', 'Soccer']
FEATURES = ['ease', 'strategy', 'injury', 'individual', 'venue', 'season',
            'music', 'food', 'colour', 'song', 'feeling', 'price']
NUMERIC = ['ease', 'strategy', 'injury', 'individual', 'price']
MULTI = ['venue', 'season', 'music']
TEXT = ['food', 'colour', 'song', 'feeling']

def load_data(path):
    expected = json.loads(Path(__file__).with_name('schema.json').read_text())
    df = pd.read_csv(path, keep_default_na=False)
    if list(df.columns) != expected:
        raise ValueError('CSV headers differ from the official training schema.')
    df.columns = ['id'] + FEATURES + ['label']
    if df.id.isna().any() or (df.id.astype(str).str.strip() == '').any():
        raise ValueError('Missing respondent ID.')
    if not set(df.label).issubset(CLASSES):
        raise ValueError('Unexpected label.')
    return df

def split_data(df):
    ids = np.array(sorted(df.id.unique()))
    np.random.default_rng(SEED).shuffle(ids)
    n = int(np.ceil(0.15 * len(ids)))
    groups = {'test': ids[:n], 'validation': ids[n:2*n], 'train': ids[2*n:]}
    result = {k: df.loc[df.id.isin(v)].copy() for k, v in groups.items()}
    assert not (set(groups['train']) & set(groups['test']))
    assert not (set(groups['train']) & set(groups['validation']))
    assert not (set(groups['test']) & set(groups['validation']))
    assert sum(map(len, result.values())) == len(df)
    return result

def save_manifest(df, splits, source, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    assignment = {int(i): k for k, d in splits.items() for i in d.id.unique()}
    pd.DataFrame({'row_index': np.arange(len(df)), 'id': df.id,
                  'split': df.id.map(assignment)}).to_csv(out/'split_manifest.csv', index=False)
    meta = {'seed': SEED, 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'splits': {k: {'respondents': int(d.id.nunique()), 'rows': len(d)}
                       for k, d in splits.items()},
            'note': 'EDA uses train only. No validation/test feature summaries or model scores.'}
    (out/'split_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')

