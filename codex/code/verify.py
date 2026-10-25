"""Behavior checks for parsing, group leakage, fitted state and model compatibility."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from data import load_data, split_data, FEATURES, CLASSES, SEED
from features import parse_price, parse_ease, SurveyPreprocessor
from models import specifications

def check(csv,out):
    for raw,expected in [('$1,000',1000),('$10 000',10000),('CAD 150',150),
        ('$150-$250',200),('300$-500$',400),('Nothing',0),('Not willing.',0)]:
        assert parse_price(raw)==expected,raw
    for raw in ['none','USD 100','1€','-50','200 or 500 depending on game',
                '1 to 10 scale i will choose 2','100-50','nan']:
        assert np.isnan(parse_price(raw)),raw
    assert parse_ease('3.5')==3.5 and np.isnan(parse_ease('11'))
    splits=split_data(load_data(csv)); tr=splits['train']
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    folds=list(cv.split(tr[FEATURES],tr.label,groups=tr.id))
    for i,j in folds: assert set(tr.id.iloc[i]).isdisjoint(tr.id.iloc[j])
    a,b=folds[0]; train=tr.iloc[a]; probe=tr.iloc[b]
    prep=SurveyPreprocessor().fit(train[FEATURES])
    cap=prep.price_cap_; med=prep.medians_.copy(); names=prep.get_feature_names_out().copy()
    exotic=probe[FEATURES].head(2).copy()
    for c in ['food','colour','song','feeling']: exotic[c]='zzzxnewtokenzzx'
    exotic['price']=['$1000000000000','USD 100']; exotic['venue']='unseen venue'
    z=prep.transform(exotic)
    assert np.isfinite(z.data).all() and z.shape[1]==len(names)
    assert prep.price_cap_==cap and np.array_equal(prep.medians_,med)
    assert np.array_equal(prep.get_feature_names_out(),names)
    assert not any('zzzxnewtokenzzx' in str(s) for s in names)
    # Appending ID/target must have no influence on the selected inputs.
    xx=train.copy(); xx['id']=999; xx['label']='not a feature'
    assert (prep.transform(train[FEATURES]) != prep.transform(xx)).nnz==0
    checked=[]
    for name,(model,grid) in specifications().items():
        assert len(ParameterGrid(grid))==24
        pipe=Pipeline([('prep',SurveyPreprocessor()),('model',model)])
        pipe.fit(train[FEATURES],train.label)
        pred=pipe.predict(probe[FEATURES])
        assert len(pred)==len(probe) and set(pred).issubset(CLASSES)
        checked.append(name)
    result={'passed':True,'group_disjointness':True,'five_group_folds':True,
       'parser_edge_cases':True,'unknown_tokens_handled':True,
       'transform_does_not_refit':True,'id_and_label_excluded':True,
       'model_smoke_fits':checked,'candidate_counts':{n:24 for n in checked},
       'validation_or_test_evaluated':False,
       'note':'Smoke fits use one internal training fold. No performance claims or model selection.'}
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',required=True)
    p.add_argument('--out',default='verification.json'); a=p.parse_args(); check(a.data,a.out)
