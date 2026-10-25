"""Optional implementation of the proposal's training plan; test remains sealed.

The submitted proposal reports EDA, not these future model-comparison results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data import load_data, split_data, FEATURES, CLASSES, SEED
from features import SurveyPreprocessor

def specifications():
    return {
        'softmax': (LogisticRegression(solver='lbfgs', max_iter=3000, tol=1e-5),
                    {'model__C':[.001,.01,.1,1,10,100], 'prep__min_df':[1,2,3,5]}),
        'forest': (RandomForestClassifier(n_estimators=200, max_features='sqrt',
                    criterion='gini', bootstrap=True, random_state=SEED, n_jobs=1),
                    {'model__max_depth':[3,6,12,None], 'model__min_samples_leaf':[1,3,8],
                     'model__max_features':['sqrt',.3]}),
        'knn': (KNeighborsClassifier(algorithm='brute', metric='minkowski', n_jobs=1),
                    {'model__n_neighbors':[3,5,9,15,25,41],
                     'model__weights':['uniform','distance'], 'model__p':[1,2]})}

def run(csv,out):
    splits=split_data(load_data(csv)); tr=splits['train']; va=splits['validation']
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
               .split(tr[FEATURES],tr.label,groups=tr.id))
    for i,j in folds: assert set(tr.id.iloc[i]).isdisjoint(tr.id.iloc[j])
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,(model,grid) in specifications().items():
        search=GridSearchCV(Pipeline([('prep',SurveyPreprocessor()),('model',model)]),grid,
            scoring={'macro_f1':'f1_macro','accuracy':'accuracy'},refit='macro_f1',cv=folds,
            n_jobs=1,error_score='raise',return_train_score=True)
        search.fit(tr[FEATURES],tr.label)
        pd.DataFrame(search.cv_results_).to_csv(out/f'{name}_cv.csv',index=False)
        pred=search.predict(va[FEATURES])
        record={'family':name,'best_params':search.best_params_,
            'cv_macro_f1':float(search.best_score_),
            'cv_macro_f1_sd':float(search.cv_results_['std_test_macro_f1'][search.best_index_]),
            'validation_macro_f1':float(f1_score(va.label,pred,average='macro')),
            'validation_accuracy':float(accuracy_score(va.label,pred)),
            'validation_report':classification_report(va.label,pred,labels=CLASSES,output_dict=True,zero_division=0),
            'validation_confusion':confusion_matrix(va.label,pred,labels=CLASSES).tolist()}
        records.append(record)
    # Operational rule: highest validation macro-F1; within 0.01 prefer softmax, then KNN.
    best=max(r['validation_macro_f1'] for r in records)
    preference={'softmax':0,'knn':1,'forest':2}
    selected=min((r for r in records if best-r['validation_macro_f1']<=.01),
                 key=lambda r:preference[r['family']])['family']
    (out/'comparison.json').write_text(json.dumps({'models':records,'selected':selected,
        'test_evaluated':False,'note':'No held-out test metrics are computed by this command.'},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--out',default='model_results')
    a=p.parse_args(); run(a.data,a.out)
