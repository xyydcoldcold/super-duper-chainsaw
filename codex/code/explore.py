"""Reproduce all proposal EDA. Never summarize validation or test features/labels."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from data import load_data, split_data, save_manifest, FEATURES, NUMERIC, MULTI, TEXT, CLASSES
from features import numeric_frame, options, normalize, clean_text, is_missing, SurveyPreprocessor

def run(csv, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = load_data(csv)
    splits = split_data(df)
    save_manifest(df, splits, csv, out)
    tr = splits['train']
    n = numeric_frame(tr)
    rows = []
    for c in FEATURES:
        raw_missing = tr[c].map(is_missing)
        effective = n[c].isna() if c in NUMERIC else raw_missing
        v = n[c].dropna() if c in NUMERIC else tr.loc[~raw_missing,c].map(normalize)
        row = {'feature': c, 'type': ('numeric' if c in ['ease','price'] else
                'ordinal' if c in NUMERIC else 'multi-select' if c in MULTI else 'text'),
                'raw_missing_n': int(raw_missing.sum()), 'raw_missing_pct': 100*raw_missing.mean(),
                'effective_missing_n': int(effective.sum()), 'effective_missing_pct': 100*effective.mean(),
                'distinct_nonmissing': int(v.nunique()),
                'mode': str(v.mode().iloc[0]) if len(v) else '',
                'mode_pct': 100*float(v.value_counts().iloc[0])/len(v) if len(v) else 0}
        if c in NUMERIC:
            row.update({k:float(v.quantile(q)) for k,q in [('min',0),('q25',.25),('median',.5),('q75',.75),('max',1)]})
            row['mean'] = float(v.mean()); row['std'] = float(v.std())
        else:
            lengths = v.str.len()
            row['median_chars'] = float(lengths.median()) if len(v) else 0
            row['max_chars'] = int(lengths.max()) if len(v) else 0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'feature_summary.csv', index=False)
    tr.label.value_counts().reindex(CLASSES).rename_axis('class').to_csv(out/'class_balance.csv')
    means = n.assign(label=tr.label).groupby('label').agg(['mean','std','median','count']).reindex(CLASSES)
    means.to_csv(out/'numeric_by_class.csv')
    corr = n.corr(method='spearman'); corr.to_csv(out/'numeric_spearman.csv')
    missing_class = pd.DataFrame({c:(n[c].isna() if c in NUMERIC else tr[c].map(is_missing))
                                 for c in FEATURES}).assign(label=tr.label).groupby('label').mean()
    missing_class.reindex(CLASSES).to_csv(out/'missing_rate_by_class.csv')
    multi_rows = []
    for c in MULTI:
        all_options = sorted({x for v in tr[c] for x in options(v)})
        for opt in all_options:
            indicator = tr[c].map(lambda x: opt in options(x))
            row = {'feature':c,'option':opt,'count':int(indicator.sum())}
            row.update({cl:float(indicator[tr.label==cl].mean()) for cl in CLASSES})
            row['class_rate_range'] = max(row[cl] for cl in CLASSES)-min(row[cl] for cl in CLASSES)
            multi_rows.append(row)
    multi = pd.DataFrame(multi_rows).sort_values('class_rate_range',ascending=False)
    multi.to_csv(out/'multiselect_by_class.csv',index=False)
    # Complete distributions for every original input, without respondent-level raw text.
    distributions = {}
    for c in FEATURES:
        if c in NUMERIC:
            distributions[c] = n[c].value_counts(dropna=False).sort_index().to_dict()
        elif c in MULTI:
            distributions[c] = {o:int(sum(o in options(x) for x in tr[c]))
                                for o in sorted({o for x in tr[c] for o in options(x)})}
        else:
            distributions[c] = tr[c].map(clean_text).str.len().value_counts().sort_index().to_dict()
    (out/'distributions.json').write_text(json.dumps(distributions,default=str,indent=2)+'\n')
    for c in TEXT:
        tr[c].map(clean_text).str.len().rename('characters').to_frame().groupby('characters').size().to_csv(out/f'{c}_length_distribution.csv')
    prep = SurveyPreprocessor().fit(tr[FEATURES])
    X = prep.transform(tr[FEATURES])
    names = prep.get_feature_names_out()
    token_rows = []
    # Chi-square is computed only on nonnegative text TF-IDF blocks, not scaled numbers.
    for c,v in prep.vectorizers_.items():
        if v is None: continue
        z = v.transform(tr[c].map(clean_text)); score,_ = chi2(z,tr.label)
        support = np.asarray((z>0).sum(axis=0)).ravel()
        for j in np.argsort(np.nan_to_num(score,nan=-1))[::-1]:
            if support[j] < 5: continue
            record = {'feature':c,'term':v.get_feature_names_out()[j],
                      'chi2':float(score[j]),'support':int(support[j])}
            record.update({cl:float((z[tr.label.to_numpy()==cl,j]>0).mean()) for cl in CLASSES})
            token_rows.append(record)
    pd.DataFrame(token_rows).to_csv(out/'text_associations.csv',index=False)
    zero_n = int(np.sum(np.asarray((X!=0).sum(axis=0)).ravel()==0))
    metrics = {'training_rows':len(tr),'training_respondents':int(tr.id.nunique()),
        'feature_count':int(X.shape[1]), 'density':X.nnz/(X.shape[0]*X.shape[1]),
        'price_cap_99pct':prep.price_cap_, 'constant_zero_processed_columns':zero_n,
        'exact_duplicate_feature_label_rows':int(tr.duplicated(FEATURES+['label']).sum()),
        'price_iqr_outliers':int(((n.price < n.price.quantile(.25)-1.5*(n.price.quantile(.75)-n.price.quantile(.25)))|
                                (n.price > n.price.quantile(.75)+1.5*(n.price.quantile(.75)-n.price.quantile(.25)))).sum()),
        'foreign_currency_rows':int(tr.price.str.contains(r'(?i)(?<![a-z])(?:usd|eur|gbp)(?![a-z])|[€£]',regex=True).sum()),
        'text_vocab_sizes':{c:len(v.vocabulary_) if v is not None else 0 for c,v in prep.vectorizers_.items()},
        'multiselect_sizes':{c:len(opts) for c,opts in prep.options_.items()}}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    # Every numeric/ordinal feature: class means with SD (price is on log scale).
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors = ['#39749d','#c07539','#57966d']
    fig,axes = plt.subplots(1,2,figsize=(10.5,3.05),layout='constrained')
    q = n.copy(); q.price=np.log1p(q.price)
    for j,cl in enumerate(CLASSES):
        vals=q[tr.label==cl]
        axes[0].bar(np.arange(5)+(j-1)*.25,vals.mean(),width=.24,label=cl,color=colors[j])
    axes[0].set_xticks(np.arange(5),['Ease','Strategy','Injury','Individual','log(1+price)'],rotation=20)
    axes[0].set_ylabel('Training-set mean (observed values)')
    axes[0].set_title('(a) Numeric and ordinal responses',fontsize=11)
    top = multi.head(5).iloc[::-1]
    labels=[r.feature+': '+r.option for r in top.itertuples()]
    for j,cl in enumerate(CLASSES):
        axes[1].barh(np.arange(len(top))+(j-1)*.25,100*top[cl],height=.24,color=colors[j])
    axes[1].set_yticks(np.arange(len(top)),labels,fontsize=8)
    axes[1].set_xlabel('Respondents selecting option (%)'); axes[1].set_xlim(0,100)
    axes[1].set_title('(b) Largest class differences in multi-select rates',fontsize=11)
    axes[0].legend(fontsize=8,ncol=3,loc='upper left',bbox_to_anchor=(0,-.25))
    fig.savefig(out/'relationships.pdf',bbox_inches='tight'); plt.close(fig)
    # Full distributions for the entire training portion, also provided outside the concise PDF.
    fig,axes=plt.subplots(3,4,figsize=(14,9),layout='constrained')
    for ax,c in zip(axes.flat,FEATURES):
        if c in NUMERIC:
            v=n[c].dropna(); v=np.log1p(v) if c=='price' else v
            ax.hist(v,bins=15,color=colors[0]); ax.set_xlabel('log(1+price)' if c=='price' else 'Value')
        elif c in MULTI:
            counts=pd.Series(distributions[c]).sort_values().tail(10)
            ax.barh(range(len(counts)),counts.values,color=colors[0]); ax.set_yticks(range(len(counts)),counts.index,fontsize=6)
        else:
            ax.hist(tr[c].map(clean_text).str.len(),bins=20,color=colors[0]); ax.set_xlabel('Cleaned response characters')
        ax.set_title(c); ax.set_ylabel('Training rows')
    fig.savefig(out/'all_feature_distributions.pdf'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r'); fig.colorbar(im,ax=ax,label='Spearman correlation')
    ax.set_xticks(range(5),NUMERIC,rotation=35); ax.set_yticks(range(5),NUMERIC)
    for i in range(5):
        for j in range(5): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.savefig(out/'numeric_correlations.pdf'); plt.close(fig)
    print(json.dumps(metrics,indent=2))
    print(summary.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--out',default='analysis')
    args=p.parse_args(); run(args.data,args.out)
