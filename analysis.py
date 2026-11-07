"""Work through APS classification, from data preparation to model comparison."""
from pathlib import Path
import io, json, urllib.request, zipfile, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, recall_score, precision_score, average_precision_score)

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'results'
SEED=552; THREADS=4; BOOST_THREADS=1
ALPHAS=[0.0001, 0.001, 0.01, 0.1]
ROUNDS=100

def load_data():
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    names=['aps_failure_training_set.csv','aps_failure_test_set.csv']
    if not all((DATA/n).exists() for n in names):
        url='https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip'
        print('Downloading the official APS archive...',flush=True)
        try:
            with urllib.request.urlopen(url,timeout=180) as r:
                z=zipfile.ZipFile(io.BytesIO(r.read()))
            for n in names:
                matches=[m for m in z.namelist() if Path(m).name==n]
                if len(matches)!=1: raise ValueError(f'Missing or ambiguous archive member: {n}')
                (DATA/n).write_bytes(z.read(matches[0]))
        except Exception as e:
            raise RuntimeError('Download failed. Download the APS ZIP from UCI manually and put both CSVs in data/.') from e
    frames=[]
    for n in names:
        lines=(DATA/n).read_text().splitlines()
        start=next(i for i,l in enumerate(lines) if l.startswith('class,'))
        frames.append(pd.read_csv(io.StringIO('\n'.join(lines[start:])),na_values='na'))
    tr,te=frames
    assert tr.shape==(60000,171) and te.shape==(16000,171), (tr.shape,te.shape)
    assert tr.columns.equals(te.columns)
    assert set(tr['class'])=={'neg','pos'}
    X=tr.drop(columns='class'); T=te.drop(columns='class')
    y=tr['class'].map({'neg':0,'pos':1}).to_numpy()
    yt=te['class'].map({'neg':0,'pos':1}).to_numpy()
    assert y.sum()==1000 and yt.sum()==375
    return X,T,y,yt

def evaluation_impute(frame):
    """Follow the homework footnote: use this unlabeled evaluation batch's medians."""
    return SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(frame)

def metrics(y,p):
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Predicted probabilities must be finite and in [0,1].')
    pred=(p>=.5).astype(int)
    return dict(error=float(1-accuracy_score(y,pred)),auc=float(roc_auc_score(y,p)),
      recall=float(recall_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
      ap=float(average_precision_score(y,p)),cm=confusion_matrix(y,pred,labels=[0,1]).tolist())

def plot_probabilities(name,y,yt,ps):
    fig,ax=plt.subplots(1,3,figsize=(9,3.1))
    for side,truth,p,a in zip(['train','test'],[y,yt],ps,ax[:2]):
        cm=confusion_matrix(truth,p>=.5,labels=[0,1]); a.imshow(cm,cmap='Blues')
        for (i,j),v in np.ndenumerate(cm): a.text(j,i,str(v),ha='center',va='center',fontsize=11,color='white' if v>cm.max()/2 else 'black')
        a.set(xticks=[0,1],yticks=[0,1],xticklabels=['neg','pos'],yticklabels=['neg','pos'],
          xlabel='Predicted',ylabel='Actual',title=side+' confusion matrix')
    for s,truth,p in zip(['train','test'],[y,yt],ps):
        fpr,tpr,_=roc_curve(truth,p); ax[2].plot(fpr,tpr,label=f'{s}: AUC={roc_auc_score(truth,p):.4f}')
    ax[2].plot([0,1],[0,1],'k--',lw=.7); ax[2].set(xlabel='False positive rate',ylabel='True positive rate',title=name)
    ax[2].legend(fontsize=9); fig.tight_layout(); fig.savefig(OUT/(name+'.png'),dpi=170); plt.close(fig)

def evaluate(name, model, X,T,y,yt, results):
    ps=[model.predict_proba(a)[:,1] for a in (X,T)]
    results[name]={s:metrics(b,p) for s,b,p in zip(['train','test'],[y,yt],ps)}
    np.savez_compressed(OUT/(name+'_predictions.npz'),train=ps[0],test=ps[1])
    plot_probabilities(name,y,yt,ps)
    print(name,results[name],flush=True)

def booster(alpha):
    if alpha <= 0:
        raise ValueError("Use a positive alpha for the L1-penalized models.")
    from xgboost import XGBClassifier
    return XGBClassifier(booster='gblinear',objective='binary:logistic',
       reg_alpha=alpha,reg_lambda=0,n_estimators=ROUNDS,learning_rate=.1,
       updater='coord_descent',feature_selector='cyclic',n_jobs=BOOST_THREADS,
       random_state=SEED,eval_metric='logloss')

def cv_linear(X,y,smote):
    from imblearn.over_sampling import SMOTE
    # Fit the model and scaler on training rows. Impute validation using its own unlabeled batch, as the footnote asks.
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    errors=np.zeros((len(ALPHAS),5)); aucs=np.zeros_like(errors)
    oof=np.zeros((len(ALPHAS),len(y)),dtype=np.float32)
    for k,(a,b) in enumerate(folds):
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        scaler=StandardScaler()
        xa=scaler.fit_transform(np.log1p(imp.fit_transform(X.iloc[a]))).astype('float32')
        xb=scaler.transform(np.log1p(evaluation_impute(X.iloc[b]))).astype('float32')
        ya=y[a]
        if smote: xa,ya=SMOTE(random_state=SEED+k,k_neighbors=5).fit_resample(xa,ya)
        for j,alpha in enumerate(ALPHAS):
            model=booster(alpha).fit(xa,ya)
            p=model.predict_proba(xb)[:,1]; oof[j,b]=p
            print(f'SMOTE={smote}, fold={k+1}, alpha={alpha}: fitted',flush=True)
            errors[j,k]=np.mean((p>=.5)!=y[b]); aucs[j,k]=roc_auc_score(y[b],p)
        print(f'SMOTE={smote}: finished fold {k+1}/5',flush=True)
    table=pd.DataFrame({'alpha':ALPHAS,'mean_error':errors.mean(1),
       'sd_error':errors.std(1,ddof=1),'mean_auc':aucs.mean(1)})
    # Choose the lowest CV error. If errors tie, prefer the higher AUC.
    best=int(table.sort_values(['mean_error','mean_auc'],ascending=[True,False]).index[0])
    tag='smote' if smote else 'plain'
    table.to_csv(OUT/f'cv_{tag}.csv',index=False)
    np.savez_compressed(OUT/f'oof_{tag}.npz',probability=oof[best],truth=y)
    return ALPHAS[best], float(errors[best].mean())

def exploratory(X,T,y,yt):
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    A=pd.DataFrame(imp.fit_transform(X),columns=X.columns)
    B=evaluation_impute(T)
    means=A.mean(); cv=A.std(ddof=1).div(means.where(means!=0))
    pd.DataFrame({'mean':means,'sample_sd':A.std(ddof=1),'CV':cv,
      'missing_fraction':X.isna().mean()}).to_csv(OUT/'feature_CV_all_170.csv')
    top=cv.replace([np.inf,-np.inf],np.nan).dropna().nlargest(int(np.sqrt(170))).index.tolist()
    assert len(top)==13
    corr=A.corr()
    corr.to_csv(OUT/'correlations_all_170.csv')
    fig,ax=plt.subplots(figsize=(10,9)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set(xticks=np.arange(0,170,10),yticks=np.arange(0,170,10),
      xticklabels=X.columns[::10],yticklabels=X.columns[::10],title='Training feature correlations (all 170)')
    plt.setp(ax.get_xticklabels(),rotation=90); fig.colorbar(im,ax=ax); fig.tight_layout()
    fig.savefig(OUT/'correlation.png',dpi=200); plt.close(fig)
    # Plot every training observation in the 13-by-13 matrix. The log1p transform
    # makes the skewed values easier to see; CV is still calculated in original units.
    Z=np.log1p(A[top]); n=len(top)
    fig,axes=plt.subplots(n,n,figsize=(22,22))
    colors=np.where(y==1,'#b62832','#246a9a')
    for i in range(n):
      for j in range(n):
        ax=axes[i,j]
        if i==j:
          for c,color in [(0,'#246a9a'),(1,'#b62832')]:
            ax.hist(Z.iloc[y==c,i],bins=25,density=True,histtype='step',color=color)
        else: ax.scatter(Z.iloc[:,j],Z.iloc[:,i],c=colors,s=.4,alpha=.12,rasterized=True)
        ax.tick_params(labelsize=4)
        if i==n-1: ax.set_xlabel(top[j],fontsize=7)
        if j==0: ax.set_ylabel(top[i],fontsize=7)
    fig.suptitle('Top 13 CV features: log1p scatter matrix; blue=neg, red=pos',fontsize=18)
    fig.tight_layout(rect=[0,0,1,.98]); fig.savefig(OUT/'scatter_matrix.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(13,8))
    for ax,col in zip(axes.flat,top):
        ax.boxplot([Z.loc[y==0,col],Z.loc[y==1,col]],tick_labels=['neg','pos'],showfliers=True,flierprops={'marker':'.','markersize':2,'alpha':.3})
        ax.set_title(col,fontsize=9); ax.set_ylabel('log1p(value)',fontsize=7)
    for ax in list(axes.flat)[13:]: ax.axis('off')
    fig.suptitle('Class-wise box plots: all observations, including outliers')
    fig.tight_layout(); fig.savefig(OUT/'boxplots.png',dpi=170); plt.close(fig)
    return A.to_numpy(dtype='float32'), B.astype('float32'), {'top13':top,
      'train_neg':int((y==0).sum()),'train_pos':int(y.sum()),
      'test_neg':int((yt==0).sum()),'test_pos':int(yt.sum()),
      'missing_fraction':float(X.isna().to_numpy().mean())}

def main():
    from imblearn.over_sampling import SMOTE
    X,T,y,yt=load_data(); A,B,info=exploratory(X,T,y,yt)
    results={'data':info,'models':{}}
    for tag,weight in [('rf_plain',None),('rf_weighted','balanced_subsample')]:
        model=RandomForestClassifier(n_estimators=300,max_features='sqrt',bootstrap=True,
          oob_score=True,class_weight=weight,n_jobs=THREADS,random_state=SEED)
        model.fit(A,y); evaluate(tag,model,A,B,y,yt,results['models'])
        results['models'][tag]['oob_error']=float(1-model.oob_score_)
    for smote in [False,True]:
        tag='xgb_smote' if smote else 'xgb_plain'
        alpha,cv_error=cv_linear(X,y,smote)
        imp=SimpleImputer(strategy='median',keep_empty_features=True); scaler=StandardScaler()
        xa=scaler.fit_transform(np.log1p(imp.fit_transform(X))).astype('float32')
        xb=scaler.transform(np.log1p(evaluation_impute(T))).astype('float32')
        fitx,fity=(SMOTE(random_state=SEED,k_neighbors=5).fit_resample(xa,y) if smote else (xa,y))
        model=booster(alpha).fit(fitx,fity)
        # Score the original training rows so the comparison uses the real class balance.
        evaluate(tag,model,xa,xb,y,yt,results['models'])
        results['models'][tag].update(alpha=alpha,cv_error=cv_error,fit_rows=int(len(fity)))
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    import report
    report.build(results)
    print('Finished. Open results/HW6_Results.pdf',flush=True)
    return results

if __name__=='__main__': main()
