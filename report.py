from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
ST=getSampleStyleSheet()
ST['BodyText'].fontSize=10; ST['BodyText'].leading=14; ST['BodyText'].spaceAfter=9
ST['Heading1'].fontSize=17; ST['Heading1'].leading=21
ST['Heading2'].fontSize=12

def picture(path,maxw=490,maxh=410):
    w,h=ImageReader(str(path)).getSize(); ratio=min(maxw/w,maxh/h)
    return Image(str(path),width=w*ratio,height=h*ratio)

def svm_figure():
    x=np.array([[3,4],[2,2],[4,4],[1,4],[2,1],[4,3],[4,1]])
    fig,ax=plt.subplots(figsize=(7,4.4))
    for i,(a,b) in enumerate(x):
        ax.scatter(a,b,c='#ad2635' if i<4 else '#216b9a',s=80,zorder=4)
        ax.annotate(str(i+1),(a,b),xytext=(7,5),textcoords='offset points')
    z=np.linspace(.5,4.6,100)
    ax.plot(z,z-.5,'k-',label='optimal: x2=x1-0.5')
    ax.plot(z,z,'k--',lw=.8,label='supporting lines'); ax.plot(z,z-1,'k--',lw=.8)
    ax.fill_between(z,z-1,z,color='gray',alpha=.12)
    ax.plot(z,z-.25,':',color='purple',label='alternative: x2=x1-0.25')
    ax.scatter(2,3,marker='x',s=100,c='#216b9a',label='new Blue point')
    ax.set(xlim=(.5,4.6),ylim=(.4,4.6),xlabel='X1',ylabel='X2',aspect='equal')
    ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1)); fig.tight_layout()
    fig.savefig(OUT/'svm_geometry.png',dpi=170); plt.close(fig)

def table(data):
    t=Table(data,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.gray)]))
    return t

def build(results=None):
    svm_figure()
    import learning_plots
    learning_plots.make_all()
    story=[]
    def p(s): story.append(Paragraph(s,ST['BodyText']))
    def h(s): story.append(Paragraph(s,ST['Heading1']))
    h('DSCI 552 | Homework 6 study notes')
    p('Yash Desai | Study notes on APS classification and the ISLR exercises')
    if results is None:
        p('<b>The model runs are not available in this copy.</b> Run analysis.py to create the results before using the numerical comparisons.')
    else: p('<b>How to read these notes:</b> the code and results are followed by the reason behind each step. '+('All four implemented models have been run.' if len(results['models'])==4 else 'Some model results are still pending.')+' The official split has 60,000 training and 16,000 test rows. The positive class is pos=1, and probabilities of at least 0.5 are treated as positive predictions. No majority-class downsampling is used.')
    h('1(a)-(b). Data and preparation')
    p('The goal is to tell APS-related failures from failures involving other components. The training set has 59,000 negatives and 1,000 positives (59:1; 1.667% positive). The test set has 15,625 negatives and 375 positives (41.667:1; 2.344% positive). There are 170 numeric predictors. Always predicting negative would give only 2.344% test error, but it would miss every APS failure. That is why accuracy alone is not enough.')
    p('<b>(b.i)</b> Missing entries can be handled with a mean, median or mode; KNN; iterative or multiple imputation; missing-value indicators; or a model that handles them directly. Deleting incomplete rows could discard a lot of useful data. Here we use training-feature medians because they are practical and less sensitive to extreme values. This keeps all rows and 170 predictors. An entirely missing training column is retained with a zero fill. Median filling can still reduce variation and change correlations, so it has limitations.')
    p('Following the homework footnote, missing test values are filled with medians from the unlabeled test batch itself. Training rows have their own median imputer. CV mirrors this rule: each validation batch is imputed separately, while the scaler and model are fitted only on the training rows in that fold. No evaluation labels enter imputation, and evaluation statistics do not enter training or SMOTE. This batch-based protocol differs from the usual deployment practice of storing training medians for future individual predictions.')
    p("<b>(b.ii)</b> CV=s/m compares a feature's sample standard deviation with its mean. We calculate it after median imputation, before scaling, using ddof=1. A mean of zero makes the ratio undefined, so it is stored as NaN. The complete 170-feature result is in feature_CV_all_170.csv.")
    p('<b>(b.iii)-(iv)</b> The correlation matrix uses all 170 imputed training features. For the scatter and box plots, we select the 13 highest finite CV values because floor(sqrt(170))=13. This choice is only for the plots: every model still uses all 170 predictors. The plots use log1p values to make the skew easier to see. Every training observation is included, and box-plot outliers are shown with small markers.')
    p('A visible pattern can suggest an association, but it cannot establish statistical significance, conditional usefulness or causality. High CV means high relative variation, not necessarily strong predictive value. Formal significance claims would need a separate inferential procedure, with attention to multiple comparisons.')
    story.append(PageBreak()); h('1(c)-(f). Models and validation')
    p('<b>(c)</b> Start with a random forest that makes no adjustment for imbalance: 300 bootstrap trees, sqrt(p) candidate features at each split, unrestricted depth and seed 552. The outputs show training and test confusion matrices, ROC curves, AUC and error=(FP+FN)/n. Matrix rows are actual [neg,pos]; columns are predicted [neg,pos]. For OOB error, each observation is predicted using trees that left it out of their bootstrap sample. We compare that estimate with test error. Since the imputer was fitted before the forest, OOB checks the forest with that imputer already in place, rather than cross-fitting the whole preprocessing pipeline.')
    p('<b>(d)</b> Ways to address imbalance include class weights, balanced bootstrap samples, oversampling, undersampling and thresholds chosen on training validation data. Here balanced_subsample recalculates inverse-frequency weights in each bootstrap sample; the other forest settings stay the same. Full-training weights would be about 0.5085 for neg and 30 for pos, a 59:1 relative ratio. Weighting makes rare-class mistakes matter more during fitting, but its effect on the test results still needs to be checked.')
    p("<b>(e): a wording issue to keep in mind.</b> XGBoost's gblinear booster fits one global logistic model with an L1 penalty controlled by reg_alpha. It does not fit logistic regressions at individual tree nodes. On a tree booster, reg_alpha penalizes leaf scores instead. This project uses gblinear with binary:logistic for the L1-logistic part of the task. It therefore does <b>not</b> claim to implement the literal node-wise model tree described in the question; that interpretation still needs clarification.")
    p('For the linear models, median filling is followed by log1p and standardization. All APS predictors are nonnegative. Training-only checks found numerical saturation without the log transform, so the transform is used consistently in both comparisons. The fit uses coordinate descent, cyclic feature selection, 100 rounds, learning rate 0.1 and no L2 penalty. Five shuffled, stratified folds compare positive alpha values {0.0001,0.001,0.01,0.1}, using seed 552. The lowest mean classification error wins, with AUC as tie breaker. Because the same folds select alpha, the selected CV error may be optimistic; it is not nested CV. Test error is reported separately.')
    p('<b>(f)</b> SMOTE belongs inside each CV training fold. Fit the training imputer and scaler, then create synthetic minority examples from the transformed training rows. Impute validation rows using their own unlabeled batch medians and apply the training-fitted scaler; never resample validation rows. Applying SMOTE before CV could let validation observations influence synthetic training points and make the score look too good. After selecting a positive alpha, refit preprocessing and SMOTE(k=5) on the full training set: 59,000 negatives and 59,000 positives, including 58,000 synthetic positives. That gives 118,000 fitting rows. Scores still use the original 60,000 training and 16,000 test observations. No threshold is chosen from test results.')
    p('The point of the comparison is to see the tradeoff, not assume that SMOTE must win. We look at missed failures, false alarms, recall, precision, ROC AUC, average precision and overall error. Scaling helps keep large-unit features from dominating SMOTE distances. Distances in many dimensions and interpolated histogram values still have limitations, and the changed training balance can affect probability calibration.')
    if results:
      story.append(PageBreak()); h('1. What the model results show')
      rows=[['Model / split','Error','ROC AUC','Recall','Precision','AP']]
      for name,m in results['models'].items():
        for split in ['train','test']:
          d=m[split]; rows.append([name+'/'+split]+[f'{d[k]:.4f}' for k in ['error','auc','recall','precision','ap']])
      story.append(table(rows))
      for name,m in results['models'].items():
        if 'oob_error' in m:
          p(f"{name}: the OOB error is {m['oob_error']:.5f}, compared with {m['test']['error']:.5f} on the test set. The test-minus-OOB difference is {m['test']['error']-m['oob_error']:+.5f}.")
        else:
          p(f"{name}: CV selected alpha={m['alpha']}. Its five-fold error is {m['cv_error']:.5f}, compared with test error {m['test']['error']:.5f}. The test-minus-CV difference is {m['test']['error']-m['cv_error']:+.5f}. The final model was fitted on {m['fit_rows']:,} rows.")
      for a,b in [('rf_plain','rf_weighted'),('xgb_plain','xgb_smote')]:
        if a not in results['models'] or b not in results['models']: continue
        x=results['models'][a]['test']; z=results['models'][b]['test']
        p(f"{b} versus {a}: test error changes by {z['error']-x['error']:+.5f}, recall by {z['recall']-x['recall']:+.5f}, and AUC by {z['auc']-x['auc']:+.5f}. False negatives change from {x['cm'][1][0]} to {z['cm'][1][0]}; false positives from {x['cm'][0][1]} to {z['cm'][0][1]}.")
      story.append(PageBreak()); h('1(b). Feature results')
      cv=__import__('pandas').read_csv(OUT/'feature_CV_all_170.csv',index_col=0)
      p('The 13 features chosen for the plots are: '+', '.join(results['data']['top13'])+'.')
      p('Several high-CV features have similar class medians or a median of zero in both classes. Rare extreme values account for much of their variation. The correlation matrix also shows overlapping information across predictors. Together, these plots explain why the highest-CV features should not automatically be treated as the best classifiers.')
      story.append(table([['Feature','Mean','Sample SD','CV']]+[[f]+[f'{cv.loc[f,k]:.5g}' for k in ['mean','sample_sd','CV']] for f in results['data']['top13']]))
      story.append(PageBreak()); h('1(b.iii). Correlation matrix')
      story.append(picture(OUT/'correlation.png',maxh=500))
      story.append(PageBreak()); h('1(b.iv). Visualizations')
      story.append(picture(OUT/'scatter_matrix.png',maxh=370)); story.append(picture(OUT/'boxplots.png',maxh=240))
      p('Open scatter_matrix.png separately to zoom into the 169 panels. The colors distinguish the classes. These plots help with exploration; they do not establish significance.')
      story.append(PageBreak()); h('1(c)-(f). Confusion matrices and ROC')
      for tag in results['models']: story.append(picture(OUT/(tag+'.png'),maxh=155))
      story.append(PageBreak()); h('1(e)-(f). Alpha selection')
      if len(results['models'])<4: p('The alpha-search results are not available in this copy. Run analysis.py with the required packages to complete this comparison.')
      for tag in ['plain','smote']:
        if not (OUT/f'cv_{tag}.csv').exists(): continue
        p('Five-fold CV: '+tag)
        df=__import__('pandas').read_csv(OUT/f'cv_{tag}.csv')
        story.append(table([df.columns.tolist()]+[[f'{v:.6g}' for v in r] for r in df.to_numpy()]))
        story.append(Spacer(1,15))
    for title,paragraphs in json.loads((ROOT/'theory.json').read_text()):
      story.append(PageBreak()); h(title)
      for s in paragraphs: p(s)
      if title.startswith('2.'):
        story.append(picture(OUT/'q2_lasso_trends.png',maxh=220))
        p('Learning illustration: these chosen curves show the usual trends as s increases. They are schematic, not measured APS results. The test-error curve reflects the balance between decreasing squared bias and increasing variance; the noise stays fixed.')
      if title.startswith('3.'):
        story.append(picture(OUT/'q3_ridge_lasso_geometry.png',maxh=200))
        p('Learning illustration: with identical predictors, RSS depends on their coefficient sum. The circle has one best contact point, while the highlighted diamond edge has many equally good coefficient pairs. The two pictures use the stated constraints, not a shared penalty value.')
      if title.startswith('4.'):
        story.append(picture(OUT/'q4_tree_aggregation.png',maxh=220))
        p('This plot uses the exact probabilities in the exercise. Six trees vote Red, but the average Red probability is only 0.45. That is why the two aggregation methods disagree.')
      if title.startswith('5.'): story.append(picture(OUT/'svm_geometry.png',maxh=255))
    story.append(PageBreak()); h('Sources and notes for studying')
    for s in [
      'APS data: UCI Machine Learning Repository, APS Failure at Scania Trucks, dataset 421. https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks',
      'Imputation: scikit-learn documentation, Imputation of missing values. https://scikit-learn.org/stable/modules/impute.html',
      'Random forest weighting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html',
      'XGBoost parameter reference (linear booster and alpha): https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html',
      'SMOTE: Chawla et al. (2002), JAIR 16, 321-357, DOI 10.1613/jair.953. Resampling leakage: https://imbalanced-learn.org/stable/common_pitfalls.html',
      'Textbook: James, Witten, Hastie and Tibshirani, An Introduction to Statistical Learning with Applications in R. https://www.statlearning.com/ . Chapter 6 section numbers vary across editions, and the homework uses both 6.6 and 6.8. Match these answers by the exercise number and topic: Chapter 6 Exercises 3 and 5, Chapter 8 Exercise 5, Chapter 9 Exercise 3.',
      'The separately labeled optional Extra Practice exercises are not included. Required Questions 1-5 are covered in the implementation and written discussion, subject to the model-tree interpretation and unexecuted-results limitations stated above.' if results is None else 'Question 6 is labeled Extra Practice and is not included. Keep the model-tree wording issue from 1(e) in mind when reviewing the implemented XGBoost models.',
      'To repeat the analysis, run python analysis.py. The full CV table and prediction arrays are saved separately so you can inspect the details without making the main report too long.'
    ]: p(s)
    if results and len(results['models'])==4:
      import textwrap
      from reportlab.lib.styles import ParagraphStyle
      story.append(PageBreak()); h('Appendix. Executable analysis code')
      p('The code below is also in analysis.py, so you can follow the calculations and run them again. The PDF layout is created by report.py. The analysis uses the full training set and all 170 predictors.')
      source=(ROOT/'analysis.py').read_text()
      wrapped='\n'.join('\n'.join(textwrap.wrap(line,width=102,replace_whitespace=False,drop_whitespace=False,subsequent_indent='    ')) if line else '' for line in source.splitlines())
      code_style=ParagraphStyle('Code',fontName='Courier',fontSize=8,leading=9.6)
      story.append(Preformatted(wrapped,code_style))
    dest=OUT/(('HW6_Results.pdf' if len(results['models'])==4 else 'HW6_Partial_Results.pdf') if results else 'HW6_Written_Answers.pdf')
    def footer(c,doc):
      c.setFont('Helvetica',8); c.setFillColor(colors.gray); c.drawRightString(550,26,str(doc.page))
    SimpleDocTemplate(str(dest),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=42,
      title='DSCI 552 Homework 6 - Study Notes').build(story,onFirstPage=footer,onLaterPages=footer)
    return dest
if __name__=='__main__': print(build())
