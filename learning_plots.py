"""Illustrations for Questions 2-4; these do not add or replace APS model results."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def lasso_trends():
    # These chosen curves illustrate typical trends; they are not fitted results.
    s=np.linspace(0,5,300)
    bias2=2.4*np.exp(-.9*s)
    variance=.075*s**2
    noise=np.full_like(s,.35)
    training=1.8*np.exp(-.8*s)+.2
    test=bias2+variance+noise
    fig,axes=plt.subplots(1,2,figsize=(9,3.2))
    axes[0].plot(s,training,label='Training error',color='#216b9a')
    axes[0].plot(s,test,label='Typical test error',color='#b32834')
    axes[0].set(xlabel='Constraint budget s (more flexibility to the right)',
                ylabel='Relative error',title='Training and test behavior')
    axes[1].plot(s,bias2,label='Squared bias',color='#216b9a')
    axes[1].plot(s,variance,label='Variance',color='#b32834')
    axes[1].plot(s,noise,'--',label='Irreducible error',color='#555555')
    axes[1].set(xlabel='Constraint budget s',ylabel='Relative contribution',
                title='Why the usual test-error curve bends upward')
    for ax in axes:
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Question 2: schematic learning illustration, not APS measurements',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'q2_lasso_trends.png',dpi=180);plt.close(fig)

def identical_predictor_geometry():
    # Illustrative RSS depends only on beta_1 + beta_2, just as in Question 3.
    u=np.linspace(-2.3,3.4,250);b1,b2=np.meshgrid(u,u)
    rss=(b1+b2-3)**2
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for ax in axes:
        ax.contour(b1,b2,rss,levels=[.15,1,4,9,16],colors='#aebac3',linewidths=.7)
        ax.plot(u,3-u,'--',color='#666666',label='Least-squares line: sum = 3')
        ax.set(xlim=(-2.3,3.4),ylim=(-2.3,3.4),aspect='equal',
               xlabel=r'$\beta_1$',ylabel=r'$\beta_2$')
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
    angle=np.linspace(0,2*np.pi,300);radius=np.sqrt(2)
    axes[0].fill(radius*np.cos(angle),radius*np.sin(angle),color='#bdd9eb',alpha=.6)
    axes[0].plot(radius*np.cos(angle),radius*np.sin(angle),color='#216b9a')
    axes[0].scatter([1],[1],color='#b32834',s=45,zorder=5)
    axes[0].annotate('Unique contact: (1,1)',(1,1),xytext=(-1.9,2.4),
                     arrowprops={'arrowstyle':'->'},fontsize=8)
    axes[0].set_title(r'Ridge: $\beta_1^2+\beta_2^2\leq2$',fontsize=10)
    diamond=np.array([[0,2],[2,0],[0,-2],[-2,0],[0,2]])
    axes[1].fill(diamond[:,0],diamond[:,1],color='#bdd9eb',alpha=.6)
    axes[1].plot(diamond[:,0],diamond[:,1],color='#216b9a')
    axes[1].plot([0,2],[2,0],color='#b32834',lw=4,label='Equally good coefficient pairs')
    axes[1].scatter([.3,1,1.7],[1.7,1,.3],color='#b32834',s=25,zorder=5)
    axes[1].set_title(r'Lasso: $|\beta_1|+|\beta_2|\leq2$',fontsize=10)
    axes[1].legend(fontsize=7,loc='lower left')
    fig.suptitle(r'Question 3: illustrative RSS = $(\beta_1+\beta_2-3)^2$',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'q3_ridge_lasso_geometry.png',dpi=180);plt.close(fig)

def tree_aggregation():
    # These are the exact ten probabilities given in the textbook exercise.
    red=np.array([.1,.15,.2,.2,.55,.6,.6,.65,.7,.75])
    votes=int((red>.5).sum());mean=float(red.mean())
    fig,axes=plt.subplots(1,2,figsize=(9,3.2),gridspec_kw={'width_ratios':[2,1]})
    axes[0].bar(np.arange(1,11),red,color=np.where(red>.5,'#b32834','#2f7c54'))
    axes[0].axhline(.5,color='#444444',linestyle='--',lw=1)
    axes[0].set(xticks=np.arange(1,11),ylim=(0,1),xlabel='Tree',
                ylabel='Estimated probability of Red',title=f'Majority vote: {votes} Red, {10-votes} Green -> Red')
    for i,p in enumerate(red,1):axes[0].text(i,p+.025,f'{p:g}',ha='center',fontsize=8)
    axes[1].bar(['Red','Green'],[mean,1-mean],color=['#b32834','#2f7c54'])
    axes[1].axhline(.5,color='#444444',linestyle='--',lw=1)
    axes[1].set(ylim=(0,1),ylabel='Average class probability',title='Probability average -> Green')
    for i,p in enumerate([mean,1-mean]):axes[1].text(i,p/2,f'{p:.2f}',ha='center',va='center',color='white',fontsize=11)
    fig.suptitle('Question 4: votes and confidence can lead to different predictions',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'q4_tree_aggregation.png',dpi=180);plt.close(fig)

def make_all():
    OUT.mkdir(exist_ok=True)
    lasso_trends();identical_predictor_geometry();tree_aggregation()

if __name__=='__main__':make_all()
