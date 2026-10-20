from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib import rcParams
from core import RES, ROOT
F=ROOT/'figures';F.mkdir(exist_ok=True)
rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':0.8})
BLUE='#1B4F72'; RED='#A93226'; GREY='#AEB6BF'; DGREY='#5D6D7E'; LBLUE='#D6EAF8'
def save(fig,name,dpi=600):
 fig.savefig(F/f'{name}.png',dpi=dpi,bbox_inches='tight',facecolor='white');fig.savefig(F/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)
# Fig5 main performance
s=pd.read_csv(RES/'main_summary.csv');methods=['absolute','order','traditional23','pfn','snhop'];labels=['Absolute\nfrequency','Full order','Traditional23\n+ FS','PFN core','SN-HOP']
f=s[s.method.isin(methods)&s.direction.isin(['1500_to_900','900_to_1500'])]
fig,ax=plt.subplots(figsize=(7.2,2.85));x=np.arange(len(methods));w=.34
for off,d,lab,col in [(-w/2,'1500_to_900','1500→900',DGREY),(w/2,'900_to_1500','900→1500',BLUE)]:
 vals=[float(f[(f.method==m)&(f.direction==d)].balanced_accuracy_pct.iloc[0]) for m in methods]
 ax.bar(x+off,vals,w,label=lab,color=col,edgecolor='black',linewidth=.35)
ax.axhline(33.33,ls='--',lw=.9,color=RED,label='Chance (33.33%)');ax.set_ylabel('Balanced accuracy (%)');ax.set_xticks(x,labels);ax.set_ylim(0,90);ax.grid(axis='y',alpha=.18);ax.legend(ncol=3,loc='upper center',frameon=False);fig.tight_layout();save(fig,'Fig5_main_DG')
# Fig6 taskwise SNHOP vs order/PFN
fig,axs=plt.subplots(1,2,figsize=(7.2,2.75),sharey=True)
for ax,d,title in zip(axs,['1500_to_900','900_to_1500'],['1500→900 rpm','900→1500 rpm']):
 for m,lab,col,mark in [('order','Full order',DGREY,'o'),('pfn','PFN core',RED,'s'),('snhop','SN-HOP',BLUE,'^')]:
  t=pd.read_csv(RES/f'{m}_tasks.csv');q=t[t.direction==d].copy();ax.plot(np.arange(1,11),q.target_ba*100,marker=mark,ms=4,lw=1.3,label=lab,color=col)
 ax.axhline(33.33,ls='--',lw=.7,color='black',alpha=.5);ax.set_title(title);ax.set_xlabel('Held-out bearing combination');ax.set_xticks(range(1,11));ax.grid(alpha=.16)
axs[0].set_ylabel('Balanced accuracy (%)');axs[0].set_ylim(0,100);axs[1].legend(frameon=False,loc='upper right');fig.tight_layout();save(fig,'Fig6_taskwise')
# Fig7 ablation + parameter selection
ab=pd.read_csv(RES/'ablation_summary.csv');order=['full','uniform_centers','peak_context','contrast_context','harmonics_only','context_only'];lab=['Full','Uniform','Peak+ctx','Contrast+ctx','Harmonics','Context'];vals=[float(ab[ab.variant==v].target_ba.iloc[0]) for v in order]
sel=pd.read_csv(RES/'snhop_selected.csv')
fig,axs=plt.subplots(1,3,figsize=(7.2,2.25),gridspec_kw={'width_ratios':[2.1,1,1]})
cols=[BLUE]+[GREY]*5;axs[0].bar(np.arange(6),vals,color=cols,edgecolor='black',linewidth=.35);axs[0].set_xticks(np.arange(6),lab,rotation=22,ha='right',fontsize=7.5);axs[0].set_ylabel('Mean BAcc (%)');axs[0].set_ylim(25,65);axs[0].grid(axis='y',alpha=.15);axs[0].set_title('(a) Representation ablation')
vc=sel.max_order.value_counts().sort_index();axs[1].bar([str(i) for i in [15,25,35,50]],[vc.get(i,0) for i in [15,25,35,50]],color=BLUE,edgecolor='black',linewidth=.35);axs[1].set_title('(b) $O_{max}$');axs[1].set_ylabel('Selections');axs[1].set_ylim(0,16)
vc=sel.halfwidth.value_counts().sort_index();axs[2].bar([str(i) for i in [.2,.4,.6]],[vc.get(i,0) for i in [.2,.4,.6]],color=RED,edgecolor='black',linewidth=.35);axs[2].set_title('(c) $\\Delta$');axs[2].set_ylim(0,16)
fig.tight_layout();save(fig,'Fig7_ablation_selection')
# Fig8 UDA + adaptation size
u=pd.read_csv(RES/'uda_summary.csv');fig,axs=plt.subplots(1,2,figsize=(7.2,2.75))
rows=[('absolute','CORAL','Abs.+CORAL'),('order','CORAL','Order+CORAL'),('pfn','CORAL','PFN+CORAL'),('snhop','NoDA','SN-HOP\nNoDA'),('snhop','target_standardization','Target\nstandard.'),('snhop','CORAL','SN-HOP\n+CORAL')]
vals=[]
for m,v,l in rows: vals.append(float(u[(u.method==m)&(u.kind=='uda')&(u.variant==v)].target_ba.iloc[0]))
axs[0].bar(np.arange(len(vals)),vals,color=[GREY,GREY,RED,LBUE if False else DGREY,GREY,BLUE],edgecolor='black',linewidth=.35);axs[0].set_xticks(np.arange(len(vals)),[r[2] for r in rows],rotation=25,ha='right');axs[0].set_ylabel('Mean BAcc (%)');axs[0].set_ylim(25,70);axs[0].grid(axis='y',alpha=.15);axs[0].set_title('(a) UDA comparison')
ad=u[(u.method=='snhop')&(u.kind=='adaptation_size')].sort_values('k');axs[1].plot(ad.k,ad.target_ba,marker='o',lw=1.6,color=BLUE);axs[1].set_xticks([0,1,3,5,10,15]);axs[1].set_xlabel('Unlabeled recordings per target bearing');axs[1].set_ylabel('Mean BAcc (%)');axs[1].set_ylim(50,70);axs[1].grid(alpha=.15);axs[1].set_title('(b) Adaptation-data sensitivity');fig.tight_layout();save(fig,'Fig8_UDA')
# Fig9 noise
n=pd.read_csv(RES/'noise_tasks.csv');clean=float(n[n.snr.astype(str)=='clean'].target_ba.mean()*100);rep=n[n.replicate>=0].groupby(['snr','replicate']).target_ba.mean().mul(100).reset_index();order_s=['20','10','6','3','0'];means=[rep[rep.snr.astype(str)==s].target_ba.mean() for s in order_s];stds=[rep[rep.snr.astype(str)==s].target_ba.std(ddof=1) for s in order_s]
fig,ax=plt.subplots(figsize=(3.9,3.0));x=np.arange(6);vals=[clean]+means;errs=[0]+stds;ax.errorbar(x,vals,yerr=errs,marker='o',lw=1.7,capsize=3,color=BLUE);ax.set_xticks(x,['Clean','20','10','6','3','0']);ax.set_xlabel('Target SNR (dB)');ax.set_ylabel('Mean balanced accuracy (%)');ax.set_ylim(45,62);ax.grid(alpha=.18);fig.tight_layout();save(fig,'Fig9_noise')
# Fig10 confusion matrices
cm1=np.loadtxt(RES/'snhop_mean_confusion.csv',delimiter=',');cm2=np.loadtxt(RES/'snhop_uda_mean_confusion.csv',delimiter=',');fig,axs=plt.subplots(1,2,figsize=(7.2,2.45));names=['Healthy','Outer','Inner']
for ax,cm,title in zip(axs,[cm1,cm2],['(a) Strict DG','(b) SN-HOP + CORAL']):
 ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(3),names);ax.set_yticks(range(3),names);ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_title(title)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>.55 else 'black')
fig.subplots_adjust(wspace=.42,left=.10,right=.98,bottom=.19,top=.87);save(fig,'Fig10_confusion')
