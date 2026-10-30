"""Generate all manuscript figures from observed data and saved experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import brier_score_loss
from matplotlib.text import Text
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from cycler import cycler
import textwrap

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
# User-confirmed green-purple palette; identical methods retain identical colors.
C = {"adwinDTARC": "#5E8F6A", "adwinER": "#A987C9", "phDTARC": "#2F4F3A",
     "phER": "#A7B688", "periodicER": "#9DC499", "periodic": "#B7C5A7",
     "static": "#6B7280", "logistic": "#DCC6EA"}
plt.rcParams.update({"font.family":"Arial", "pdf.fonttype":42,
                     "axes.prop_cycle":cycler(color=list(C.values()))})
METHOD_ORDER = ["static", "logistic", "periodic", "periodicER", "adwinER", "phER", "adwinDTARC", "phDTARC"]


def save(fig, name):
    # Submitted width 7.5 in; reading manuscript uses ~6.27 in. Minimum
    # native font 9.6 pt gives >=8 pt in that reading manuscript.
    w,h=fig.get_size_inches()
    heights={'fig03_prequential':5.2,'fig04_event_recovery':5.5,
             'fig06_cross_market':4.8,'fig08_wealth_drawdown':5.5,
             'fig09_sensitivity':6.5,'fig11_ablation_horizon':4.5,
             'fig13_robustness_atlas':6.2,'fig15_mechanism_audit':5.5,
             'fig16_block_sensitivity':5.3}
    fig.set_size_inches(7.5, heights.get(name,min(8.0,max(4.0,h))))
    for artist in fig.findobj(match=Text):
        artist.set_fontsize(min(12,max(9.6,artist.get_fontsize())))
    for ax in fig.axes:
        title=ax.get_title(loc="left") or ax.get_title()
        if title:
            loc="left" if ax.get_title(loc="left") else "center"
            ax.set_title(textwrap.fill(title, 32 if len(fig.axes)>2 else 40),loc=loc,fontsize=10.5)
    fig.tight_layout(pad=1.1)
    fig.savefig(FIG / f"{name}.pdf",bbox_inches=None)
    fig.savefig(FIG / f"{name}.png", dpi=400,bbox_inches=None)
    fig.savefig(FIG / f"{name}.tif", dpi=400,bbox_inches=None,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    print("figure", name, flush=True)


def box(ax, xy, wh, text, color, fontsize=8):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                       facecolor=color, edgecolor="#263238", linewidth=0.8)
    ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color="#455A64"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                linewidth=1.0, color=color))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 5.8)); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, (.03,.67), (.16,.17), "Matured stream\n$\\mathcal{D}_{t-h}$", "#E3F2FD", 10)
    box(ax, (.25,.67), (.17,.17), "Frozen prequential\nprediction $\\hat p_t$", "#E8F5E9", 10)
    box(ax, (.49,.67), (.17,.17), "Error stream\n$e_t=\\mathbb{1}[\\hat y_t\\neq y_t]$", "#FFF3E0", 10)
    box(ax, (.74,.67), (.20,.17), "ADWIN / Page–Hinkley\nchange signal", "#FCE4EC", 10)
    for a,b in [((.19,.755),(.25,.755)),((.42,.755),(.49,.755)),((.66,.755),(.74,.755))]: arrow(ax,a,b)
    box(ax, (.06,.35), (.18,.16), "Effect size $s_t$\nAbrupt vs. gradual", "#F3E5F5", 9)
    box(ax, (.31,.35), (.18,.16), "Type-conditioned quotas\nrecent + diverse", "#EDE7F6", 9)
    box(ax, (.56,.35), (.16,.16), "Anchors + uniform\nmemory", "#E0F2F1", 9)
    box(ax, (.79,.35), (.16,.16), "Proximal update\n$\\lambda(s_t)$", "#FFF8E1", 9)
    arrow(ax,(.82,.67),(.16,.51)); arrow(ax,(.24,.43),(.31,.43)); arrow(ax,(.49,.43),(.56,.43)); arrow(ax,(.72,.43),(.79,.43))
    box(ax, (.25,.08), (.50,.14), "Next-period predictor $\\theta_{t+1}$ — no sample enters training before its label matures", "#ECEFF1", 10)
    arrow(ax,(.87,.35),(.67,.22)); arrow(ax,(.40,.35),(.42,.22))
    ax.text(.03,.94,"DTARC: prediction-first adaptation under delayed feedback",fontsize=15,weight="bold",color="#263238")
    ax.text(.03,.895,"Drift type changes the replay composition; the detector determines whether an update occurs.",fontsize=10,color="#546E7A")
    save(fig,"fig01_architecture")


def psi(train, test, features):
    vals = {}
    for f in features:
        a = train[f].replace([np.inf,-np.inf],np.nan).dropna()
        b = test[f].replace([np.inf,-np.inf],np.nan).dropna()
        if len(a)==0 or len(b)==0: vals[f]=np.nan; continue
        qs = np.unique(np.quantile(a,np.linspace(0,1,11)))
        if len(qs)<3: vals[f]=0.; continue
        qs[0],qs[-1]=-np.inf,np.inf
        pa=np.histogram(a,bins=qs)[0]/len(a); pb=np.histogram(b,bins=qs)[0]/len(b)
        pa=np.clip(pa,1e-4,None);pb=np.clip(pb,1e-4,None)
        vals[f]=float(np.sum((pb-pa)*np.log(pb/pa)))
    return vals


def drift_atlas():
    info=json.loads((DATA/"dataset_summary.json").read_text()); features=info["features"]
    data={m:pd.read_csv(DATA/f"{m.lower()}_processed.csv",parse_dates=["date"]) for m in ["CN","TH"]}
    mat=pd.DataFrame({m:psi(d[d.split=="train"],d[d.split=="test"],features) for m,d in data.items()},index=features)
    mat.to_csv(RESULTS/"feature_psi.csv")
    fig=plt.figure(figsize=(7.5,8));gs=fig.add_gridspec(3,1,height_ratios=[4.1,1,1],hspace=.4)
    ax=fig.add_subplot(gs[0,0])
    # Vector cells keep the figure fully editable in journal production.
    im=ax.pcolormesh(np.arange(3)-.5,np.arange(len(mat)+1)-.5,mat.values,
                     cmap=LinearSegmentedColormap.from_list("observed_psi",["#F7FBF5","#C7E0D1","#5E8F6A","#2F4F3A"]),vmin=0,vmax=max(.3,np.nanmax(mat.values)),
                     edgecolors="white",linewidth=.35,rasterized=False)
    ax.set_xlim(-.5,1.5);ax.set_ylim(len(mat)-.5,-.5)
    ax.set_yticks(range(len(mat)),mat.index);ax.set_xticks([0,1],["China A-shares","Thailand SET"])
    for i in range(len(mat)):
        for j in range(2): ax.text(j,i,f"{mat.iloc[i,j]:.2f}",ha="center",va="center",fontsize=9.6,color="white" if im.norm(mat.iloc[i,j])>.6 else "#2F4F3A")
    ax.set_title("A  Feature PSI: training → test",loc="left",weight="bold")
    cb=fig.colorbar(im,ax=ax,pad=.01);cb.set_label("PSI")
    for j,m in enumerate(["CN","TH"]):
        ax=fig.add_subplot(gs[j+1,0]);d=data[m]
        daily=d.groupby("date").agg(ret=("ret_1","mean"),mom=("mom_5","mean"))
        ax.plot(daily.index,daily.ret.rolling(20).std()*np.sqrt(252),color=C["periodicER"],lw=1.3,label="20-day annualized volatility")
        ax2=ax.twinx();ax2.plot(daily.index,daily.mom.rolling(20).mean(),color=C["adwinER"],lw=1,alpha=.8,label="20-day mean momentum");ax2.set_ylabel("Momentum",color=C["adwinER"])
        ax.axvspan(pd.Timestamp("2026-01-01"),pd.Timestamp("2026-09-30"),color="#B0BEC5",alpha=.18)
        ax.set_title(f"{'B' if j==0 else 'C'}  {m}: observed market state",loc="left",weight="bold")
        ax.set_ylabel("Volatility",color=C["periodicER"]);ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    save(fig,"fig02_drift_atlas")


def daily_metrics(pred):
    rows=[]
    for keys,g in pred.groupby(["market","method","seed","date"]):
        y,p=g.y.values,g.p.values
        rows.append(dict(zip(["market","method","seed","date"],keys))|{
            "brier":brier_score_loss(y,p),"accuracy":np.mean((p>=.5)==y)})
    return pd.DataFrame(rows)


def prequential(pred,events):
    daily=daily_metrics(pred);daily.to_csv(RESULTS/"daily_prequential_h5.csv",index=False)
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=False)
    for ax,m in zip(axs,["CN","TH"]):
        for method in ["static","periodicER","adwinER","adwinDTARC"]:
            q=daily[(daily.market==m)&(daily.method==method)].groupby("date").brier.mean().rolling(10,min_periods=3).mean()
            ax.plot(q.index,q.values,label=method,color=C[method],ls={"static":":","periodicER":"--","adwinER":"-.","adwinDTARC":"-"}[method],lw=1.5 if method=="adwinDTARC" else 1.1)
        ev=events[(events.market==m)&(events.method=="adwinDTARC")]
        for x in sorted(pd.to_datetime(ev.date).unique()): ax.axvline(x,color="#A987C9",alpha=.25,lw=.8)
        ax.set_title(f"{m}: rolling Brier; purple lines mark ADWIN updates",loc="left",weight="bold")
        ax.set_ylabel("Brier score");ax.grid(alpha=.18);ax.legend(ncol=4,frameon=False)
    axs[-1].set_xlabel("Test date (2026)");fig.tight_layout();save(fig,"fig03_prequential")


def recovery(pred,events):
    daily=daily_metrics(pred)
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
    offsets=np.arange(-10,16)
    for col,m in enumerate(["CN","TH"]):
        for method,color in [("adwinDTARC",C["adwinDTARC"]),("static",C["static"])]:
            curves=[]
            for seed in sorted(pred.seed.unique()):
                d=daily[(daily.market==m)&(daily.method==method)&(daily.seed==seed)].sort_values("date").reset_index(drop=True)
                pos={x:i for i,x in enumerate(d.date)}
                ev=events[(events.market==m)&(events.method=="adwinDTARC")&(events.seed==seed)]
                for date in pd.to_datetime(ev.date):
                    matches=[x for x in pos if x==date]
                    if not matches: continue
                    i=pos[matches[0]]; idx=i+offsets
                    vals=np.full(len(offsets),np.nan);ok=(idx>=0)&(idx<len(d));vals[ok]=d.brier.values[idx[ok]]
                    curves.append(vals)
            arr=np.asarray(curves);mean=np.nanmean(arr,axis=0);se=np.nanstd(arr,axis=0)/np.sqrt(np.sum(np.isfinite(arr),axis=0).clip(1))
            axs[0,col].plot(offsets,mean,label=method,color=color);axs[0,col].fill_between(offsets,mean-1.96*se,mean+1.96*se,color=color,alpha=.15)
            axs[1,col].plot(offsets,np.nanmean(arr,axis=0)-np.nanmean(arr[:,offsets<0]),color=color)
        axs[0,col].axvline(0,color="black",ls="--",lw=.8);axs[1,col].axvline(0,color="black",ls="--",lw=.8)
        axs[0,col].set_title(f"{m}: event-aligned Brier (n events pooled over seeds)",weight="bold")
        axs[1,col].set_title(f"{m}: centered event-aligned Brier",weight="bold")
        for ax in axs[:,col]:ax.grid(alpha=.18)
    axs[0,0].set_ylabel("Brier score");axs[1,0].set_ylabel("Brier minus pre-event mean")
    for ax in axs[-1]:ax.set_xlabel("Trading days relative to detected update")
    axs[0,0].legend(frameon=False);fig.tight_layout();save(fig,"fig04_event_recovery")


def replay_and_events(events):
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={"width_ratios":[.85,1.4]})
    quotas=pd.DataFrame({"Abrupt":{"Recent":.5,"Diverse":.2,"Anchor":.2,"Uniform":.1},"Gradual":{"Recent":.2,"Diverse":.5,"Anchor":.2,"Uniform":.1}}).T
    bottom=np.zeros(2)
    for kind,color in zip(quotas.columns,["#5E8F6A","#DCC6EA","#9DC499","#A7B688"]):
        axs[0].barh(quotas.index,quotas[kind],left=bottom,label=kind,color=color)
        for i,v in enumerate(quotas[kind]):axs[0].text(bottom[i]+v/2,i,f"{v:.0%}",ha="center",va="center",fontsize=8,color="white" if kind=="Recent" else "#2F4F3A")
        bottom+=quotas[kind].values
    axs[0].set_xlim(0,1);axs[0].set_xlabel("Replay quota");axs[0].set_title("A  Type-conditioned allocation",loc="left",weight="bold");axs[0].legend(ncol=2,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.35))
    summary=events[events.method.isin(["adwinDTARC","phDTARC"])].groupby(["market","detector","type"]).size().rename("count").reset_index()
    labels=[];vals=[];colors=[]
    for m in ["CN","TH"]:
        for det in ["adwin","ph"]:
            for typ in ["abrupt","gradual"]:
                q=summary[(summary.market==m)&(summary.detector==det)&(summary.type==typ)]
                labels.append(f"{m}-{det}\n{typ}");vals.append(0 if q.empty else q['count'].iloc[0]/5);colors.append("#A987C9" if typ=="abrupt" else "#5E8F6A")
    axs[1].bar(range(len(vals)),vals,color=colors);axs[1].set_xticks(range(len(vals)),labels,rotation=45,ha="right");axs[1].set_ylabel("Mean triggered updates per seed")
    axs[1].set_title("B  DTARC update types",loc="left",weight="bold");axs[1].grid(axis="y",alpha=.18)
    fig.tight_layout();save(fig,"fig05_replay_and_events")


def cross_market(metrics):
    fig,axs=plt.subplots(1,2,figsize=(11,5.6),sharey=True)
    avg=metrics.groupby(["market","method"])[["mcc","auc"]].mean()
    for ax,metric in zip(axs,["mcc","auc"]):
        for i,method in enumerate(METHOD_ORDER):
            cn=avg.loc[("CN",method),metric];th=avg.loc[("TH",method),metric]
            ax.annotate("",xy=(th,i),xytext=(cn,i),arrowprops=dict(arrowstyle="->",color="#B0BEC5",lw=1))
            ax.scatter(cn,i,color="#5E8F6A",s=42,zorder=3);ax.scatter(th,i,color="#A987C9",s=42,zorder=3)
        ax.axvline(.5 if metric=="auc" else 0,color="black",lw=.8,ls="--");ax.grid(axis="x",alpha=.2)
        ax.set_xlabel(metric.upper());ax.set_title(f"Mean test {metric.upper()}: China → Thailand",weight="bold")
    axs[0].set_yticks(range(len(METHOD_ORDER)),METHOD_ORDER);axs[0].scatter([],[],color="#5E8F6A",label="CN");axs[0].scatter([],[],color="#A987C9",label="TH");axs[0].legend(frameon=False)
    fig.tight_layout();save(fig,"fig06_cross_market")


def pareto(metrics,runtime):
    avg=metrics.groupby(["market","method"]).mcc.mean();rt=runtime.groupby(["market","method"])[["seconds","updates"]].mean()
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    symbols=dict(zip(METHOD_ORDER,["o","s","^","v","D","P","X","*"]))
    for ax,m in zip(axs,["CN","TH"]):
        for method in METHOD_ORDER:
            if (m, method) not in rt.index:
                continue  # logistic has no comparable online-update phase
            x=rt.loc[(m,method),"seconds"];y=avg.loc[(m,method)]
            if not np.isfinite(x): continue
            size=30+5*rt.loc[(m,method),"updates"]
            ax.scatter(x,y,s=size,color=C[method],marker=symbols[method],alpha=.85,edgecolor="#2F4F3A",linewidth=.5,label=method)
        ax.axhline(0,color="black",lw=.8,ls="--");ax.set_xlabel("Online loop time (s; initial fit excluded)");ax.set_ylabel("MCC");ax.set_title(f"{m}: adaptation cost–performance map",weight="bold");ax.grid(alpha=.18)
    axs[0].legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(1.1,-.47))
    fig.tight_layout();save(fig,"fig07_cost_pareto")


def wealth(curves):
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex="col")
    for col,m in enumerate(["CN","TH"]):
        for method in ["static","logistic","periodicER","adwinER","adwinDTARC"]:
            q=curves[(curves.market==m)&(curves.method==method)&(curves.cost_bps==10)&(curves["style"]=="long_short")]
            w=q.groupby("date").wealth.mean();dd=1-w/w.cummax().clip(lower=1)
            axs[0,col].plot(w.index,w.values,label=method,color=C[method],lw=1.3)
            axs[1,col].fill_between(dd.index,dd.values,color=C[method],alpha=.35);axs[1,col].plot(dd.index,dd.values,color=C[method],lw=.8)
        q=curves[(curves.market==m)&(curves.method=="equal_weight")&(curves.cost_bps==10)&(curves["style"]=="benchmark")]
        w=q.groupby("date").wealth.mean();dd=1-w/w.cummax().clip(lower=1)
        axs[0,col].plot(w.index,w.values,label="equal_weight",color="black",lw=1.4,ls="--")
        axs[1,col].plot(dd.index,dd.values,color="black",lw=1.0,ls="--")
        axs[0,col].axhline(1,color="black",lw=.7);axs[0,col].set_title(f"{m}: mean long–short wealth, 10 bps",weight="bold");axs[1,col].set_title(f"{m}: underwater drawdown",weight="bold")
        for ax in axs[:,col]:ax.grid(alpha=.18)
    axs[0,0].set_ylabel("Wealth (initial = 1)");axs[1,0].set_ylabel("Drawdown");axs[0,0].legend(frameon=False,ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("Formation date (2026)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout();save(fig,"fig08_wealth_drawdown")


def sensitivity(metrics):
    # MCC and AUC occupy separate rows so their scales are never conflated.
    fig,axs=plt.subplots(4,4,figsize=(13,10),sharey="row")
    factors=["buffer","replay","online_lr","ph_threshold"]
    for row,(m,metric) in enumerate([(m,z) for m in ["CN","TH"] for z in ["mcc","auc"]]):
        for col,factor in enumerate(factors):
            ax=axs[row,col];q=metrics[(metrics.market==m)&(metrics.factor==factor)].groupby("value")[metric].mean().reset_index()
            x=np.arange(len(q));ax.plot(x,q[metric],marker="o" if m=="CN" else "s",color="#5E8F6A" if m=="CN" else "#A987C9")
            ax.axhline(0 if metric=="mcc" else .5,color="black",lw=.6,ls="--")
            ax.set_xticks(x,[f"{v:.0e}" if factor=="online_lr" else f"{v:g}" for v in q.value]);ax.grid(alpha=.18)
            if row==0: ax.set_title(["Memory B","Replay K","Online rate","PH threshold"][col],weight="bold")
            if row==3: ax.set_xlabel(["B","K",r"$\eta_{\mathrm{on}}$",r"$\tau_{\mathrm{PH}}$"][col])
        axs[row,0].set_ylabel(f"{m} {metric.upper()}")
    fig.tight_layout();save(fig,"fig09_sensitivity")


def bootstrap_forest(boot):
    fig,axs=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,metric in zip(axs,["mcc","auc"]):
        comps=sorted(boot.comparison.unique());n=len(comps)
        for j,m in enumerate(["CN","TH"]):
            for i,comp in enumerate(comps):
                q=boot[(boot.market==m)&(boot.comparison==comp)&(boot.metric==metric)].iloc[0]
                y=i+(j-.5)*.22;color="#5E8F6A" if m=="CN" else "#A987C9"
                ax.plot([q["ci_2.5"],q["ci_97.5"]],[y,y],color=color,lw=1.5);ax.scatter(q.mean_difference,y,color=color,s=38,label=m if i==0 else None)
        ax.axvline(0,color="black",ls="--",lw=.8);ax.set_title(f"DTARC paired difference in {metric.upper()}",weight="bold");ax.set_xlabel("Mean difference (95% moving-block CI)");ax.grid(axis="x",alpha=.18)
    axs[0].set_yticks(range(n),[x.replace("adwinDTARC - ","DTARC − ") for x in comps]);axs[0].legend(frameon=False);fig.tight_layout();save(fig,"fig10_bootstrap")


def ablation_horizon(ab,h1,h5):
    fig,axs=plt.subplots(1,2,figsize=(12,5.2),gridspec_kw={"width_ratios":[1.15,1]})
    avg=ab.groupby(["market","method"]).mcc.mean().unstack(0)
    order=["adwinDTARC","DTARC_no_anchor","DTARC_no_diversity","DTARC_no_typing","DTARC_no_proximal"]
    x=np.arange(len(order));w=.35
    for i,m in enumerate(["CN","TH"]):axs[0].bar(x+(i-.5)*w,[avg.loc[z,m] for z in order],w,label=m,color=["#5E8F6A","#A987C9"][i])
    axs[0].set_xticks(x,[z.replace("DTARC_","").replace("adwinDTARC","Full") for z in order],rotation=25,ha="right");axs[0].axhline(0,color="black",lw=.7);axs[0].set_ylabel("MCC");axs[0].set_title("A  Five-day component ablation",loc="left",weight="bold");axs[0].legend(frameon=False)
    a=h5.groupby(["market","method"]).mcc.mean();b=h1.groupby(["market","method"]).mcc.mean()
    for i,m in enumerate(["CN","TH"]):
        xx=np.arange(len(METHOD_ORDER));axs[1].plot(xx,[a.loc[(m,z)] for z in METHOD_ORDER],marker="o",label=f"{m}, h=5",color=["#5E8F6A","#A987C9"][i]);axs[1].plot(xx,[b.loc[(m,z)] for z in METHOD_ORDER],marker="s",ls="--",label=f"{m}, h=1",color=["#5E8F6A","#A987C9"][i],alpha=.65)
    axs[1].set_xticks(np.arange(len(METHOD_ORDER)),METHOD_ORDER,rotation=35,ha="right");axs[1].axhline(0,color="black",lw=.7);axs[1].set_title("B  Horizon robustness",loc="left",weight="bold");axs[1].legend(frameon=False,ncol=2,fontsize=8)
    for ax in axs:ax.grid(axis="y",alpha=.18)
    fig.tight_layout();save(fig,"fig11_ablation_horizon")


def boundary_audit_figure():
    stitched=pd.read_csv(RESULTS/"boundary_audit_stitched.csv")
    single=pd.read_csv(RESULTS/"boundary_audit_single_source.csv")
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    x=np.arange(len(stitched));w=.36
    axs[0].bar(x-w/2,stitched.boundary_return,w,label="Stitched adjustment scales",color="#A987C9")
    axs[0].bar(x+w/2,single.boundary_return,w,label="Single-source forward adjusted",color="#5E8F6A")
    axs[0].set_yscale("symlog",linthresh=.01);axs[0].axhline(0,color="black",lw=.7)
    axs[0].set_xticks(x,[c.replace("sh.","").replace("sz.","") for c in stitched.code],rotation=45,ha="right")
    axs[0].set_ylabel("Boundary simple return (symlog scale)");axs[0].set_title("A  First 2026 return by stock",loc="left",weight="bold");axs[0].legend(frameon=False,fontsize=8)
    vals=[];labels=[]
    for name,d in [("Stitched",stitched),("Single\nsource",single)]:
        vals.extend([d.pre_return_sd.values,d.post_return_sd.values]);labels.extend([f"{name}\npre",f"{name}\npost"])
    axs[1].boxplot(vals,labels=labels,showfliers=True);axs[1].set_yscale("log");axs[1].set_ylabel("20-day return SD (log scale)")
    axs[1].set_title("B  Distribution shift around the boundary",loc="left",weight="bold");axs[1].grid(axis="y",alpha=.18)
    fig.tight_layout();save(fig,"fig12_boundary_audit")


def robustness_figure():
    full=pd.read_csv(RESULTS/"metrics_h5.csv");matched=pd.read_csv(RESULTS/"metrics_matched_replay.csv")
    execution=pd.read_csv(RESULTS/"metrics_execution_aligned.csv");loo=pd.read_csv(RESULTS/"metrics_loo.csv")
    qfq=pd.read_csv(RESULTS/"metrics_sina_qianfuquan.csv");hfq=pd.read_csv(RESULTS/"metrics_sina_houfuquan.csv")
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    def bars(ax,d,methods,title,markets=["CN","TH"]):
        avg=d.groupby(["market","method"]).mcc.mean();x=np.arange(len(methods));w=.35
        for j,m in enumerate(markets):ax.bar(x+(j-.5)*w,[avg.loc[(m,z)] for z in methods],w,label=m,color=["#5E8F6A","#A987C9"][j])
        ax.set_xticks(x,methods,rotation=20,ha="right");ax.axhline(0,color="black",lw=.7);ax.set_title(title,loc="left",weight="bold");ax.set_ylabel("MCC");ax.grid(axis="y",alpha=.18);ax.legend(frameon=False)
    bars(axs[0,0],pd.concat([full,matched]),["adwinDTARC","adwinER","adwinERstrat","DTARC_no_provenance"],"A  Matched replay mechanisms")
    bars(axs[0,1],execution,["static","periodicER","adwinER","adwinDTARC"],"B  Execution-aligned targets")
    q=qfq.assign(panel="Forward");h=hfq.assign(panel="Backward");adj=pd.concat([q,h]);avg=adj.groupby(["panel","method"]).mcc.mean()
    methods=["static","periodicER","adwinER","adwinDTARC"];x=np.arange(len(methods));w=.35
    for j,panel in enumerate(["Forward","Backward"]):axs[1,0].bar(x+(j-.5)*w,[avg.loc[(panel,z)] for z in methods],w,label=panel,color=["#9DC499","#A987C9"][j])
    axs[1,0].set_xticks(x,methods,rotation=20,ha="right");axs[1,0].set_title("C  China: adjustment-convention audit",loc="left",weight="bold");axs[1,0].set_ylabel("MCC");axs[1,0].legend(frameon=False);axs[1,0].grid(axis="y",alpha=.18)
    bars(axs[1,1],loo,["static","adwinER","adwinDTARC"],"D  Leave-one-out market context")
    fig.tight_layout();save(fig,"fig13_robustness_atlas")


def protocol_figure():
    d=pd.read_csv(RESULTS/"metrics_protocol_degradation.csv");order=["correct","no_purge","premature_labels","random_row_split"]
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for ax,metric in zip(axs,["mcc","auc"]):
        vals=[d[d.method==z][metric].iloc[0] for z in order];colors=["#9DC499","#DCC6EA","#DCC6EA","#A987C9"]
        ax.bar(range(4),vals,color=colors);ax.set_xticks(range(4),["Correct","No purge","Premature\nlabels","Random row\nsplit"])
        ax.axhline(0 if metric=="mcc" else .5,color="black",lw=.7,ls="--");ax.set_ylabel(metric.upper());ax.set_title(f"Validation {metric.upper()} under deliberate protocol failures",weight="bold");ax.grid(axis="y",alpha=.18)
    fig.tight_layout();save(fig,"fig14_protocol_degradation")


def mechanism_figure():
    d=pd.read_csv(RESULTS/"metrics_mechanism_h5.csv");rt=pd.read_csv(RESULTS/"runtime_mechanism_h5.csv");ev=pd.read_csv(RESULTS/"drift_events_h5.csv")
    fig,axs=plt.subplots(2,3,figsize=(13,7.5))
    for m,color in [("CN","#5E8F6A"),("TH","#A987C9")]:
        q=ev[(ev.market==m)&(ev.method=="adwinDTARC")];axs[0,0].hist(q.severity,bins=12,alpha=.45,label=f"{m}-ADWIN",color=color)
        q=ev[(ev.market==m)&(ev.method=="phDTARC")];axs[0,0].hist(q.severity,bins=12,alpha=.35,label=f"{m}-PH",color=color,histtype="step",lw=1.4)
    axs[0,0].set_xscale("log");axs[0,0].set_xticks([.2,.5,1,2],["0.2","0.5","1","2"]);axs[0,0].minorticks_off();axs[0,0].set_xlabel("Severity (log scale)");axs[0,0].set_ylabel("Triggered updates");axs[0,0].set_title("A  Empirical severity distribution",loc="left",weight="bold");axs[0,0].legend(frameon=False,fontsize=8)
    for col,factor in enumerate(["adwin_delta","abrupt_threshold"],start=1):
        for m,color in [("CN","#5E8F6A"),("TH","#A987C9")]:
            q=rt[(rt.market==m)&(rt.factor==factor)].groupby("value").updates.mean();axs[0,col].plot(q.index,q.values,marker="o",label=m,color=color)
        axs[0,col].set_xlabel(factor);axs[0,col].set_ylabel("Mean updates");axs[0,col].set_title(f"B{col}  Update coverage vs. {factor}",loc="left",weight="bold");axs[0,col].grid(alpha=.18)
    axs[0,1].legend(frameon=False)
    for col,factor in enumerate(["adwin_delta","abrupt_threshold","prox"]):
        for m,color in [("CN","#5E8F6A"),("TH","#A987C9")]:
            q=d[(d.market==m)&(d.factor==factor)].groupby("value").mcc.mean();axs[1,col].plot(q.index,q.values,marker="o",label=m,color=color)
        axs[1,col].axhline(0,color="black",lw=.7,ls="--");axs[1,col].set_xlabel(factor);axs[1,col].set_ylabel("MCC");axs[1,col].set_title(f"D{col+1}  Discrimination vs. {factor}",loc="left",weight="bold");axs[1,col].grid(alpha=.18)
    axs[1,0].legend(frameon=False);fig.tight_layout();save(fig,"fig15_mechanism_audit")


def block_figure():
    d=pd.read_csv(RESULTS/"paired_block_bootstrap_sensitivity_h5.csv")
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for row,m in enumerate(["CN","TH"]):
        for col,metric in enumerate(["mcc","auc"]):
            ax=axs[row,col]
            for j,comp in enumerate(["adwinER","static","periodicER"]):
                q=d[(d.market==m)&(d.comparator==comp)&(d.metric==metric)].sort_values("block_days");x=q.block_days.values+(j-1)*.7
                ax.errorbar(x,q.mean_difference,yerr=[q.mean_difference-q["ci_2.5"],q["ci_97.5"]-q.mean_difference],marker=["o","s","D"][j],color=C[comp],capsize=3,label=comp)
            ax.axhline(0,color="black",lw=.7,ls="--");ax.set_xticks([5,10,20]);ax.set_xlabel("Moving-block length (trading days)");ax.set_ylabel(f"DTARC difference in {metric.upper()}");ax.set_title(f"{m}: block-length sensitivity",weight="bold");ax.grid(alpha=.18)
    axs[0,0].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,"fig16_block_sensitivity")


def main():
    pred=pd.read_csv(RESULTS/"predictions_h5.csv",parse_dates=["date"])
    events=pd.read_csv(RESULTS/"drift_events_h5.csv",parse_dates=["date"])
    metrics=pd.read_csv(RESULTS/"metrics_h5.csv")
    runtime=pd.read_csv(RESULTS/"runtime_h5.csv")
    drift_atlas();prequential(pred,events);recovery(pred,events);replay_and_events(events)
    cross_market(metrics);pareto(metrics,runtime)
    wealth(pd.read_csv(RESULTS/"economic_curves_h5.csv",parse_dates=["date"]))
    sensitivity(pd.read_csv(RESULTS/"metrics_sensitivity_h5.csv"))
    bootstrap_forest(pd.read_csv(RESULTS/"paired_block_bootstrap_h5.csv"))
    ablation_horizon(pd.read_csv(RESULTS/"metrics_ablation_h5.csv"),pd.read_csv(RESULTS/"metrics_h1.csv"),metrics)
    boundary_audit_figure();robustness_figure();protocol_figure();mechanism_figure();block_figure()


if __name__=="__main__":main()
