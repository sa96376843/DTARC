"""Create every LaTeX table directly from frozen experiment outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';T=ROOT/'manuscript'/'tables';T.mkdir(parents=True,exist_ok=True)
ORDER=['static','logistic','periodic','periodicER','adwinER','phER','adwinDTARC','phDTARC']

def write(name,content):(T/name).write_text(content);print('table',name)
def table_env(headers,rows,align=None,caption='',label='',resize=False):
    align=align or 'l'+'r'*(len(headers)-1);body=' & '.join(headers)+r' \\'+'\n\\midrule\n'
    body+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
    tab=f'\\begin{{tabular}}{{{align}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}'
    if resize:tab='\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}%\n'+tab+'\n\\endgroup'
    return f'\\begin{{table}}[htbp]\n\\centering\\small\n\\caption{{{caption}\\label{{{label}}}}}\n{tab}\n\\end{{table}}\n'
def sci(x):
    if x in [0,1]:return str(x)
    e=int(np.floor(np.log10(abs(x))));a=x/10**e
    return f'${a:g}\\times10^{{{e}}}$'
def metric_table(stem,name,caption,label):
    d=pd.read_csv(R/f'metrics_{stem}.csv');rows=[]
    for method in ORDER:
        if method not in d.method.unique():continue
        row=[method]
        for m in ['CN','TH']:
            q=d[(d.market==m)&(d.method==method)]
            for metric in ['mcc','auc','brier']:row.append(f'{q[metric].mean():.4f} $\\pm$ {q[metric].std():.4f}')
        rows.append(row)
    write(name,table_env(['Method','CN MCC','CN AUC','CN Brier','TH MCC','TH AUC','TH Brier'],rows,'lrrrrrr',caption,label,True))
def simple_metric_rows(d,methods,markets=['CN','TH']):
    rows=[]
    for method in methods:
        row=[method.replace('DTARC_','').replace('_',r'\_')]
        for m in markets:
            q=d[(d.market==m)&(d.method==method)]
            row += [f'{q.mcc.mean():.4f}',f'{q.auc.mean():.4f}',f'{q.brier.mean():.4f}']
        rows.append(row)
    return rows

def main():
    info=json.loads((ROOT/'data'/'dataset_summary.json').read_text());pred=pd.read_csv(R/'predictions_h5.csv')
    vals={}
    for m in ['CN','TH']:
        s=info['markets'][m];n=len(pred[(pred.market==m)&(pred.method=='static')&(pred.seed==11)])
        vals[m]={'Price source':'Sina Finance, single-source forward adjustment' if m=='CN' else 'SiamChart, as published',
            'Stocks':s['stocks'],'Common modeling window':f"{s['start']}--{s['end']}",'Initial training rows':s['splits']['train'],
            'Validation rows':s['splits']['validation'],'Test rows':s['splits']['test'],'Scored $h=5$ rows per seed':n,'Observed $h=5$ up rate':f"{s['up_rate_h5']:.3f}"}
    rows=[[k,vals['CN'][k],vals['TH'][k]] for k in vals['CN']]
    write('tab01_data.tex',table_env(['Characteristic','China','Thailand'],rows,'lll','Data partitions and observed samples. The Chinese analysis uses one vendor and one adjustment convention throughout training, validation, and testing.','tab:data',True))
    cfg=json.loads((R/'best_config.json').read_text())['best']
    rows=[['Initial MLP','25--32--16--1; ReLU; dropout 0.10'],['Initial optimization','120 epochs; Adam $10^{-3}$; weight decay $10^{-4}$'],
        ['Online optimization',f"2 epochs; Adam {sci(cfg['online_lr'])}; weight decay $10^{{-5}}$"],['ADWIN $\\delta$',cfg['adwin_delta']],
        ['Page--Hinkley threshold',cfg['ph_threshold']],['Memory capacity',cfg['buffer']],['Replay batch',cfg['replay']],
        ['Proximal coefficient',cfg['prox']],['Abrupt threshold',cfg['abrupt_threshold']],['Seeds','11, 22, 33, 44, 55']]
    write('tab02_protocol.tex',table_env(['Component','Frozen value'],rows,'ll','Experimental protocol. Selection used only Chinese 2025 validation data; 2026 runs restarted from the 2023--2024 initial model.','tab:protocol'))
    metric_table('h5','tab03_main_h5.tex','Five-day direction results on the untouched 2026 test period (mean $\\pm$ SD over five seeds). Brier is a loss; MCC and AUC are gains.','tab:main')
    metric_table('h1','tab04_h1.tex','One-day direction robustness on the untouched 2026 test period (mean $\\pm$ SD over five seeds).','tab:h1')
    ab=pd.read_csv(R/'metrics_ablation_h5.csv');rows=simple_metric_rows(ab,['adwinDTARC','DTARC_no_anchor','DTARC_no_diversity','DTARC_no_typing','DTARC_no_proximal'])
    write('tab05_ablation.tex',table_env(['Variant','CN MCC','CN AUC','CN Brier','TH MCC','TH AUC','TH Brier'],rows,'lrrrrrr','DTARC component ablation. Removed quotas are redistributed without changing the replay batch size.','tab:ablation',True))
    rt=pd.read_csv(R/'runtime_h5.csv');ev=pd.read_csv(R/'drift_events_h5.csv');rows=[]
    for m in ['CN','TH']:
        for method in ['periodic','periodicER','adwinER','adwinDTARC','phER','phDTARC']:
            q=rt[(rt.market==m)&(rt.method==method)];e=ev[(ev.market==m)&(ev.method==method)]
            rows.append([m,method,f'{q.updates.mean():.2f}',f'{q.seconds.mean():.3f}',f'{len(e[e.type=="abrupt"])/5:.2f}',f'{len(e[e.type=="gradual"])/5:.2f}',int(q.final_memory.mean())])
    write('tab06_runtime.tex',table_env(['Market','Method','Updates','Seconds','Abrupt/seed','Gradual/seed','Memory'],rows,'llrrrrr','Observed update frequency, online-loop runtime, drift composition, and final memory. Initial fitting is excluded.','tab:runtime'))
    em=pd.read_csv(R/'economic_metrics_h5.csv');bh=pd.read_csv(R/'economic_buy_hold_h5.csv');rows=[]
    for m in ['CN','TH']:
        for method in ORDER+['equal_weight']:
            style='benchmark' if method=='equal_weight' else 'long_short'
            q=em[(em.market==m)&(em.method==method)&(em.cost_bps==10)&(em['style']==style)]
            rows.append([m,method.replace('_',r'\_'),f'{100*q.annual_return.mean():.2f}\\%',f'{100*q.annual_excess_return.mean():.2f}\\%',f'{q.sharpe.mean():.3f}',f'{q.market_beta.mean():.3f}',f'{100*q.max_drawdown.mean():.2f}\\%'])
        q=bh[(bh.market==m)&(bh.cost_bps==10)]
        rows.append([m,r'buy\_hold',f'{100*q.annual_return.mean():.2f}\\%','--',f'{q.sharpe.mean():.3f}','1.000',f'{100*q.max_drawdown.mean():.2f}\\%'])
    write('tab07_economic.tex',table_env(['Market','Method','Ann. return','Excess','Sharpe','Beta','Max DD'],rows,'llrrrrr','Next-close long--short results at 10 bps. Excess return is relative to the same-cost equal-weight universe benchmark; buy-and-hold uses fixed initial equal weights.','tab:economic',True))
    rows=[]
    for m in ['CN','TH']:
        for method in ORDER+['equal_weight']:
            style='benchmark' if method=='equal_weight' else 'long_only';q=em[(em.market==m)&(em.method==method)&(em.cost_bps==10)&(em['style']==style)]
            rows.append([m,method.replace('_',r'\_'),f'{100*q.annual_return.mean():.2f}\\%',f'{100*q.annual_excess_return.mean():.2f}\\%',f'{q.sharpe.mean():.3f}',f'{q.market_beta.mean():.3f}',f'{100*q.max_drawdown.mean():.2f}\\%'])
    write('tab11_longonly.tex',table_env(['Market','Method','Ann. return','Excess','Sharpe','Beta','Max DD'],rows,'llrrrrr','Top-three long-only results after 10 bps, with same-cost equal-weight benchmark comparisons.','tab:longonly',True))
    boot=pd.read_csv(R/'paired_block_bootstrap_h5.csv');rows=[]
    for m in ['CN','TH']:
        for comp in boot.comparison.unique():
            for metric in ['mcc','auc','brier']:
                q=boot[(boot.market==m)&(boot.comparison==comp)&(boot.metric==metric)].iloc[0]
                holm='--' if pd.isna(q.holm_p) else f'{q.holm_p:.4f}'
                rows.append([m,comp.replace('adwinDTARC - ','DTARC $-$ '),metric.upper(),f'{q.mean_difference:.5f}',f"[{q['ci_2.5']:.5f}, {q['ci_97.5']:.5f}]",f'{q.bootstrap_p:.4f}',holm])
    write('tab08_bootstrap.tex',table_env(['Market','Comparison','Metric','Difference','95\\% CI','$p$','Holm $p$'],rows,'lllrrrr','Paired date-level moving-block bootstrap (five-day blocks, 500 replicates). Holm correction covers the 24 prespecified primary tests; Page--Hinkley comparisons are exploratory.','tab:bootstrap',True))
    se=pd.read_csv(R/'metrics_sensitivity_h5.csv');rows=[]
    for factor in ['buffer','replay','online_lr','ph_threshold']:
        for value in sorted(se[se.factor==factor].value.unique()):
            row=[factor.replace('_',r'\_'),f'{value:g}']
            for m in ['CN','TH']:
                q=se[(se.market==m)&(se.factor==factor)&(se.value==value)];row += [f'{q.mcc.mean():.4f}',f'{q.auc.mean():.4f}']
            rows.append(row)
    write('tab09_sensitivity.tex',table_env(['Factor','Value','CN MCC','CN AUC','TH MCC','TH AUC'],rows,'llrrrr','Frozen-protocol sensitivity diagnostics; none was used to revise the selected configuration.','tab:sensitivity'))
    full=pd.read_csv(R/'metrics_h5.csv');rows=[]
    for m in ['CN','TH']:
        for method in ORDER:
            q=full[(full.market==m)&(full.method==method)];rows.append([m,method]+[f'{q[x].mean():.4f}' for x in ['accuracy','macro_f1','mcc','auc','brier']])
    write('tab10_full_metrics.tex',table_env(['Market','Method','Accuracy','Macro-F1','MCC','AUC','Brier'],rows,'llrrrrr','Complete five-day classification metrics, averaged over five random seeds.','tab:fullmetrics'))
    matched=pd.read_csv(R/'metrics_matched_replay.csv');combined=pd.concat([full,matched],ignore_index=True)
    rows=simple_metric_rows(combined,['adwinDTARC','adwinER','adwinERstrat','DTARC_no_provenance'])
    write('tab12_matched_replay.tex',table_env(['Method','CN MCC','CN AUC','CN Brier','TH MCC','TH AUC','TH Brier'],rows,'lrrrrrr','Matched replay controls. Stratified uniform replay isolates class balancing; no-provenance DTARC retains ADWIN and the same update budget but assigns all replay to uniform sampling.','tab:matched',True))
    ex=pd.read_csv(R/'metrics_execution_aligned.csv');rows=simple_metric_rows(ex,['static','periodicER','adwinER','adwinDTARC'])
    write('tab13_execution_aligned.tex',table_env(['Method','CN MCC','CN AUC','CN Brier','TH MCC','TH AUC','TH Brier'],rows,'lrrrrrr','Execution-aligned targets $t+1\\rightarrow t+6$, one extra training purge row, and two-batch online delay.','tab:exec',True))
    frames=[]
    for panel,file in [('Forward adjusted','metrics_sina_qianfuquan.csv'),('Backward adjusted','metrics_sina_houfuquan.csv'),('Leave-one-out context','metrics_loo.csv')]:
        d=pd.read_csv(R/file);d['panel']=panel;frames.append(d)
    rob=pd.concat(frames);rows=[]
    for panel in rob.panel.unique():
        for method in ['static','adwinER','adwinDTARC']:
            q=rob[(rob.panel==panel)&(rob.market=='CN')&(rob.method==method)]
            if q.empty:continue
            rows.append([panel,method,f'{q.mcc.mean():.4f}',f'{q.auc.mean():.4f}',f'{q.brier.mean():.4f}'])
        if panel=='Leave-one-out context':
            for method in ['static','adwinER','adwinDTARC']:
                q=rob[(rob.panel==panel)&(rob.market=='TH')&(rob.method==method)]
                rows.append(['TH: '+panel,method,f'{q.mcc.mean():.4f}',f'{q.auc.mean():.4f}',f'{q.brier.mean():.4f}'])
    write('tab14_data_robustness.tex',table_env(['Panel','Method','MCC','AUC','Brier'],rows,'llrrr','Single-source adjustment and leave-one-out market-context audits. Unless prefixed by TH, rows refer to China.','tab:datarob'))
    pr=pd.read_csv(R/'metrics_protocol_degradation.csv');rows=[]
    for _,q in pr.iterrows():rows.append([q['method'].replace('_',r'\_'),f'{q.accuracy:.4f}',f'{q.mcc:.4f}',f'{q.auc:.4f}',f'{q.brier:.4f}'])
    write('tab15_protocol_degradation.tex',table_env(['Protocol','Accuracy','MCC','AUC','Brier'],rows,'lrrrr','Deliberate protocol-degradation diagnostics on Chinese validation data (seed 42). Random row splitting is invalid for deployment and is shown only to quantify optimistic bias.','tab:degradation'))
    rows=[]
    for panel in ['stitched','single_source']:
        s=pd.read_csv(R/f'boundary_summary_{panel}.csv').iloc[0]
        rows.append([panel.replace('_',' '),f"{100*s.median_abs_boundary_return:.2f}\\%",f"{100*s.max_abs_boundary_return:.2f}\\%",f'{s.median_return_psi:.3f}',f'{s.max_return_psi:.3f}',f'{s.median_volume_log_ratio:.3f}'])
    write('tab16_boundary_audit.tex',table_env(['Panel','Median $|$boundary $r|$','Max $|$boundary $r|$','Median PSI','Max PSI','Median volume log ratio'],rows,'lrrrrr','Twenty-observation pre/post-boundary audit. The single-source panel removes adjustment-scale jumps but does not eliminate genuine distributional change.','tab:boundary',True))
    bs=pd.read_csv(R/'paired_block_bootstrap_sensitivity_h5.csv');rows=[]
    for m in ['CN','TH']:
        for comp in ['adwinER','static','periodicER']:
            for metric in ['mcc','auc']:
                q=bs[(bs.market==m)&(bs.comparator==comp)&(bs.metric==metric)]
                rows.append([m,comp,metric.upper()]+[f"[{z['ci_2.5']:.4f}, {z['ci_97.5']:.4f}]" for _,z in q.sort_values('block_days').iterrows()])
    write('tab17_block_sensitivity.tex',table_env(['Market','Comparator','Metric','Block 5 CI','Block 10 CI','Block 20 CI'],rows,'lllrrr','Moving-block length sensitivity for the prespecified comparisons (500 paired replicates per block length).','tab:block',True))
    mech=pd.read_csv(R/'metrics_mechanism_h5.csv');rtm=pd.read_csv(R/'runtime_mechanism_h5.csv');rows=[]
    for factor in ['adwin_delta','abrupt_threshold','prox']:
        for value in sorted(mech[mech.factor==factor].value.unique()):
            row=[factor.replace('_',r'\_'),f'{value:g}']
            for m in ['CN','TH']:
                q=mech[(mech.market==m)&(mech.factor==factor)&(mech.value==value)];u=rtm[(rtm.market==m)&(rtm.factor==factor)&(rtm.value==value)]
                row += [f'{q.mcc.mean():.4f}',f'{q.auc.mean():.4f}',f'{u.updates.mean():.1f}']
            rows.append(row)
    write('tab18_mechanism.tex',table_env(['Factor','Value','CN MCC','CN AUC','CN updates','TH MCC','TH AUC','TH updates'],rows,'llrrrrrr','Detector, typing-threshold, and proximal-scale diagnostics under the frozen protocol.','tab:mechanism',True))
if __name__=='__main__':main()
