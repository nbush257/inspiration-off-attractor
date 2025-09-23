import pandas as pd
import sys
sys.path.append('../')
from utils import GENOTYPE_COLORS,GENOTYPE_LABELS,set_style,PHASE_MAP,sig2star
from pathlib import Path
import seaborn.objects as so
import numpy as np
from analyze_rec_no_stim_fit import map_stim_to_genotype
import pingouin as pg
import scipy.stats

map_stim_to_color = {k:GENOTYPE_COLORS[v] for k,v in map_stim_to_genotype.items()}
set_style()
def get_phasic_data():
    flist = list(Path('./').rglob('*_nostim/phasic_stims.csv'))
    df = pd.read_csv(flist[0])
    for fn in flist[1:]:
        df = pd.concat([df,pd.read_csv(fn)])
    rr_change = 100*(df['rr'] - df['rr_pre'])/df['rr_pre']
    amp_change = 100*(df['amp'] - df['amp_pre'])/df['amp_pre']
    df['delta_rr'] = rr_change
    df['delta_amp'] = amp_change
    return df

def get_amplitude_data():
    # Amplitude sweeps
    flist = list(Path('./').rglob('*_nostim/hold_stims.csv'))
    df = pd.read_csv(flist[0])
    for fn in flist[1:]:
        df = pd.concat([df,pd.read_csv(fn)])
    return(df)

def get_reset_curve_data():
    # Reset curves
    flist = list(Path('./').rglob('*_nostim/reset_curves.csv'))
    df = pd.read_csv(flist[0])
    for fn in flist[1:]:
        df = pd.concat([df,pd.read_csv(fn)])
    return(df)

def get_rebound_data():
    # Rebound
    flist = list(Path('./').rglob('*_nostim/rebound_latency.csv'))
    df = pd.read_csv(flist[0])
    for fn in flist[1:]:
        df = pd.concat([df,pd.read_csv(fn)])
    return(df)

def plot_summary_amplitude_sweeps(df,fs=(2,2)):

    df = df.copy().reset_index(drop=True).query('stim_amp >= 0')
    xmax = df['stim_amp'].max()
    p = (
        so.Plot(df,x='stim_amp',y='resp_rate',color='stim')
        .add(so.Line(),so.Agg('mean'),legend=False)
        .add(so.Band(),so.Est('mean','se'),legend=False)
        # .add(so.Line(alpha=0.2),group='eid',legend=False)
        .limit(x=(0,xmax),y=(0,None))
        .label(x='Stimulus amplitude (a.u.)',y='Resp. rate (Hz)')
        .layout(size=fs,engine='constrained')
        .scale(color=map_stim_to_color)
    ).plot()
    print(f'Number of recordings for amplitude sweep: {df["eid"].nunique()}')

    return(p)

def plot_summary_phasic_one_stim_amp(df,fs=(3,2)):
    # Phasic stimulus
    df_use = df.copy().query('((direction=="both" and phase=="exp") or (direction =="rising" and phase=="insp")) and stim_amp==4').reset_index(drop=True)
    # Set outliers where inspiratory stim drastically abolishes RR or any stim increses rr massively (bad mapping of stim to response)
    df_use = df_use.query('delta_rr<400') # Remove the massive outliers from viz
    df_use['outlier'] = df_use.eval('(phase=="insp" and delta_rr<-85)') # Keep insp outliers in viz
    p_rr = (
        so.Plot(df_use,x='phase',y='delta_rr',color='phase')
        .facet('stim')
        .add(so.Bar(width=0.25),so.Agg('median'),legend=False)
        .add(so.Range(),so.Est('median','ci'),legend=False)
        .add(so.Dot(pointsize=3,edgecolor='k'),so.Shift(0.25),so.Jitter(),group='eid',legend=False,marker='outlier')
        .label(x='',y=r'$\Delta$ Resp. rate (%)')
        .scale(color=PHASE_MAP,marker=['o','x'])
        .limit(y=(-100,None))
        .layout(size=fs,engine='constrained')
    ).plot()
    axs = p_rr._figure.axes
    for ax in axs:
        ss = ax.get_title()
        new_title = GENOTYPE_LABELS[map_stim_to_genotype[ss]] + '-like'
        ax.set_title(new_title,color=map_stim_to_color[ss],fontsize='x-small')
        ax.set_xticks(ax.get_xticks())
        xticklabels = ax.get_xticklabels()
        ax.set_xticklabels(x.get_text().capitalize() for x in xticklabels)
        ax.axhline(0,linestyle='--',color='k')
    p_rr
    p_amp = (
        so.Plot(df_use,x='phase',y='delta_amp',color='phase')
        .facet('stim')
        .add(so.Bar(width=0.25),so.Agg('median'),legend=False)
        .add(so.Range(),so.Est('median'),legend=False)
        .add(so.Dot(pointsize=3,edgecolor='k'),so.Shift(0.25),so.Jitter(),group='eid',legend=False,marker='outlier')
        .label(x='',y=r'$\Delta$ Dia. amp. (%)')
        .scale(color=PHASE_MAP,marker=['o','x'])
        .layout(size=fs,engine='constrained')
    ).plot()
    axs = p_amp._figure.axes
    for ax in axs:
        ss = ax.get_title()
        new_title = GENOTYPE_LABELS[map_stim_to_genotype[ss]] + '-like'
        ax.set_title(new_title,color=map_stim_to_color[ss],fontsize='x-small')
        ax.set_xticks(ax.get_xticks())
        xticklabels = ax.get_xticklabels()
        ax.set_xticklabels(x.get_text().capitalize() for x in xticklabels)
        ax.axhline(0,linestyle='--',color='k')
    # Stats
    stats =pd.DataFrame()
    for var in ['delta_rr','delta_amp']:
        for phase in ['insp','exp']:
            for stim in df_use['stim'].unique():
                stim_label = map_stim_to_genotype[stim]+'-like'
                _df = df_use.query(f'stim=="{stim}" and phase=="{phase}" and not outlier')
                
                statistic, p_value = scipy.stats.wilcoxon(_df[var].values)
                sig_star = sig2star(p_value)
                wilcoxon = pd.DataFrame({'statistic':statistic,
                                        'p-val':p_value,
                                        'sig_star':sig_star,
                                        'phase':phase,
                                        'var':var,
                                        'n':len(_df),
                                        'stim':stim_label},index=[0])
                stats = pd.concat([stats,wilcoxon])
            
    return(p_rr,p_amp,stats)

def plot_reset_curves(df,fs=(2.5,2)):

    df = df.copy().reset_index(drop=True)
    df = df.query('xstim<1.5 and ystim<5')

    # bin the xstim and ystim
    bins = np.arange(0,2.1,0.1)
    df['xstim'] = bins[np.digitize(df['xstim'], bins)-1]
    df['ystim'] = bins[np.digitize(df['ystim'], bins)-1]


    p = (
        so.Plot(df,x='xstim',y='ystim',color='condition')
        .add(so.Line(),so.Agg(),legend=False)
        .add(so.Band(),so.Est(),legend=False)
        .limit(x=(0,1.5),y=(0,2))
        .label(x='Stim phase (normalized)',y='Cycle duration (normalized)')
        .layout(size=fs)
        .scale(x=so.Continuous().tick(every=0.5),y=so.Continuous().tick(every=0.5),color=map_stim_to_color)
    ).plot()
    ax = p._figure.axes[0]
    ax.plot([0,2],[0,2],color='k',linestyle='--')
    ax.axhline(1,linestyle='--',color='k')
    ax.axvline(df['duty_cycle'].mean(),linestyle='-',color='C4')
    ax.axvline(df['duty_cycle_pk'].mean(),linestyle='--',color='C4')
    ax.text(df['duty_cycle'].mean(),2,'Dia. end',color='C4',rotation=90,fontsize='xx-small',ha='right',va='top')
    ax.text(df['duty_cycle_pk'].mean(),2,'Dia. peak',color='C4',rotation=90,fontsize='xx-small',ha='right',va='top')

    print(f'Number of recordings for reset curves: {df["eid"].nunique()}')
    return(p)

def plot_rebound_latency(df,fs=(2,1.5)):
    df_use = df.copy().reset_index(drop=True).query('amp==2')
    df_use = pd.pivot_table(df_use,values='latency',index=['eid','stim'],aggfunc='median').reset_index()
    df_use['latency'] *= 1000
    df_use['outlier'] = df_use['latency'].apply(lambda x: x > 1000)
    df_use.loc[df_use['outlier'],'latency'] = 1000
    p = (
        so.Plot(df_use,x='latency',y='stim',color='stim')
        .add(so.Bar(width=0.25),so.Agg('median'),legend=False)
        .add(so.Range(),so.Est('median'),legend=False)
        .add(so.Dot(color='w',edgecolor='k',pointsize=3,edgewidth=0.5),so.Shift(y=-0.25),so.Jitter(),group='eid',legend=False,marker='outlier')
        .scale(color=map_stim_to_color)
        .label(y='',x='Rebound latency (ms)',marker={True:'x',False:'o'})
        .layout(size=fs,engine='constrained')
    ).plot()
    ax = p._figure.axes[0]
    yticklabels = [GENOTYPE_LABELS[map_stim_to_genotype[k.get_text()]]+'-like' for k in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(yticklabels)
    
    normality = pg.normality(df_use, dv='latency',group='stim')

    # Stats
    df_use['stim'] = df_use['stim'].replace(map_stim_to_genotype)
    friedman = pg.friedman(df, dv='latency', within='stim',subject='eid')
    friedman['sig_star'] = friedman['p-unc'].apply(sig2star)

    holm = pg.pairwise_tests(df_use, dv='latency', between='stim', padjust='holm')
    holm['sig_star'] = holm['p-corr'].apply(sig2star)

    return(p,friedman,holm)

def plot_rising_vs_falling_insp_stims(df,fs=(2,1.5)):
    df_use = df.copy().reset_index(drop=True)
    df_use = df_use.query('stim_amp==4 and phase=="insp" and stim=="uniform_insp"')
    cc = {'Rising':"#976873",'Falling':"#68978C",'Both':"#0F0F0F"}
    df_use['direction'] = df_use['direction'].replace({'rising':'Rising','falling':'Falling','both':'Both'})
    df_use['outlier'] = df_use.eval('direction=="Rising" and delta_rr<-85')
    p = (
        so.Plot(df_use,x='direction',y='delta_rr',color='direction')
        .add(so.Bar(width=0.25),so.Agg('median'),legend=False)
        .add(so.Range(),so.Est('median'),legend=False)
        .add(so.Dot(pointsize=3,edgecolor='k'),so.Jitter(),so.Shift(0.25),group='eid',legend=False,marker='outlier')
        .scale(color=cc,
            x=so.Nominal(order=['Rising','Falling','Both']),
            marker = ['o','x']
        )
        .layout(size=fs,engine='constrained')
        .label(x='',y=r"$\Delta$ Resp. rate (% diff)")
    ).plot()


    stats = pd.DataFrame()
    for direction in df_use['direction'].unique():
        _df = df_use.query(f'direction=="{direction}" and not outlier')
        statistic, p_value = scipy.stats.wilcoxon(_df['delta_rr'].values)
        sig_star = sig2star(p_value)
        wilcoxon = pd.DataFrame({'statistic':statistic,
                                'p-val':p_value,
                                'sig_star':sig_star,
                                'direction':direction,
                                'n':len(_df)},index=[0])
        
        stats = pd.concat([stats,wilcoxon])  
        stats['stat'] = 'one sample wilcoxon'
    return(p,stats)


def main():
    # Amplitude sweeps
    df = get_amplitude_data()
    p1 = plot_summary_amplitude_sweeps(df)
    p1.save('amplitude_sweeps.pdf')

    # Phasic stimulus
    df = get_phasic_data()
    p2,p3,stats = plot_summary_phasic_one_stim_amp(df)
    p2.save('phasic_resp_rate.pdf')
    p3.save('phasic_amplitude.pdf')
    stats.to_csv('phasic_sim_stats.csv')
    
    # Rising vs Falling
    p6,stats = plot_rising_vs_falling_insp_stims(df)
    p6.save('rising_vs_falling_insp_stims.pdf')
    stats.to_csv('rising_vs_falling_insp_stims_stats.csv')

    # Reset curves
    df = get_reset_curve_data()
    p4 = plot_reset_curves(df)
    p4.save('reset_curves.pdf')

    # Rebound latency
    df= get_rebound_data()
    p5,friedman,holm = plot_rebound_latency(df)
    p5.save('rebound_latency.pdf')
    friedman.to_csv('anova_rebound_latency.csv')
    holm.to_csv('pairwise_rebound_latency.csv')

if __name__ == "__main__":
    main()