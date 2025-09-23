""" 
Apply hyopthetical stims to recordings that were fit without stims
"""
from analyze_single_rec import Stimulus,get_stim_field,rotate_vector,one,EIDS_REGEN_NOSTIM,Rec,plot_summary_amp_sweep,sample_with_stim,burst_stats_dia
from matplotlib import pyplot as plt  
import numpy as np 
import sys
import pandas as pd
sys.path.append('../')
from utils import GENOTYPE_COLORS,GENOTYPE_LABELS
import click
from cibrrig.analysis.population import compute_projection_speed
import numpy as np
np.random.seed(0)
FIGSIZE = (1.5,1.5)

map_stim_to_genotype = {
    'sigmoidal_exp': 'vglut2ai32',
    'sink': 'vgatai32',
    'sigmoidal_insp': 'vgatai32',
    'uniform_insp': 'vgatcre_ntschrmine',
    'uniform_exp': 'vglut2ai32',

}
def make_stimfield_with_reset(rec,stims,stim_amplitude=4,nreps=100):
    
    duty_cycle = np.mean(rec.breaths.duration_sec/rec.breaths.IBI)
    duty_cycle_pk = np.mean((rec.breaths.pk_time-rec.breaths.times)/rec.breaths.IBI)

    df_reset = pd.DataFrame()

    f, axs = plt.subplots(
        figsize=(5, 3),
        ncols=len(stims),
        nrows=2,
        sharex='row',
        sharey='row',
        constrained_layout=True,
        gridspec_kw={'height_ratios': [4, 1]}
    )
    for ii,(k,stim) in enumerate(stims.items()):
        ax = axs[0][ii]
        X,Y,u,v = get_stim_field(rec, stim)
        zs,xs,ys = rec.rslds.sample(200, with_noise=False, input=None)
        ax.plot(*xs.T, color="k", alpha=0.75, lw=1)
        rec.make_streamplot(ax=ax,xlim=20, ylim=20,density=0.3)
        ax.quiver(X, Y, u, v, angles='xy', scale_units='xy', scale=0.05, width=0.01,alpha=0.8)
        ax.set_xlabel(r"$x_{1}$")
        ax.set_ylabel(r"$x_{2}$")
        title  = f"{GENOTYPE_LABELS[map_stim_to_genotype[k]]} - like"
        ax.set_title(title, fontsize=8,color =GENOTYPE_COLORS[map_stim_to_genotype[k]])
        try:
            xstim,ystim,xcontrol,ycontrol = rec.sim_reset_curve(
                applied=stim,
                nreps=nreps,
                stim_dur=0.05,
                ISI=3,
                stim_amplitude=stim_amplitude,
                plot_tgl=False,
            )
        except:
            # One of the recordings goes to infinity - likely due to the saddle
            xstim = np.full(nreps, np.nan)
            ystim = np.full(nreps, np.nan)
        ax = axs[1][ii]
        ax.plot(xstim,ystim,'o',color='C3', alpha=0.75,ms=1)
        ax.axhline(1, color='k', lw=0.5, ls='--')
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.set_xlim(0,1.5)
        ax.set_ylim(0,1.5)   
        ax.axline([0,0],slope=1, color='silver')
        ax.set_xlabel('Stimulus time (normalized)')

        ax.axvline(duty_cycle, color="C4", lw=0.5, ls='-')
        ax.axvline(duty_cycle_pk, color="C4", lw=0.5, ls='--')

        _df = pd.DataFrame()
        _df['xstim'] = xstim
        _df['ystim'] = ystim
        _df['condition'] = k

        df_reset = pd.concat([df_reset, _df])
    df_reset['duty_cycle'] = duty_cycle
    df_reset['duty_cycle_pk'] = duty_cycle_pk
    df_reset['eid'] = rec.eid
    df_reset['genotype'] = rec.genotype

    axs[1][0].set_ylabel("Cycle duration\n(normalized)")
    return(f,df_reset)
    
def get_rebound_latency(rec,stims,amps,nreps=10):
    binsize = rec.binsize

    pre_time = 5
    post_time = 5
    pad_time = 2
    dur = 20 + (pad_time * 2)
    max_dur = pre_time + dur + post_time

    pre_bins = int(pre_time / binsize)
    post_bins = int(post_time / binsize)
    max_bins = int(max_dur / binsize)
    dur_bins = int(dur / binsize)
    v = np.concatenate([np.zeros(pre_bins), np.ones(dur_bins), np.zeros(post_bins)])
    v = v[:, np.newaxis]

    # Loop over stimulus types
    df = pd.DataFrame()
    for k,stim in stims.items():
        for amp in amps:
            _df = pd.DataFrame()
            latencies=[]
            for ii in range(nreps):
                a, b, c = sample_with_stim(rec, max_bins, stim, input=v * amp)
                _predicted_dia = rec.predict_dia(b)
                offset_time = pre_time+dur
                breaths = burst_stats_dia(_predicted_dia,1/rec.binsize,dia_thresh=1.5)
                breaths = breaths.query('on_sec>@offset_time')
                try:
                    latency = breaths.iloc[0].on_sec - offset_time
                    latencies.append(latency)
                except:
                    latencies.append(np.nan)
            _df['latency'] = latencies
            _df['stim'] = k
            _df['amp'] = amp
            df = pd.concat([df, _df])
    df['eid'] = rec.eid
    df['genotype'] = rec.genotype    
    return(df)

def compute_phasic(rec,stim,stim_amps):
    pre_time = 10
    df = pd.DataFrame()
    for phase in ['insp','exp']:
        for direction in ['both','rising','falling']:
            for stim_amp in stim_amps:
                rez = rec.sim_phasic_stims(mode=phase,direction=direction,applied=stim,pre_time=pre_time,stim_amplitude=stim_amp)
                dia_predicted  = rez['dia_predicted']
                tmax = len(dia_predicted)*rec.binsize
                tstim = tmax-pre_time
                breaths = burst_stats_dia(rez['dia_predicted'],1/rec.binsize,dia_thresh=1.5)
                nbreaths_pre = len(breaths.query('on_sec<=@pre_time'))

                if nbreaths_pre > 0:
                    rr_pre = nbreaths_pre/pre_time
                    amp_pre = breaths.query('on_sec<=@pre_time')['amp'].mean()
                else:
                    rr_pre = 0
                    amp_pre = 0
                
                breaths = breaths.query('on_sec>@pre_time')
                nbreaths = len(breaths)
                rr = nbreaths/tstim
                if nbreaths ==0:
                    amp = 0
                else:
                    amp = breaths['amp'].mean()
                _df = pd.DataFrame()
                _df['rr'] = [rr]
                _df['amp'] = [amp]
                _df['stim_amp'] = [stim_amp]
                _df['amp_pre'] = [amp_pre]
                _df['phase'] = [phase]
                _df['direction'] = [direction]
                _df['rr_pre'] = [rr_pre]
                df = pd.concat([df, _df])
        df['eid'] = rec.eid
    df['genotype'] = rec.genotype
    return(df)
                   

def run_recording(ii):
    rec = Rec(one, EIDS_REGEN_NOSTIM[ii])
    rec.load_rslds(suffix='_nostim')
    rec.fit_sim_dia('SVR')
    scale_stim = 1/np.mean(compute_projection_speed(rec.q.mean_continuous_states[0]))

    stims = {}
    stims['sigmoidal_exp'] = Stimulus('sigmoid', stim_vector=rec.get_slow_exp_direction()*scale_stim, k=rec.k_phase.index("exp"),slope=0.1)
    stims['sigmoidal_insp'] = Stimulus('sigmoid', stim_vector=-rec.get_slow_exp_direction()*scale_stim, k=rec.k_phase.index("insp"),slope=0.1)
    stims['uniform_insp'] = Stimulus('uniform', stim_vector=-rec.get_slow_exp_direction()*scale_stim)
    

    # COMPUTE REBOUND LATENCY
    rebound_latency = get_rebound_latency(rec,stims,amps=[1,2,3,4],nreps=10)
    rebound_latency.to_csv(rec.model_path.joinpath(f"rebound_latency.csv"),index=False)

    # COMPUTE RESETCUVES
    f,reset_curves = make_stimfield_with_reset(rec,stims)
    plt.savefig(rec.model_path.joinpath(f"stim_field_and_reset.pdf"))
    reset_curves.to_csv(rec.model_path.joinpath(f"reset_curves.csv"),index=False)

    # Loop through stims for other plots
    all_amp_sweeps = pd.DataFrame()
    all_phasic = pd.DataFrame()
    for ii,(k,stim) in enumerate(stims.items()):    

        # Amplitude sweep example
        amps = np.round(np.arange(0,4,0.5),1)
        rec.plot_amp_sweep(stim,amps = amps)
        plt.savefig(rec.model_path.joinpath(f"amp_sweep_{k}.pdf"))

        # Example phasic
        f = rec.plot_example_phasic(applied=stim,amp=4,dia_thresh=1)
        f.savefig(rec.model_path.joinpath(f"example_phasic_{k}_both.pdf"))
        plt.close('all')

        p = rec.plot_example_phasic(applied=stim,amp=4,direction='rising')
        p.savefig(rec.model_path.joinpath(f"example_phasic_{k}_both.pdf"))
        plt.close('all')
        # All phasic sweep
        amps = np.arange(0,5.1,1)
        phasic_rez = compute_phasic(rec,stim,amps)
        phasic_rez['stim'] = k
        all_phasic = pd.concat([all_phasic,phasic_rez])

        # Amplitude hold sweep
        amps = np.arange(0,5.1,0.1)
        amp_sweep_rez = rec.compute_amp_sweep(applied=stim,amps=amps,dia_thresh=1.5,mean_sub=True)
        p = plot_summary_amp_sweep(amp_sweep_rez,'a.u.',fs=FIGSIZE)
        p.save(rec.model_path.joinpath(f"amp_sweep_summary_{k}.pdf"))
        amp_sweep_rez['stim'] = k
        all_amp_sweeps = pd.concat([all_amp_sweeps,amp_sweep_rez])

    # Save outpus
    all_amp_sweeps.to_csv(rec.model_path.joinpath(f"hold_stims.csv"),index=False)
    all_phasic.to_csv(rec.model_path.joinpath(f"phasic_stims.csv"),index=False)
    plt.close('all')

@click.command()
@click.argument("ii", type=int)
def main(ii):
    run_recording(ii)

if __name__ == '__main__':
    main()