import pandas as pd
import sys

sys.path.append("../")
from utils import (
    GENOTYPE_COLORS,
    GENOTYPE_LABELS,
    set_style,
    PHASE_MAP,
    one,
    EIDS_REGEN,
)
from pathlib import Path
import seaborn.objects as so
import numpy as np
from analyze_rec_no_stim_fit import map_stim_to_genotype
from analyze_single_rec import Rec
import pickle
import cibrrig.plot as cbp
from singlecell.plot_singlecell import plot_example_spiketrains
import matplotlib.pyplot as plt
LOAD_PATH = Path(r"/data/hps/assoc/private/medullary/projects/VLAD/VLAD/model")


def load_reset_curve_data():
    # Reset curves
    plist = list(LOAD_PATH.glob("*_norm"))
    flist = []
    for p in plist:
        flist.append(p.joinpath("reset_curve_sweep.pqt"))

    df = pd.DataFrame()
    for fn in flist:
        if fn.exists():
            df = pd.concat([df, pd.read_parquet(fn)])

    df.reset_index(drop=True, inplace=True)
    return df


def plot_reset_curves(df, fs=(2, 1.5)):
    bins = np.arange(0, 2, 0.1)
    df_use = df.copy().query("stim_amp==2 & stim_dur==0.05")
    df_use["x_stim"] = bins[np.digitize(df_use["x_stim"], bins=bins) - 1]
    df_use["y_stim"] = bins[np.digitize(df_use["y_stim"], bins=bins) - 1]
    p = (
        so.Plot(df_use, x="x_stim", y="y_stim", color="genotype")
        .add(so.Line(), so.Agg("mean"), legend=False)
        .add(so.Band(), so.Est("mean", "se"), legend=False)
        .scale(color=GENOTYPE_COLORS)
        .limit(x=(0, 1.5), y=(0, 2))
        .label(x="Stim phase\n(normalized)", y="Cycle duration\n(normalized)")
        .layout(size=fs)
        .scale(x=so.Continuous().tick(every=0.5), y=so.Continuous().tick(every=0.5))
    ).plot()
    ax = p._figure.axes[0]
    ax.plot([0, 2], [0, 2], color="k", linestyle="--")
    ax.axhline(1, linestyle="--", color="k")
    return p


def plot_example_fit():
    eid = EIDS_REGEN[1]
    rec = Rec(one, eid)
    rec_path = LOAD_PATH.joinpath("rslds_m2025-01_g1_norm")
    fn = LOAD_PATH.joinpath(
        LOAD_PATH.joinpath("rslds_m2025-01_g1_norm/rslds_K2_D2_binsize0.010.pkl")
    )
    rec.load_rslds(fn=str(fn))
    rec.fit_sim_dia()
    pp = rec.q.mean_continuous_states[0][:200,:]
    sim_dia = rec.predict_dia(pp)


    f = plt.figure(figsize=(3,1.5))
    gs = f.add_gridspec(1, 2, width_ratios=[1, 0.5])
    ax = f.add_subplot(gs[0])
    ax = plot_example_spiketrains(rec, pre_time=0.5,post_time=12,stimulus='50ms',save_tgl=False,ax=ax,n_per_phase=5)
    ax2 = f.add_subplot(gs[1])
    ax2.set_prop_cycle(color=["#bb521f", "#a59586"])
    ax2.plot(pp)
    ax2.set_xlim(0,200)
    ax2.axis('off')
    ax2.set_ylim(-20,30)
    ax3 = ax2.twinx()
    ax3.plot(sim_dia,color='k',ls='--')
    ax3.set_ylim(-20,4)
    ax3.axis('off')
    return f

plot_example_fit()
plt.savefig('fit_schematic.pdf')