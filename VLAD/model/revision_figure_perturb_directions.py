from analyze_single_rec import Rec,Stimulus,rotate_vector
import numpy as np
from joblib import Parallel, delayed
from utils import EIDS_REGEN,one,set_style
import pandas as pd
import seaborn.objects as so
from tqdm import tqdm
from pathlib import Path
import warnings

# suppress ALFWarning from one.util while running this script
warnings.filterwarnings('ignore', message='No default revision for dataset', category=UserWarning, module=r'one\.util')

set_style()

# angles = [0,45,90,135,180,225,270,315,360]
# labels = ['inline','inline+45','orthogonal','orthogonal+45','opposite','opposite+45','orthogonal-45','inline-45','inline']

angles = [0,90,180,270,360]
labels = ['inline','inward','opposite','outward','inline']


amps = [0.5,1]
fn = Path("revision_perturbation_directions.parquet")

clockwise_recs = [
    'm2024_31_g0',
    'm2024_32_g1',
    'm2024_34_g1',
    'm2024_37_g0',
    'm2024_40_g1',
    'm2024_40_g2',
    'm2024_59_g1',
    'm2025-01_g1',
    'm2025-02_g0'
]

def _angle_worker(eid, angle, angle_to_label):
    """Run one angle’s sweep. Rec is built inside the worker to avoid pickling issues."""
    rec = Rec(one, eid)
    rec.load_rslds(suffix="_norm")
    rec.fit_sim_dia("SVR")
    v_e = rec.get_slow_exp_direction()
    mode = "uniform"
    stim_vector = rotate_vector(np.real(v_e), angle)
    stim = Stimulus(mode, stim_vector=stim_vector)
    rez = rec.compute_reset_curve_sweep(
        amps=amps, applied=stim, stim_durs=[0.05], nreps=500
    )
    rez["angle"] = angle
    rez["label"] = angle_to_label[angle]
    return rez


def compute():
    DF = pd.DataFrame()
    for eid in EIDS_REGEN:
        # lightweight Rec for prefix/genotype only
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        rec_lookup = Rec(one, eid,load_spikes=False)
        warnings.filterwarnings("default", category=RuntimeWarning)


        # decide orientation and create mapping
        if rec_lookup.prefix[:11] in clockwise_recs:
            angles_to_use = np.array(angles)
            angle_to_label = dict(zip(angles, labels))
        else:
            angles_to_use = np.array(angles)[::-1]
            angle_to_label = dict(zip(angles_to_use, labels))

        # parallel evaluation of angles
        results = Parallel(n_jobs=-1)(
            delayed(_angle_worker)(eid, angle, angle_to_label)
            for angle in angles_to_use
        )

        df = pd.concat(results, ignore_index=True)
        df["eid"] = eid
        df["genotype"] = rec_lookup.genotype
        DF = pd.concat([DF, df], ignore_index=True)

    DF.to_parquet(fn)

if not fn.exists():
    compute()

DF = pd.read_parquet(fn)
DF = DF.query('y_stim<3')
# Bin x_stim
DF["x_stim_binned"] = pd.cut(DF["x_stim"], bins=np.arange(0, 2.0, 0.05), labels=np.arange(0.05, 2, 0.05)).astype(float)



by_res = DF.query('stim_amp==0.5')\
    .groupby(['eid','label','stim_amp','x_stim_binned'])['y_stim'].mean().reset_index()


# Map x-stim to radial (where0-1 maps from 0 to to 2pi)

# Plot on radial axis
p=(
    so.Plot(by_res,x='x_stim_binned',y='y_stim',color='label')
    .facet('label',wrap=3)
    .add(so.Line(linewidth=2),so.Agg())
    .add(so.Lines(alpha=0.5,linewidth=0.5),group='eid')
    .add(so.Band(alpha=0.5),so.Est())
    .scale(color=so.Continuous('husl').tick(every=90))
    .limit(x=(0,1.1),y=(0.5,1.5))
    .layout(size=(6,6))
    .label(x="Stimulus phase (Normalized)",y="Period (Normalized)",title="{} Degrees".format)
).plot()

for ax in p._figure.axes:
    ax.axhline(1, ls='--', color='k', lw=0.5)
p.save("perturbation_directions.pdf")
