import sys

sys.path.append("../")
sys.path.append("VLAD/")
from utils import CACHE_DIR, HB_MAP
from utils import Rec
from one.api import One
import logging
import matplotlib.pyplot as plt
from cibrrig.analysis.population import Population
import cibrrig.videos as cbv
from pathlib import Path
import numpy as np

plt.rcParams["animation.ffmpeg_path"] = (
    "/data/hps/assoc/private/medullary/miniforge-pypy3/envs/iblenv/bin/ffmpeg"
)

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

def make_video(pop,rec, stim_type,cmap='bone',**kwargs):
    '''
    Wrapper to cibrrig videos
    '''
    stim_color=rec.laser_color
    if stim_type == "hold":
        intervals= rec.get_pulse_stims(2)[0]
    elif stim_type == "insp":
        intervals = rec.get_phasic_stims("insp")[0]
    elif stim_type == "exp":
        intervals = rec.get_phasic_stims("exp")[0]
    elif stim_type == "50":
        intervals = rec.get_pulse_stims(0.05)[0]
    elif stim_type == "10":
        intervals = rec.get_pulse_stims(0.01)[0]
    elif stim_type == "HB5":
        intervals = rec.get_HB_stims(5)[0]
        stim_color = HB_MAP[5]
    elif stim_type == "HB2":
        intervals = rec.get_HB_stims(2)[0]
        stim_color = HB_MAP[2]
    else:
        raise ValueError("No valid stim types provided")
    fn_out = f"{rec.subject}_g{rec.sequence}_{rec.genotype}_{stim_type}_pca.mp4"

    cbv.make_aux_raster_projection_with_stims(
        pop,
        intervals=intervals,
        aux=rec.diaphragm.filtered,
        aux_t=rec.diaphragm.times,
        fn_out=fn_out,
        stim_color=stim_color,
        aux_label="dia",
        cmap=cmap,
        baseline=5,
        **kwargs
    )


one = One(cache_dir=CACHE_DIR)
style = '../VLAD.mplstyle'
style = 'dark_background'
for subject in ['m2024-40',"m2024-34",'m2024-30']:
    K=2
    D=2
    binsize=0.01
    
    eid = one.search(subject=subject, datasets="spikes.times.npy")[0]
    rec = Rec(one, eid,load_raw_dia=True)

    pop = Population(rec.spikes.times, rec.spikes.clusters,t0=300,tf=600)
    pop.compute_projection()

    make_video(pop,rec, "hold", duration=6,style=style)
    make_video(pop,rec, "insp", duration=8,style=style)
    make_video(pop,rec, "exp", duration=4,style=style)
    make_video(pop,rec, "HB5", duration=8,style=style)

# ------------------------ #
# Make just a simple trajectory video
# ------------------------ #
subject = 'm2024-40'
eid = one.search(subject=subject, datasets="spikes.times.npy")[0]
rec = Rec(one, eid,load_raw_dia=True)
pop = Population(rec.spikes.times, rec.spikes.clusters,t0=300,tf=600)
pop.compute_projection()
phi2 = pop.sync_var(rec.phi,rec.phi_t)
pop.plot_projection_line(dims=[0,1],t0=300,tf=310,cvar=phi2,cmap='RdBu_r',vmin=-np.pi,vmax=np.pi)
cbv.make_projection(
    pop,
    dims=[0,1],
    fn_out=f"{rec.subject}_g{rec.sequence}_{rec.genotype}_simpletrajectory.mp4",
    duration=10,
    t0=300,
    cvar=phi2,
    vmin=-np.pi,
    vmax=np.pi,
    rotation_delay=0,
    mode='line',
    history_kwargs=dict(alpha=0),
    )

cbv.make_aux_raster_projection_with_stims(
    pop,
    intervals=np.array([[300,300]]),
    aux=rec.diaphragm.filtered,
    aux_t=rec.diaphragm.times,
    stim_color='C1',
    aux_label="dia",
    cmap='bone',
    dims=[0,1,2],
    fn_out=f"{rec.subject}_g{rec.sequence}_{rec.genotype}_baseline.mp4",
    duration=21,
    lead_in=20,
    rotation_delay=0,
    elev_speed=0.1,
    azim_speed=0.05,
    projection_kwargs=dict(alpha=0.5,linewidth=0.5,cvar=phi2,cmap='RdBu_r',vmin=-np.pi,vmax=np.pi),
    history_kwargs=dict(alpha=0),
    fps=60
    )