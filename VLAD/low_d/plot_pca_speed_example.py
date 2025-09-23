""" 
Plot example trajectory colored by speed (specifically for SfN talk)
"""

import sys

sys.path.append("../")
sys.path.append("VLAD/")
from utils import SUBJECTS, GENOTYPES, PHASE_MAP, CACHE_DIR, HB_MAP, GENOTYPE_COLORS,EIDS_NEURAL,one,GENOTYPE_LABELS,Rec
from cibrrig.analysis.population import Population
import matplotlib.pyplot as plt
plt.style.use("../VLAD.mplstyle")

def plot_speed_example(eid):
    rec = Rec(one, eid)    
    pop = Population(rec.spikes.times, rec.spikes.clusters, t0=300, tf=600)
    pop.compute_projection()
    pop.compute_projection_speed()
    WAVELENGTH=473
    if 'chrmine' in rec.genotype:
        WAVELENGTH=635
    dims=[0,1]
    post_time=0.025

    f = plt.figure()
    ax = f.add_subplot()
    pop.plot_projection_line(dims=dims,t0=500,tf=600,cvar=pop.projection_speed,colorbar_title='Trajectory speed (a.u.)',alpha=0.5,ax=ax,vmin=0.,vmax=0.6)
    plt.savefig(f'example_speed_{rec.genotype}.png')


eid = one.search(subject='m2024-30',datasets='spikes.times.npy')[0]
plot_speed_example(eid)