# Find if the expioratory fixed points lie inside the inspiratory or expiratory partition, and if they are stable,saddle, or unstable.
import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from utils import EIDS_REGEN,one
import pandas as pd
from analyze_single_rec import Rec



DF = pd.DataFrame()
for eid in EIDS_REGEN:
    rec = Rec(one, eid)
    rec.load_rslds(suffix='_norm')
    rslds = rec.rslds

    if rec.genotype=='vglut2ai32':
        continue
    # get exp k
    exp_k = 0 if rec.k_phase[0] == 'exp' else 1

    out = []
    stability_out = []
    spiral_out = []
    amps = [0,0.1,0.5,1,2,10]
    for v in amps:
        fp,stability = rec.compute_fixedpoints(v)
        A = rec.rslds.dynamics.As[exp_k]
        eigs = np.linalg.eigvals(A)
        xy = fp[exp_k].reshape(-1, 2)
        spiral = True if np.any(np.imag(eigs) != 0) else False
        z = np.argmax(xy.dot(rslds.transitions.Rs.T) + rslds.transitions.r, axis=1)
        out.append(rec.k_phase[z[0]])
        stability_out.append(stability[exp_k])
        spiral_out.append(spiral)

    df = pd.DataFrame({'v': amps, 'phase': out, 'stability': stability_out, 'spiral': spiral_out})
    df['eid'] = eid
    DF = pd.concat([DF, df], ignore_index=True)

df_0 = DF.query('v==0')

df_0.groupby('phase')[['stability','spiral']].value_counts()

df_all = DF.sort_values('v')
df_all['initial_stability'] = df_all.groupby('eid')['stability'].transform('first')
df_all['initial_phase'] = df_all.groupby('eid')['phase'].transform('first')

df_all = df_all.sort_values(['initial_phase','eid','v'])
import seaborn.objects as so
df_all['phase'] = df_all['phase'].map({'exp': 'Exp.', 'insp': 'Insp.'})

f,ax = plt.subplots(figsize=(4.5,2.5))

(
    so.Plot(df_all, x='v', y='phase',color='initial_phase')
    .facet(row='eid',wrap=4)
    .add(so.Dot(pointsize=5,edgewidth=0.5,edgecolor='k'),legend=False,marker='phase')
    .label(title='',y='',x='')
    .scale(color={'exp': "#d2d588", 'insp': "#8ffaf3"},marker={'Exp.': '^', 'Insp.': 'o'},x=so.Nominal())
    .limit()
    .on(f)
).plot()
ax.set_visible(False)
f.supylabel('Fixed point partition',x=0,fontsize='medium')
f.supxlabel('Stim amp (a.u.)',y=-0.10,fontsize='medium')
f.savefig('exp_eigenvector_transitions.pdf')


rec = Rec(one,EIDS_REGEN[1])
rec.load_rslds(suffix='_norm')
rec.plot_example_stimmed_dynamics([0,1,1.1])
plt.savefig('example_stimmed_dynamics_saddle.pdf')

rec = Rec(one,EIDS_REGEN[2])
rec.load_rslds(suffix='_norm')
rec.plot_example_stimmed_dynamics([0,1,1.1])
plt.savefig('example_stimmed_dynamics_sink.pdf')
