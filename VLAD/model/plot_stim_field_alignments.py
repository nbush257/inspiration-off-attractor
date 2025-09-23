import  pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn.objects as so
import pandas as pd
import sys
sys.path.append('../')
from utils import GENOTYPE_COLORS,GENOTYPE_LABELS,set_style
GENOTYPE_ORDER = ['vglut2ai32','vgatai32','vgatcre_ntschrmine']
set_style()

class Dynamics:
    def __init__(self, d):
        self.As = d["As"]
        self.bs = d["bs"]
        self.Vs = d["Vs"]
        self.Rs = d["Rs"]
        self.r = d["r"]
        self.exp_fast = d["exp_fast"]
        self.exp_slow = d["exp_slow"]
        self.phase_k = d["phase_k"]
        self.subject = d["subject"]
        self.number = d["number"]
        self.eid = d["eid"]
        self.genotype = d["genotype"]
        cc = {'insp':'red', 'exp':'blue'}
        self.k_colors = [cc[k] for k in self.phase_k]
    
    def plot_stream(self,input_strength=0.0,ax=None):
        x = np.linspace(-30,30, 10)
        y = np.linspace(-30,30, 10)
        X, Y = np.meshgrid(x, y)
        K = np.zeros_like(X)

        xy = np.c_[X.ravel(), Y.ravel()]

        z = np.argmax(xy.dot(self.Rs.T) + self.r, axis=1)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        cmap = mcolors.ListedColormap(self.k_colors)
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)

        for k, (A, b, v) in enumerate(
            zip(self.As, self.bs, self.Vs)
        ):
            ival = np.ones(1) * input_strength
            v = v.dot(ival)
            dxydt_m = xy.dot(A.T) + b + v - xy

            U.ravel()[z == k] = dxydt_m[z == k, 0]
            V.ravel()[z == k] = dxydt_m[z == k, 1]
            K.ravel()[z == k] = k
        ax.streamplot(
            X,
            Y,
            U,
            V,
            color=K,
            linewidth=1,
            cmap=cmap,
            density=0.6,
            arrowsize=2,
        )

    def plot_summary(self, plot_scale=20):
        fig, ax = plt.subplots()
        
        # Plot the streamlines using the dynamics object
        self.plot_stream(ax=ax)
        
        fast = np.real(self.exp_fast)
        slow = np.real(self.exp_slow)
        ax.arrow(0,0,*fast*plot_scale,lw=1,head_width=0.1*plot_scale, head_length=0.1*plot_scale, fc='k', ec='k',alpha=0.5)
        ax.text(*fast*plot_scale,'fast')

        ax.arrow(0,0,*slow*plot_scale,lw=1,head_width=0.1*plot_scale, head_length=0.1*plot_scale, fc='k', ec='k',alpha=0.5)
        ax.text(*slow*plot_scale,'slow')


def plot_all_summary():
    fn = "dynamics.pkl"
    with open(fn, "rb") as f:
        D = pickle.load(f)
    for eid in D.keys():
        d = Dynamics(D[eid])
        d.plot_summary()
        title = f"{d.subject} {d.number} {d.genotype}"
        plt.title(title)
        plt.savefig(f"{d.subject}_{d.number}_{d.genotype}.pdf")
        plt.close()

def compute():
    fn = "dynamics.pkl"
    with open(fn, "rb") as f:
        D = pickle.load(f)


    df = pd.DataFrame()
    phases = ['insp','exp']
    eids =  list(D.keys())
    # Loop stimulus vector over all recordings
    for eid in eids:
        d = Dynamics(D[eid])
        # Loop stimulus vector over all phases
        for phase_1 in phases: 
            k1 = d.phase_k.index(phase_1)
            V = d.Vs[k1]
            stim_strength = np.linalg.norm(V)
            V = V/stim_strength
            V = V.ravel()
            # Loop comparison to dynamics over all recordings (including self)
            for eid in eids:
                d_compare = Dynamics(D[eid])
                # Loop comparison dynamics over all phases in comparison recording
                theta_fast = np.real(np.dot(d_compare.exp_fast, V))
                theta_slow = np.real(np.dot(d_compare.exp_slow, V))
                _df = pd.DataFrame()
                _df['dynamics_speed'] = ['slow','fast']
                _df['dot_product'] = [theta_slow, theta_fast]
                _df['phase_stimulus'] = phase_1
                _df['phase_dynamics']= 'exp'
                _df['eid_stimulus'] = d.eid
                _df['eid_dynamics'] = d_compare.eid
                _df['genotype_stimulus'] = d.genotype
                _df['genotype_dynamics'] = d_compare.genotype
                _df['stimulus_norm'] = stim_strength
                df = pd.concat([df,_df],ignore_index=True)

    return(df)

df = compute()
plot_all_summary()

# Plot self comparison 
df['self'] = df.eval('eid_stimulus == eid_dynamics')
df['self_genotype'] = df.eval('genotype_stimulus == genotype_dynamics')
PS = 3

df_use = df.query('self').copy()
p = (
    so.Plot(df_use, x='dynamics_speed', y='dot_product', color='genotype_stimulus')
    .facet(col='genotype_stimulus',row='phase_stimulus',)
    .add(so.Dots(),so.Jitter(),group='eid_stimulus',pointsize='stimulus_norm')
    .add(so.Lines(),so.Jitter(),group='eid_stimulus')
    # .add(so.Bar(),so.Agg())
    .limit(y=(-1.2,1.2))
    .scale(color=GENOTYPE_COLORS)
).plot()
p
p.save('all_dot_products.pdf')

for speed in ['slow','fast']:
    df_use = df.query('phase_stimulus == "insp" & dynamics_speed == @speed')
    df_use['self'] = df_use['self'].map({True:'Within',False:'Shuffle'})
    p = (
        so.Plot(df_use, y='self', x='dot_product', color='genotype_dynamics')
        .facet(row='genotype_dynamics',order=GENOTYPE_ORDER)
        .add(so.Range(alpha=0.5),so.Est(),legend=False)
        .add(so.Dash(width=0.25),so.Agg(),legend=False)
        .add(so.Dot(edgecolor='k'),so.Jitter(),so.Shift(y=-0.25),group='eid_dynamics',legend=False,alpha='self',pointsize='self')
        .label(x=f'$v_{{exp}}^{speed} \cdot v_{{insp}}^{{stim}}$',y='',color='Genotype')
        .scale(color=GENOTYPE_COLORS,alpha={'Within':1,'Shuffle':0.5},pointsize={'Within':PS,'Shuffle':PS/2})
        .layout(size=(2,2.5))
    ).plot()
    axs = p._figure.axes
    for ax in axs:
        ax.axvline(0, color='k', linestyle='--', lw=0.5)
        tt = ax.get_title()
        ax.set_title(GENOTYPE_LABELS[tt],fontsize='xx-small',color=GENOTYPE_COLORS[tt])
    p.save(f'{speed}_dynamics_dot_product.pdf')
    



df_self = df.query('self').reset_index()
df_use = df_self.query('phase_dynamics == "exp" and dynamics_speed == "slow"')
df_use['phase_stimulus'] = df_use['phase_stimulus'].map({'insp':'Insp.','exp':'Exp.'})
p = (
    so.Plot(
        df_use, y='phase_stimulus', x='stimulus_norm', color='genotype_stimulus'
    )
    .facet(row='genotype_stimulus',order=GENOTYPE_ORDER)
    .add(so.Bar(),so.Agg(),legend=False)
    .add(so.Range(color='k',linewidth=1),so.Est(),legend=False)
    .add(so.Dot(pointsize=3,edgecolor='k',color='w'),so.Jitter(),group='eid_stimulus',legend=False)
    .add(so.Lines(color='k',alpha=0.5),group='eid_stimulus',legend=False)
    .scale(color=GENOTYPE_COLORS)
    .label(y='',x=r'Stimulus strength ($|v^{stim}_{phase}|$)',color='Genotype')
    .layout(size=(2,2.5),engine='constrained')
).plot()
f = p._figure
f.supylabel('Phase of stimulus',fontsize='x-small')
axs = p._figure.axes
for ax in axs:
    ax.axvline(0, color='k', linestyle='--', lw=0.5)
    tt = ax.get_title()
    ax.set_title(GENOTYPE_LABELS[tt],fontsize='xx-small',color=GENOTYPE_COLORS[tt])
p.save('stimulus_strength_by_phase.pdf')
