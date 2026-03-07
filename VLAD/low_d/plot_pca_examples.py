"""
Plot PCA examples. Not summary statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import click

sys.path.append("../")
sys.path.append("VLAD/")
from utils import (
    EIDS_NEURAL,
    Rec,
    one,
    GENOTYPE_COLORS,
    HB_MAP,
    GENOTYPE_LABELS,
    set_style,
)
from compute_pca_perturbations import compute_expiratory_attractor
from cibrrig.analysis.population import Population
from cibrrig.plot import replace_timeaxis_with_scalebar, trim_yscale_to_lims
from scipy.signal import coherence
import seaborn as sns
from matplotlib.collections import LineCollection
from cycler import cycler
from pathlib import Path

ATTRACTOR_PLOT_KWARGS = {
    'marker': 'o',
    'color': 'w',
    'markersize': 8,
    'markeredgewidth': 1.5,
    'markeredgecolor': 'k'
}

set_style()

PHASE_LABELS = {
    "insp": "Inspiration triggered",
    "exp": "Expiration triggered",
}
EXAMPLE_FIGS = Path('example_figs')


class Rec(Rec):
    def __init__(self, one, eid, load_raw_dia=False):
        super().__init__(one, eid, load_raw_dia=load_raw_dia)
        self.pop = Population(self.spikes.times, self.spikes.clusters, t0=300, tf=600)
        self.pop.compute_projection()
        self.pop.compute_projection_speed()
        self.phi2 = self.pop.sync_var(self.phi, self.phi_t)
        self.coherence_ordered = self.compute_coherence_dims(ndims=10)

    def plot_stim_projection_phasic(self, figsize=(2.5, 3)):
        pop = self.pop
        phi2 = self.phi2
        coherence_ordered = self.coherence_ordered

        f, ax = plt.subplots(
            2, 1, figsize=figsize, sharex=True, sharey=True, constrained_layout=True
        )
        for ii, phase in enumerate(["insp", "exp"]):
            intervals, stims = self.get_stims(phase)
            t0, tf = stims.iloc[0][["start_time", "end_time"]]
            intervals = intervals[
                np.logical_and(intervals[:, 0] > t0, intervals[:, 1] < tf)
            ]

            pop.plot_projection_line(
                dims=coherence_ordered[:2],
                t0=500,
                tf=520,
                cvar=phi2,
                cmap="RdBu_r",
                colorbar_title=r"$\phi$ (rads.)",
                alpha=0.5,
                plot_colorbar=False,
                ax=ax[ii],
            )
            pop.plot_projection_line(
                t0=t0,
                tf=tf,
                dims=coherence_ordered[:2],
                intervals=intervals,
                colorbar_title="Stimulus",
                alpha=0.5,
                ax=ax[ii],
                stim_color=self.laser_color,
                base_color="k",
            )
            ax[ii].set_title(PHASE_LABELS[phase])
            ax[ii].legend().set_visible(False)
        attractor = compute_expiratory_attractor(self, self.pop, ndims=2, t0=500, tf=520)
        for ii in range(2):
            ax[ii].plot(attractor[0], attractor[1], **ATTRACTOR_PLOT_KWARGS)
        plt.suptitle(
            GENOTYPE_LABELS[self.genotype],
            color=GENOTYPE_COLORS[self.genotype],
            fontsize='medium',
            # y=0.90,
            ha="center",
            va="top",
        )

    def plot_stim_projection_pulse(self, figsize=(2.5, 3)):
        pop = self.pop
        phi2 = self.phi2

        coherence_ordered = self.coherence_ordered
        intervals, stims = self.get_stims("50ms")

        f, ax = plt.subplots(
            1, 1, figsize=figsize, sharex=True, sharey=True, constrained_layout=True
        )
        t0 = stims.iloc[0]["start_time"] - 1
        tf = stims.iloc[-1]["end_time"]

        pop.plot_projection_line(
            dims=coherence_ordered[:2],
            t0=500,
            tf=520,
            cvar=phi2,
            cmap="RdBu_r",
            colorbar_title=r"$\phi$ (rads.)",
            alpha=0.5,
            plot_colorbar=False,
            ax=ax,
        )
        pop.plot_projection_line(
            t0=t0,
            tf=tf,
            dims=coherence_ordered[:2],
            intervals=intervals,
            colorbar_title="Stimulus",
            alpha=0.5,
            ax=ax,
            stim_color=self.laser_color,
            base_color="none",
            use_arrow=True,
            multi_arrow=True,
            mutation_scale=5,
        )
        ax.legend().set_visible(False)
        attractor = compute_expiratory_attractor(self, self.pop, ndims=2, t0=500, tf=520)
        ax.plot(attractor[0], attractor[1], **ATTRACTOR_PLOT_KWARGS)
        plt.title(
            GENOTYPE_LABELS[self.genotype],
            color=GENOTYPE_COLORS[self.genotype],
            fontsize=plt.rcParams["axes.titlesize"] + 1,
            ha="center",
            va="top",
        )

    def plot_stim_projection_hb(self, figsize=(2.5, 3)):
        pop = self.pop
        phi2 = self.phi2

        coherence_ordered = self.coherence_ordered
        intervals, stims = self.get_stims("hb")

        # TODO: Fix the bug here with plotting only one stim sequence
        f, ax = plt.subplots(
            1, 1, figsize=figsize, sharex=True, sharey=True, constrained_layout=True
        )
        t0 = stims.iloc[0]["start_time"] - 1
        tf = stims.iloc[0]["end_time"] + 5

        pop.plot_projection_line(
            dims=coherence_ordered[:2],
            t0=500,
            tf=520,
            cvar=phi2,
            cmap="RdBu_r",
            colorbar_title=r"$\phi$ (rads.)",
            alpha=0.5,
            plot_colorbar=True,
            ax=ax,
        )
        pop.plot_projection_line(
            t0=t0,
            tf=tf,
            dims=coherence_ordered[:2],
            intervals=intervals[:1,:],
            colorbar_title="Stimulus",
            alpha=0.5,
            ax=ax,
            stim_color=HB_MAP[5],
            base_color="none",
            use_arrow=True,
            multi_arrow=True,
            mutation_scale=5,
        )
        ax.legend().set_visible(False)
        attractor = compute_expiratory_attractor(self, self.pop, ndims=2, t0=500, tf=520)
        ax.plot(attractor[0], attractor[1], **ATTRACTOR_PLOT_KWARGS)

    def plot_stim_projection_time(self, figsize=(2.5, 1.25),stim='hb',ndims=3):
        pop = self.pop
        coherence_ordered = self.coherence_ordered
        dims = coherence_ordered[:ndims]

        intervals, stims = self.get_stims(stim)
        stim_color = HB_MAP[5] if stim == 'hb' else self.laser_color

        t0 = stims.iloc[1]["start_time"] - 1
        tf = stims.iloc[1]["end_time"] + 2

        stim_on, stim_off = stims.iloc[1][["start_time", "end_time"]]

        # gridspec
        f = plt.figure(figsize=figsize, constrained_layout=True)
        gs = f.add_gridspec(2, 1, height_ratios=[0.4, 1])
        ax0 = f.add_subplot(gs[0])
        ax1 = f.add_subplot(gs[1], sharex=ax0)
        # Plot dia
        s0, sf = np.searchsorted(self.diaphragm.times, [t0, tf])
        t = self.diaphragm.times[s0:sf]
        x = self.diaphragm.filtered[s0:sf]
        ax0.plot(t, x, color=plt.rcParams["text.color"])
        ax0.set_ylim(np.min(x), np.max(x))
        ax0.axis("off")
        ax0.text(
            t0,
            0,
            r"Dia. EMG",
            rotation=90,
            ha="right",
            va="center",
            fontsize="xx-small",
        )

        # plot projection line
        s0, sf = np.searchsorted(pop.tbins, [t0, tf])
        t = pop.tbins[s0:sf]
        x = pop.projection[s0:sf, dims]
        ax1.set_prop_cycle(
            cycler("color", plt.cm.Greys(np.linspace(1, 0.4, x.shape[1])))
        )
        ax1.plot(t, x)
        ax1.set_xlim(t0, tf)
        ymax = np.ceil(np.max(x))

        sns.despine(ax=ax0, left=True, trim=True)
        for ax in [ax0, ax1]:
            ax.axvspan(stim_on, stim_off, color=stim_color, alpha=0.2)
        # Set x spine off
        ax1.spines["bottom"].set_visible(False)
        ax1.set_yticks([-ymax, 0, ymax])
        ax1.set_ylim(-ymax, ymax)
        ax1.set_xticks([])
        ax1.hlines(
            -ymax + 0.2, t0 + 0.1, t0 + 1.1, color=plt.rcParams["text.color"], lw=0.5
        )
        ax1.text(t0 + 0.1, -ymax + 0.1, "1s", va="top", fontsize="xx-small")
        ax1.set_ylabel("PCs (a.u.)")

        # plot speed line
        axspeed = ax1.twinx()
        s0, sf = np.searchsorted(pop.tbins, [t0, tf])
        t = pop.tbins[s0:sf]
        x = pop.projection_speed[s0:sf] * 10
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.min(x), vmax=np.max(x))
        points = np.array([t, x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=0.5)
        lc.set_array(x)

        # Add LineCollection to the axis
        axspeed.add_collection(lc)
        axspeed.set_ylim(-10, np.max(x) * 1.1)
        axspeed.axis("off")
        axspeed.text(
            1,
            0.9,
            "speed",
            transform=axspeed.transAxes,
            ha="left",
            va="center",
            fontsize="xx-small",
            rotation=90,
        )

    def compute_coherence_dims(self, ndims=10):
        # Compute coherence with Phi using scipy
        pop = self.pop
        phi2 = self.phi2
        C = []
        for ii in range(ndims):
            t, c = coherence(
                phi2, pop.projection[:, ii], fs=1 / np.diff(pop.tbins).mean()
            )
            C.append(np.max(c))
        C = np.array(C)
        return np.argsort(C)[::-1]


@click.command()
@click.option("--ext", default="pdf", help="File extension")
def main(ext):
    use_data = {
        "m2025-01": 0,
        "m2024-30": 0,
        "m2024-34": 1,
        "m2024-40": 2,

    }
    for subject, sequence in use_data.items():
        eid = one.search(subject=subject, datasets="spikes.times.npy", number=sequence)[0]
        rec = Rec(one, eid, load_raw_dia=True)
        rec.plot_stim_projection_phasic()
        fn = EXAMPLE_FIGS.joinpath(f'{subject}_g{sequence}_phasic_stims.{ext}')
        plt.savefig(fn)
        plt.close("all")

        rec.plot_stim_projection_pulse()
        fn = EXAMPLE_FIGS.joinpath(f'{subject}_g{sequence}_pulse_stims.{ext}')
        plt.savefig(fn)
        plt.close("all")

        rec.plot_stim_projection_hb()
        fn = EXAMPLE_FIGS.joinpath(f'{subject}_g{sequence}_hb_stims.{ext}')
        plt.savefig(fn)
        plt.close("all")

        for stim in ['hb','hold']:
            rec.plot_stim_projection_time(stim=stim)
            fn = EXAMPLE_FIGS.joinpath(f'{subject}_g{sequence}_{stim}_time_stims.{ext}')
            plt.savefig(fn)
            plt.close("all")

        del rec


if __name__ == "__main__":
    main()
