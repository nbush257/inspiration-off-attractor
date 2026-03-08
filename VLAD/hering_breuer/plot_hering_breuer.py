"""
Comptue and plot data about the Hering-Breuer stimulations.
There is a ton of inefficiency in this code (multiple re-loading of the same data)
It can be  improved in cleanliness and speed by making many of the computations methods of the recording class.
"""

import sys

sys.path.append("../")
sys.path.append("VLAD/")
from utils import (
    one,
    EIDS_NEURAL,
    EIDS_PHYSIOL,
    Rec,
    PHASE_MAP,
    GENOTYPE_COLORS,
    GENOTYPE_LABELS,
    QC_QUERY,
    HB_MAP,
    VLAD_ROOT,
    get_prefix,
    set_style,
    sessions_to_include,
    sig2star
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn.objects as so
from cibrrig.analysis.population import Population, smooth_raster
from tqdm import tqdm
import seaborn as sns
from cibrrig.utils.utils import weighted_histogram
import brainbox.singlecell as bsc
from scipy.stats import spearmanr, ttest_rel
from cibrrig.plot import (
    plot_sweeps,
    clean_polar_axis,
    clean_linear_radial_axis,
    trim_yscale_to_lims,
    replace_timeaxis_with_scalebar,
)
from cibrrig.analysis.singlecell import get_phase_curve
from cibrrig.utils.utils import get_eta
from scipy.stats import ttest_rel, wilcoxon
import click
import brainbox.metrics.single_units as bbmetrics

import pingouin as pg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

set_style()

LABELS = {"hb": "Hering-Breuer", "control": "Control"}
UNIT_TYPES = {"insp": "Inspiratory", "exp": "Expiratory", "tonic": "Tonic"}
CONDITION_COLORS = {"control": "#333333", 'random': "#333333","hb": HB_MAP[5]}
from cibrrig.plot import laser_colors
PC_FIGURES = Path("pca_figures")
PC_FIGURES.mkdir(exist_ok=True)
POPULATION_FIGURES = Path("population_peth_figures")
POPULATION_FIGURES.mkdir(exist_ok=True)


class Rec(Rec):
    def __init__(self, one, eid, curate=True, load_spikes=True, load_raw_dia=False):
        super().__init__(one, eid, curate, load_spikes, load_raw_dia)
        self.breaths_control = None
        self.breaths_hb = None
        if self.has_spikes:
            self.pop = Population(
                self.spikes.times, self.spikes.clusters, t0=300, tf=600, binsize=0.005
            )
            self.pop.compute_projection()
            self.pop.compute_projection_speed()
            self.phi2 = self.pop.sync_var(self.phi, self.phi_t)

    def get_condition_breaths(self):
        if self.breaths_control is not None:
            return self.breaths_control, self.breaths_hb
        intervals = self.get_HB_stims(duration=5)[0]
        breaths = self.breaths.to_df()
        breaths_control = pd.DataFrame()
        breaths_hb = pd.DataFrame()
        for t0, tf in intervals:
            cf = t0
            c0 = cf - 5
            breaths_control = pd.concat(
                [breaths_control, breaths.query(f"on_sec>{c0} & off_sec<{cf}")]
            )
            breaths_hb = pd.concat(
                [breaths_hb, breaths.query(f"on_sec>{t0} & off_sec<{tf}")]
            )
        self.breaths_control = breaths_control
        self.breaths_hb = breaths_hb
        return breaths_control, breaths_hb

    def get_single_unit_peths(self, event="on_sec"):
        breaths_control, breaths_hb = self.get_condition_breaths()
        control_peth, binned_spikes = bsc.calculate_peths(
            self.spikes.times,
            self.spikes.clusters,
            np.unique(self.spikes.clusters),
            breaths_control[event],
            pre_time=0.25,
            post_time=0.5,
            bin_size=0.025,
            smoothing=0.01,
            return_fr=True,
        )

        hb_peth, binned_spikes = bsc.calculate_peths(
            self.spikes.times,
            self.spikes.clusters,
            np.unique(self.spikes.clusters),
            breaths_hb[event],
            pre_time=0.25,
            post_time=0.5,
            bin_size=0.025,
            smoothing=0.01,
            return_fr=True,
        )
        DF = pd.DataFrame()
        for condition in ["control", "hb"]:
            if condition == "control":
                peth = control_peth
            else:
                peth = hb_peth
            for ii, cid in enumerate(peth["cscale"]):
                ts = self.spikes.times[self.spikes.clusters == cid]
                cv, cvs, fr = bbmetrics.firing_rate_coeff_var(ts)
                std = np.mean(fr) * cv
                m = peth["means"][ii]
                n = m.shape[0]
                s = peth["stds"][ii] / np.sqrt(n)
                t = peth["tscale"]
                df = pd.DataFrame(
                    {"t": np.round(t, 2), "mean": m, "lb": m - s, "ub": m + s}
                )
                df["mean_norm"] = m / std
                df["lb_norm"] = (m - s) / std
                df["ub_norm"] = (m + s) / std
                df["condition"] = condition
                df["uuid"] = self.clusters.uuids[cid]
                df["eid"] = self.eid
                df["cid"] = cid
                df["event"] = event
                df["category"] = self.clusters.category[cid]
                DF = pd.concat([DF, df])
        return DF

    def comput_preI_spikerates(self, pre_time=0.05):
        """
        Compute the firing rate of each unit in the 50ms before the HB stim
        """
        breaths = self.get_condition_breaths()
        labels = ["control", "hb"]
        df = pd.DataFrame()
        for label, _breaths in zip(labels, breaths):
            onsets = _breaths["on_sec"].values
            peth, _ = bsc.bin_spikes2D(
                self.spikes.times,
                self.spikes.clusters,
                self.cluster_ids,
                onsets,
                pre_time=pre_time,
                post_time=0,
                bin_size=pre_time,
            )
            preI_fr = np.squeeze(peth.mean(axis=0)) / pre_time
            _df = pd.DataFrame()
            _df["preI_fr"] = preI_fr
            _df["condition"] = label
            _df["eid"] = self.eid
            _df["category"] = self.clusters.category[self.cluster_ids]
            _df["respMod"] = self.clusters.respMod[self.cluster_ids]
            _df["uuid"] = self.clusters.uuids[self.cluster_ids].values
            df = pd.concat([df, _df])
        return df

    def compute_pca_phase_curve(self):
        pop = self.pop
        DF = pd.DataFrame()
        for ii in range(3):
            bins, control_x, control_err, hb_x, hb_err = self.compute_phase_histogram(
                pop.projection[:, ii],
            )
            for condition in ["control", "hb"]:
                if condition == "control":
                    val = control_x
                    err = control_err
                else:
                    val = hb_x
                    err = hb_err

                df = pd.DataFrame({"Phase (rads.)": bins, "val": val, "err": err})
                df["condition"] = condition
                df["eid"] = self.eid
                df["pc"] = ii
                DF = pd.concat([DF, df])
        DF = DF.reset_index(drop=True)
        return DF

    def compute_pca_PETH(self, ndims=3, event="on_sec", pre_time=0.25, post_time=3):
        breaths_control, breaths_hb = self.get_condition_breaths()
        pop = self.pop

        df = pd.DataFrame()
        conditions = ["control", "hb"]
        for ii in range(ndims):
            for condition in conditions:
                breaths = breaths_control if condition == "control" else breaths_hb
                _df = pd.DataFrame()
                eta = get_eta(
                    pop.projection[:, ii],
                    pop.tbins,
                    breaths[event],
                    pre_win=pre_time,
                    post_win=post_time,
                )
                _df = pd.DataFrame(eta)
                _df["condition"] = condition
                _df["pc"] = ii
                _df["event"] = event
                t = np.linspace(-pre_time, post_time, _df.shape[0])
                _df["t"] = np.round(t, 4)
                df = pd.concat([df, _df])
        df["eid"] = self.eid
        df = df.reset_index(drop=True)

        return df

    def compute_pc_speeds_by_phase(self, nbins=25):
        pop = Population(
            self.spikes.times, self.spikes.clusters, t0=300, tf=600, binsize=0.005
        )
        pop.compute_projection()
        pop.compute_projection_speed()
        bins, control_speed, control_err, hb_speed, hb_err = (
            self.compute_phase_histogram(pop.projection_speed, nbins=nbins)
        )

        control = pd.DataFrame(
            {"tscale": bins, "speed": control_speed, "err": control_err}
        )
        control["condition"] = "control"
        hb = pd.DataFrame({"tscale": bins, "speed": hb_speed, "err": hb_err})
        hb["condition"] = "hb"
        df = pd.concat([control, hb]).reset_index(drop=True)
        df["eid"] = self.eid

        return df

    def compute_pc_speed_PETH(self, event="on_sec"):
        control_breaths, hb_breaths = self.get_condition_breaths()
        pop = self.pop

        control = get_eta(
            pop.projection_speed,
            pop.tbins,
            control_breaths[event],
            pre_win=0.25,
            post_win=0.5,
        )
        hb = get_eta(
            pop.projection_speed,
            pop.tbins,
            hb_breaths[event],
            pre_win=0.25,
            post_win=0.5,
        )

        control = pd.DataFrame(control)
        control["condition"] = "control"
        hb = pd.DataFrame(hb)
        hb["condition"] = "hb"
        df = pd.concat([control, hb])
        df["eid"] = self.eid
        df["t"] = np.round(df["t"], 2)
        df["event"] = event
        return df

    def compute_phase_histogram(self, x, nbins=25):
        pop = self.pop
        phi = self.phi
        phi_t = self.phi_t
        phi2 = pop.sync_var(phi, phi_t)

        intervals = self.get_HB_stims(duration=5)[0]
        control_vals = []
        hb_vals = []
        bins = np.linspace(-np.pi, np.pi, nbins)
        for t0, tf in intervals:
            cf = t0
            c0 = cf - 5
            s0, sf = np.searchsorted(pop.tbins, [c0, cf])
            _phi = phi2[s0:sf]
            _x = x[s0:sf]
            _, vals = weighted_histogram(_phi, _x, bins=bins, wrap=True)
            control_vals.append(vals)

            s0, sf = np.searchsorted(pop.tbins, [t0, tf])
            _phi = phi2[s0:sf]
            _x = x[s0:sf]
            _, vals = weighted_histogram(_phi, _x, bins=bins, wrap=True)
            hb_vals.append(vals)
        control_vals = np.stack(control_vals)
        hb_vals = np.stack(hb_vals)

        control_x = control_vals.mean(axis=0)
        control_err = control_vals.std(axis=0)
        hb_x = hb_vals.mean(axis=0)
        hb_err = hb_vals.std(axis=0)
        return bins, control_x, control_err, hb_x, hb_err

    def get_firing_rate_in_phase(self):
        """Get the mean firingg rate of each neruon in each phase of the respiratory cycle in control and hb"""
        intervals = self.get_HB_stims(duration=5)[0]
        intervals_control = np.c_[intervals[:, 0] - 5, intervals[:, 0]]
        n_stims = intervals.shape[0]
        spike_samps = self.phi_t.searchsorted(self.spikes.times) - 1
        is_insp = self.phi[spike_samps] > 0
        df = pd.DataFrame(
            columns=["control", "hb", "category", "respMod", "eid"],
            index=self.cluster_ids,
        )

        
        durations = {"control": {"insp": 0, "exp": 0}, "hb": {"insp": 0, "exp": 0}}
        dt = np.diff(self.phi_t)[0]
        for ii in range(n_stims):
            c0, cf = self.phi_t.searchsorted(intervals_control[ii])
            s0, sf = self.phi_t.searchsorted(intervals[ii])
            durations["control"]["insp"] += (self.phi[c0:cf] > 0).sum() * dt
            durations["control"]["exp"] += (self.phi[c0:cf] <= 0).sum() * dt
            durations["hb"]["insp"] += (self.phi[s0:sf] > 0).sum() * dt
            durations["hb"]["exp"] += (self.phi[s0:sf] <= 0).sum() * dt

        for clu_id in self.cluster_ids:
            # Filter for only spikes from this unit in the correct phase
            category = self.clusters.category[clu_id]
            if category == "tonic":
                df.loc[clu_id] = [np.nan, np.nan, category, np.nan, self.eid]
                continue
            unit_phase = (
                is_insp if self.clusters.category[clu_id] == "insp" else ~is_insp
            )
            idx = np.logical_and(
                self.spikes.clusters == clu_id,  # This cluster
                unit_phase,  # During congruent phase
            )
            ts = self.spikes.times[idx]

            control_spikes = 0
            hb_spikes = 0
            control_duration = 0
            hb_duration = 0
            for ii in range(n_stims):
                control_idx = np.logical_and(
                    ts >= intervals_control[ii, 0], ts < intervals_control[ii, 1]
                )
                hb_idx = np.logical_and(ts >= intervals[ii, 0], ts < intervals[ii, 1])
                control_spikes += control_idx.sum()
                hb_spikes += hb_idx.sum()
                control_duration += np.diff(intervals_control[ii])[0]
                hb_duration += np.diff(intervals[ii])[0]
            control_duration = durations["control"][category]
            hb_duration = durations["hb"][category]
            control_fr = control_spikes / control_duration
            hb_fr = hb_spikes / hb_duration
            respMod = self.clusters.respMod[clu_id]
            df.loc[clu_id] = [control_fr, hb_fr, category, respMod, self.eid]
        return df

    def compute_stim_aligned_PCA(self, ndims=3):
        pop = self.pop
        intervals = self.get_HB_stims(duration=5)[0]
        df = pd.DataFrame()
        for ii in range(ndims):
            eta = get_eta(
                pop.projection[:, ii], pop.tbins, intervals[:, 0], pre_win=5, post_win=5
            )
            eta["pc"] = ii
            df = pd.concat([df, pd.DataFrame(eta)])
        df["eid"] = self.eid
        df.reset_index(drop=True, inplace=True)
        return df

    def compute_stim_aligned_speed_PCA(self, pre_time=3, post_time=3):
        df = pd.DataFrame()
        for stimulus in ['hb','hold']:
            pop = self.pop
            if stimulus == 'hb':
                intervals = self.get_HB_stims(duration=5)[0]
            else:
                intervals = self.get_stims(stimulus)[0]
            pre_samps = int(pre_time / pop.binsize)
            win_samps = int((5 + post_time) / pop.binsize)
            # t = np.linspace(-pre_time,5+post_time,win_samps+pre_samps)
            t = np.arange(-pre_time, 5 + post_time, pop.binsize)
            for ii, (on, off) in enumerate(intervals):
                s0 = np.searchsorted(pop.tbins, on)
                _df = pd.DataFrame()
                _df["speed"] = pop.projection_speed[s0 - pre_samps : s0 + win_samps]
                _df["t"] = t
                _df["stim_num"] = ii
                _df['stimulus_type'] = stimulus
                df = pd.concat([df, _df])

            df["eid"] = self.eid
            df.reset_index(drop=True, inplace=True)
        return df

    def plot_pca_sweeps(self, fs=(3, 2), ext="pdf"):
        intervals = self.get_HB_stims(5)[0]
        events = intervals[:, 0]
        f, axs = plt.subplots(
            figsize=fs, nrows=3, sharex=True, sharey=True, constrained_layout=True
        )
        colors = ["C4", "C5", "C6"]
        for ii, ax in enumerate(axs):
            color = colors[ii]
            plot_sweeps(
                self.pop.tbins,
                self.pop.projection[:, ii],
                events,
                pre=3,
                post=8,
                ax=ax,
                color=color,
            )
            ax.axvspan(0, 5, color=CONDITION_COLORS["hb"], alpha=0.3)

    def plot_average_dia(self, pre_win=0.1, post_win=0.3, fs=(1.5, 1.5), ext="pdf"):
        control_breaths, hb_breaths = self.get_condition_breaths()
        avg = {}
        conditions = ["control", "hb"]
        breaths = [control_breaths, hb_breaths]
        for condition, _breaths in zip(conditions, breaths):
            eta = get_eta(
                self.physiology.dia,
                self.physiology.times,
                _breaths["on_sec"],
                pre_win=pre_win,
                post_win=post_win,
            )
            avg[condition] = eta

        f = plt.figure(figsize=fs, constrained_layout=True)
        ax = f.add_subplot(111)
        for condition in conditions:
            mm = avg[condition]["mean"]
            lb = avg[condition]["lb"]
            ub = avg[condition]["ub"]
            t = avg[condition]["t"]
            ax.plot(t, mm, color=CONDITION_COLORS[condition], label=LABELS[condition])
            ax.fill_between(t, lb, ub, color=CONDITION_COLORS[condition], alpha=0.3)
        ax.axvline(0, color="k", ls="--")
        ax.set_xlim(-pre_win, post_win)
        ax.set_ylim(0, None)
        # ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\int{Dia}  (\sigma)$")
        plt.savefig(f"{self.subject}_{self.sequence}_average_dia.{ext}")


# ------------------------- #
# Computation functions
# ------------------------- #
def load_data():
    save_fn_firing_rates = VLAD_ROOT.joinpath("singlecell/all_firing_rates.pqt")

    mean_frs_raw = pd.read_parquet(save_fn_firing_rates)
    mean_frs = mean_frs_raw.pivot(
        values="fr", columns=["condition"], index=["category", "uuids"]
    )

    # Load cluster level features
    cluster_features_fn = VLAD_ROOT.joinpath("singlecell/cluster_features.pqt")
    cluster_features = pd.read_parquet(cluster_features_fn)
    # Identify units that did not pass QC and set those categorues to qc_fail
    idx = cluster_features.query(f"~({QC_QUERY})").index
    cluster_features.loc[idx, "category"] = "qc_fail"
    cluster_features.set_index("uuids", inplace=True)

    # Compute delta log FR for each unit in each condition compared to control
    TINY = 1e-2
    pct_diff = mean_frs[["insp", "exp", "hold", "hb"]].div(mean_frs["control"], axis=0)[
        ["insp", "exp", "hold", "hb"]
    ]
    pct_diff = (
        pct_diff.melt(ignore_index=False, value_name="FR_pct_change")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    pct_diff["deltalogFR"] = np.log2(pct_diff["FR_pct_change"] + TINY)
    pct_diff.index.map(GENOTYPE_LABELS)

    mean_frs = mean_frs[["control", "hb"]].reset_index()

    return pct_diff, mean_frs, cluster_features


def get_breath_durations():
    # Plot breath duration
    breath_durations = pd.DataFrame()
    for eid in EIDS_PHYSIOL:
        rec = Rec(one, eid, load_spikes=False)
        control_breaths, hb_breaths = rec.get_condition_breaths()
        if control_breaths.shape[0] == 0:
            continue
        control_breath_duration = control_breaths["duration_sec"].mean()
        if hb_breaths.shape[0] == 0:
            hb_breath_duration = 0
        else:
            hb_breath_duration = hb_breaths["duration_sec"].mean()
        breath_durations = pd.concat(
            [
                breath_durations,
                pd.DataFrame(
                    {"control": control_breath_duration, "hb": hb_breath_duration},
                    index=[eid],
                ),
            ]
        )

    breath_durations.index.name = "eid"
    breath_durations_melt = breath_durations.melt(
        ignore_index=False, value_name="Inspiration duration", var_name="Condition"
    ).reset_index()

    stats = wilcoxon(breath_durations["control"], breath_durations["hb"])
    breath_durations.to_csv("breath_durations.csv")
    stats = pd.DataFrame(
        {
            "wilcoxon": stats.statistic,
            "p": stats.pvalue,
            "n": breath_durations.shape[0],
        },
        index=["breath_duration"],
    )
    stats.to_csv("breath_duration_stats.csv")
    print(f"Control vs HB breath duration: p={stats.loc['breath_duration', 'p']:.3f}")
    return breath_durations, breath_durations_melt


def get_time_since_offset(t, breaths):
    t_out = np.zeros_like(t)
    offs = breaths.off_sec.astype(float)
    for off in tqdm(offs):
        t_out[t > off] = t[t > off] - off
    return t_out


def run_all_eids():
    """
    Compute all the data for all the experiments
    """

    # Preallocate the dataframes to store the results
    all_pca_speed_phase = pd.DataFrame()
    all_pca_speed_PETH = pd.DataFrame()
    all_pca_phase_curves = pd.DataFrame()
    all_unit_peth = pd.DataFrame()
    all_firing_rate_in_phase = pd.DataFrame()
    all_pca_stim_aligned = pd.DataFrame()
    all_pca_peth = pd.DataFrame()
    all_pca_speed_stim_aligned = pd.DataFrame()
    all_preI_spikerates = pd.DataFrame()

    # Computation loop
    for eid in tqdm(EIDS_NEURAL):
        rec = Rec(one, eid)
        all_pca_speed_phase = pd.concat(
            [all_pca_speed_phase, rec.compute_pc_speeds_by_phase(nbins=25)]
        )
        for event in ["on_sec", "off_sec"]:
            all_pca_speed_PETH = pd.concat(
                [all_pca_speed_PETH, rec.compute_pc_speed_PETH(event=event)]
            )
            all_unit_peth = pd.concat(
                [all_unit_peth, rec.get_single_unit_peths(event=event)]
            )
            all_pca_peth = pd.concat([all_pca_peth, rec.compute_pca_PETH(event=event)])

        all_pca_stim_aligned = pd.concat(
            [all_pca_stim_aligned, rec.compute_stim_aligned_PCA()]
        )

        all_pca_phase_curves = pd.concat(
            [all_pca_phase_curves, rec.compute_pca_phase_curve()]
        )
        all_firing_rate_in_phase = pd.concat(
            [all_firing_rate_in_phase, rec.get_firing_rate_in_phase()]
        )

        all_pca_speed_stim_aligned = pd.concat(
            [all_pca_speed_stim_aligned, rec.compute_stim_aligned_speed_PCA()]
        )
        all_preI_spikerates = pd.concat(
            [all_preI_spikerates, rec.comput_preI_spikerates()]
        )
        del rec

    # Reset the index
    all_pca_speed_phase.reset_index(drop=True, inplace=True)
    all_pca_speed_PETH.reset_index(drop=True, inplace=True)
    all_pca_phase_curves.reset_index(drop=True, inplace=True)
    all_unit_peth.reset_index(drop=True, inplace=True)
    all_firing_rate_in_phase.reset_index(drop=True, inplace=True)
    all_pca_stim_aligned.reset_index(drop=True, inplace=True)
    all_pca_peth.reset_index(drop=True, inplace=True)
    all_pca_speed_stim_aligned.reset_index(drop=True, inplace=True)
    all_preI_spikerates.reset_index(drop=True, inplace=True)

    # Save to Parquet
    all_unit_peth.to_parquet("all_unit_hb_peth.pqt")
    all_pca_speed_phase.to_parquet("all_pca_speed_phase.pqt")
    all_pca_speed_PETH.to_parquet("all_pca_speed_peth.pqt")
    all_pca_phase_curves.to_parquet("all_pca_phase_curves.pqt")
    all_firing_rate_in_phase.to_parquet("all_firing_rate_in_phase.pqt")
    all_pca_stim_aligned.to_parquet("all_pca_stim_aligned.pqt")
    all_pca_peth.to_parquet("all_pca_peth.pqt")
    all_pca_speed_stim_aligned.to_parquet("all_pca_speed_stim_aligned.pqt")
    all_preI_spikerates.to_parquet("all_preI_spikerates.pqt")


# ------------------------- #
# Plotting functions#
# ------------------------- #


def plot_scatter_control_vs_hb(mean_frs, fs=(5, 3), ext="pdf"):
    ps = 1
    half_line = lambda x: x / 2
    double_line = lambda x: x * 2
    x = np.logspace(-2, 4, 100)

    p = (
        so.Plot(mean_frs, x="control", y="hb", color="category")
        .facet(col="category", order=["insp", "exp", "tonic"])
        .add(so.Dot(pointsize=ps, edgewidth=0, alpha=0.5), legend=False)
        .scale(
            x=so.Continuous(trans="symlog"),
            y=so.Continuous(trans="symlog"),
            color=PHASE_MAP,
        )
        .layout(size=fs, engine="constrained")
    ).plot()
    for ii, ax in enumerate(p._figure.axes):
        ax.axline([1e-2, 1e-2], [1e2, 1e2], color="k", linestyle="--")
        ax.plot(x, double_line(x), color="gray", linestyle="--")
        ax.plot(x, half_line(x), color="gray", linestyle="--")

        ax.set_aspect("equal")
        ax.set_xlim(1e-2, 2e2)
        ax.set_ylim(1e-2, 2e2)
        ax.set_title(UNIT_TYPES[ax.get_title()])
        ax.set_ylabel("")
        ax.set_xlabel("")
        if ii == 0:
            ax.text(
                50,
                100,
                "2x",
                color="gray",
                fontsize="xx-small",
                rotation=45,
                va="bottom",
                ha="right",
            )
            ax.text(
                100,
                75,
                "0.5x",
                color="gray",
                fontsize="xx-small",
                rotation=45,
                va="top",
                ha="left",
            )

    p._figure.supxlabel("Control firing rate (sp/s)", y=0.15, fontsize=8)
    p._figure.supylabel("Hering-Breuer firing rate (sp/s)", fontsize=8)
    p.save(f"control_vs_hb.{ext}")
    return p


def plot_CDF_control_vs_hb(pct_diff, fs=(1.5, 1.75), ext="pdf"):
    p = (
        so.Plot(pct_diff, x="deltalogFR", color="category")
        .add(
            so.Line(),
            so.Hist("percent", cumulative=True, common_norm=False),
            legend=False,
        )
        .scale(color=PHASE_MAP)
        .scale(
            x=so.Continuous().tick(at=[-2, -0, 2]),
            y=so.Continuous().tick(at=[0, 25, 50, 75, 100]),
        )
        .layout(size=fs, engine="constrained")
        .limit(x=(-4, 4))
        .label(x=r"$\frac{FR_{HB}}{FR_{control}}$", y="Cumulative %")
    ).plot()
    for ii, ax in enumerate(p._figure.axes):
        ax.axvline(0, color="k", linestyle="--")
        ax.axhline(50, color="k", linestyle="--")
        ax.grid(axis="y", ls="--", lw=0.25)
        ax.set_xticks([-2, 0, 2])
        ax.set_xticklabels(["0.25x", "1x", "4x"])

    p.save(f"CDF_control_vs_hb.{ext}")


def plot_delta_FR_along_AP(merged, fs=(4, 6), binwidth=150, ext="pdf"):
    f = plt.figure(figsize=fs, constrained_layout=True)
    gs = f.add_gridspec(4, 1, height_ratios=[2, 2, 2, 1])
    ax1 = f.add_subplot(gs[0])
    ax2 = f.add_subplot(gs[1], sharex=ax1)
    ax3 = f.add_subplot(gs[2], sharex=ax1)
    ax4 = f.add_subplot(gs[3], sharex=ax1)
    ax = [ax1, ax2, ax3, ax4]

    bins = np.arange(-1000, 4000, binwidth)
    cats = ["insp", "exp", "tonic"]
    for ii, cat in enumerate(cats):
        _dum = merged.query(f"category == '{cat}'")
        a, b = weighted_histogram(_dum["depthsVII"], _dum["deltalogFR"], bins=bins)
        ax[ii].step(a, b, label=cat, color=PHASE_MAP[cat])
        # Fill under the step
        ax[ii].fill_between(a, b, step="pre", alpha=0.3, color=PHASE_MAP[cat])
    # Show the
    sns.histplot(
        merged,
        x="depthsVII",
        hue="category",
        ax=ax[-1],
        element="step",
        stat="count",
        common_norm=False,
        fill=True,
        palette=PHASE_MAP,
        legend=False,
        binwidth=binwidth,
    )
    sns.histplot(
        merged,
        x="depthsVII",
        hue="category",
        ax=ax[-1],
        element="step",
        stat="count",
        common_norm=False,
        fill=False,
        palette=PHASE_MAP,
        legend=False,
        binwidth=binwidth,
    )

    for a in ax:
        a.axvline(0, color="k", linestyle="--")

    # Convert yticks from log2 to linear
    for ii in range(3):
        ax[ii].set_yticks([-2, -1, 0, 1, 2])
        ax[ii].set_yticklabels(["25%", "50%", "100%", "200%", "400%"])
        ax[ii].set_xlim(-1000, 3200)
        ax[ii].set_ylim(-1.5, 1.5)
        ax[ii].set_ylabel("$\\frac{FR_{HB}}{FR_{Control}}$")
        ax[ii].grid("on", ls="--", lw=0.5)
    # ax[0].set_ylabel("$\\frac{FR_{HB}}{FR_{Control}}$")
    ax[-1].set_ylabel("# units")
    ax[-1].set_xlabel(r"AP location ($\mu$m caudal to VII)")
    ax[-1].grid("on", ls="--", lw=0.5, axis="x")

    plt.savefig(f"delta_FR_along_AP.{ext}")


def plot_phase_curve(curve, err="sem", **kwargs):
    m = curve["rate_mean"]
    if err == "std":
        err = curve["rate_std"]
    elif err == "sem":
        err = curve["rate_sem"]
    lb = m - err
    ub = m + err
    line = plt.step(curve["bins"], m, **kwargs)
    plt.fill_between(curve["bins"], lb, ub, alpha=0.3, step="pre", **kwargs)
    return line


def plot_peth(
    all_unit_peth,
    breath_durations,
    eid,
    cids,
    norm=False,
    save_fn=None,
    event="on_sec",
    wd=2,
    ht=0.75,
    ext="pdf",
):
    """
    Plot multiple PETHs for a single recording
    Args:
        all_unit_peth (pd.DataFrame): Dataframe with the PETH data
        breath_durations (pd.DataFrame): Dataframe with the breath durations
        eid (str): Experiment ID
        cids (list): List of cluster IDs
        event (str): Event to plot
        wd (float): Width of the figure
        ht (float): Height of the figure  for each unit
        ext (str): File extension to save fig to
    """

    # Unpack data
    if isinstance(cids, int):
        cids = [cids]

    try:
        cbm = breath_durations.query("eid == @eid")["control"].values[0]
        hbm = breath_durations.query("eid == @eid")["hb"].values[0]
    except:
        cbm = np.nan
        hbm = np.nan
    df = all_unit_peth.query(f"eid == @eid & event == @event")
    df = df.query(f"cid in {cids}")
    if norm:
        y = "mean_norm"
        ymin = "lb_norm"
        ymax = "ub_norm"
        ylims = (-1, None)
        ytick = 5
        ylabel = r"Firing rate ($\sigma$)"
    else:
        y = "mean"
        ymin = "lb"
        ymax = "ub"
        ylims = (-2, None)
        ytick = 20
        ylabel = "Firing rate (sp/s)"
    p = (
        so.Plot(data=df, x="t", y=y, color="condition", ymin=ymin, ymax=ymax)
        .facet(row="cid", order=cids)
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), legend=False)
        .scale(color=CONDITION_COLORS, y=so.Continuous().tick(every=ytick))
        .limit(x=(-0.25, 0.5), y=ylims)
        .share(y=False)
        .layout(size=(wd, ht * len(cids)))
        .label(x="Time (s)", y="", title="")
    ).plot()
    p._figure.supylabel(ylabel, fontsize=8, x=0.1)
    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        ax.axvline(cbm, color=CONDITION_COLORS["control"], ls="--")
        ax.axvline(hbm, color=CONDITION_COLORS["hb"], ls="--")
    if save_fn is None:
        if norm:
            save_fn = f"control_vs_hb_multiple_peths_norm.{ext}"
        else:
            save_fn = f"control_vs_hb_multiple_peths.{ext}"
    p.save(save_fn)


def plot_population_peth_all(
    all_unit_peth, breath_durations, fs=(3, 3), ext="pdf", norm=False
):
    order_dict = {"row": ["insp", "exp", "tonic"], "col": ["on_sec", "off_sec"]}
    control_breath_durations = breath_durations["control"]
    cbm = control_breath_durations.mean()
    cblb = cbm - control_breath_durations.sem()
    cbub = cbm + control_breath_durations.sem()

    hb_breath_durations = breath_durations["hb"]
    hbm = hb_breath_durations.mean()
    hblb = hbm - hb_breath_durations.sem()
    hbub = hbm + hb_breath_durations.sem()

    y = "mean_norm" if norm else "mean"
    save_fn = (
        f"population_peth_all_norm.{ext}" if norm else f"population_peth_all.{ext}"
    )
    ylabel = r"Firing rate ($\sigma$)" if norm else "Firing rate (sp/s)"

    p = (
        so.Plot(data=all_unit_peth, x="t", y=y, color="condition")
        .facet(row="category", col="event", order=order_dict)
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .layout(size=fs)
        .scale(color=CONDITION_COLORS, y=so.Continuous().tick(count=3))
        .limit(x=(-0.25, 0.5), y=(0, None))
        .share(y="row")
    ).plot()
    for ii, ax in enumerate(p._figure.axes):
        event, category = ax.get_title().split(" | ")
        ax.axvline(0, color="k", ls="--")
        if ii < 2:
            if event == "on_sec":
                ax.set_title("Breath onset")
            else:
                ax.set_title("Breath offset")
        else:
            ax.set_title("")
        ax.set_xlabel("")

        if ii % 2 == 0:
            ax.axvline(cbm, color=CONDITION_COLORS["control"], ls="--")
            ax.fill_between(
                [cblb, cbub],
                [0, 0],
                [100, 100],
                color=CONDITION_COLORS["control"],
                alpha=0.3,
            )
            ax.axvline(hbm, color=CONDITION_COLORS["hb"], ls="--")
            ax.fill_between(
                [hblb, hbub],
                [0, 0],
                [100, 100],
                color=CONDITION_COLORS["hb"],
                alpha=0.3,
            )
        else:
            ax.axvline(-cbm, color=CONDITION_COLORS["control"], ls="--")
            ax.fill_between(
                [-cbub, -cblb],
                [0, 0],
                [100, 100],
                color=CONDITION_COLORS["control"],
                alpha=0.3,
            )
            ax.axvline(-hbm, color=CONDITION_COLORS["hb"], ls="--")
            ax.fill_between(
                [-hbub, -hblb],
                [0, 0],
                [100, 100],
                color=CONDITION_COLORS["hb"],
                alpha=0.3,
            )
        ax.set_ylabel(UNIT_TYPES[category], color=PHASE_MAP[category])
    p._figure.supylabel(ylabel, fontsize=8, x=-0.05, color=plt.rcParams["text.color"])
    p._figure.supxlabel("Time (s)", fontsize=8, y=0.1, color=plt.rcParams["text.color"])
    p.save(save_fn)


def plot_population_peth_rec(
    all_unit_peth, breath_durations, eid, event="on_sec", fs=(2, 3), ext="pdf"
):
    df = all_unit_peth.query(f"eid == '{eid}'")
    try:
        control_breath_duration = breath_durations.query("eid == @eid")[
            "control"
        ].values[0]
        hb_breath_duration = breath_durations.query("eid == @eid")["hb"].values[0]
    except Exception as e:
        control_breath_duration = np.nan
        hb_breath_duration = np.nan

    p = (
        so.Plot(data=df.query(f"event==@event"), x="t", y="mean", color="condition")
        .facet(row="category", order=["insp", "exp", "tonic"])
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .layout(size=fs)
        .scale(color=CONDITION_COLORS, y=so.Continuous().tick(count=3))
        .limit(x=(-0.25, 0.5), y=(0, None))
        .share(y=False)
    ).plot()
    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(UNIT_TYPES[ax.get_title()], color=PHASE_MAP[ax.get_title()])
        ax.set_title("")
        ax.axvline(control_breath_duration, color=CONDITION_COLORS["control"], ls="--")
        ax.axvline(hb_breath_duration, color=CONDITION_COLORS["hb"], ls="--")
    p._figure.supylabel(
        "Firing rate (sp/s)", fontsize=8, x=0.1, color=plt.rcParams["text.color"]
    )

    prefix = get_prefix(one, eid)
    save_fn = POPULATION_FIGURES.joinpath(f"population_peth_{prefix}.{ext}")
    p.save(save_fn)


def plot_PCA_PETH(all_pca_peth, eid=None, event=None, fs=(5, 2), ext="pdf"):
    df = all_pca_peth.query(f"eid == '{eid}'")

    p = (
        so.Plot(data=df, x="t", y="mean", color="condition", ymin="lb", ymax="ub")
        .facet(col="event", row="pc")
        .facet(row="pc")
        .add(so.Line(linewidth=0.25), legend=False)
        .add(so.Band(), legend=False)
        .layout(size=fs)
        .scale(
            color=CONDITION_COLORS,
            y=so.Continuous().tick(count=3).label(like="{x:.1f}"),
        )
        .limit(x=(-0.25, 1))
        .label(x="Time (s)", y="", title="")
    ).plot()
    p._figure.supylabel("Projection (a.u.)", fontsize=8, x=0.1)
    for ii, ax in enumerate(p._figure.axes):
        ax.axvline(0, color="k", ls="--")
        ax.set_ylabel(f"PC {ii + 1}")
    p
    fn = PC_FIGURES.joinpath(f"PCA_PETH_{eid}.{ext}")
    p.save(fn)


def plot_breath_durations(breath_durations, fs=(1.75, 2), ext="pdf"):
    data = breath_durations.melt(
        "eid", value_name="Inspiration duration", var_name="Condition"
    )
    data["Inspiration duration"] = data["Inspiration duration"] * 1000
    p = (
        so.Plot(
            data=data,
            x="Condition",
            y="Inspiration duration",
            color="Condition",
        )
        .add(so.Bar(width=0.5), so.Agg(), legend=False)
        .add(so.Dot(pointsize=3), group="eid", legend=False)
        .add(so.Line(color="k"), group="eid", legend=False)
        .layout(size=fs)
        .limit(y=(0, 250))
        .scale(y=so.Continuous().tick(at=[0, 125, 250]), color=CONDITION_COLORS)
    ).plot()
    ax = p._figure.axes[0]

    for ii, r in breath_durations.iterrows():
        ax.plot([0, 1], [r["control"] * 1000, r["hb"] * 1000], color="k", alpha=0.75)

    ax.set_xticklabels(["Control", "Hering-Breuer"])
    ax.set_ylabel("Inspiration duration (ms)")
    ax.set_xlabel("")

    p.save(f"breath_durations.{ext}")


def plot_pca_speed_by_phase(all_pca_speeds, eid, nbins=25, fs=(2, 2), ext="pdf"):
    df = all_pca_speeds.query(f"eid == '{eid}'")

    handles = []
    f, ax = plt.subplots(figsize=fs)
    for condition in ["control", "hb"]:
        _dat = df.query(f"condition == '{condition}'")
        t = _dat["tscale"]
        m = _dat["speed"]
        lb = m - _dat["err"]
        ub = m + _dat["err"]
        color = CONDITION_COLORS[condition]
        (_ll,) = ax.plot(t, m, color=color)
        ax.fill_between(t, lb, ub, alpha=0.3, color=color)
        handles.append(_ll)

    clean_linear_radial_axis(ax)
    ax.set_xlabel("Phase (rads)")
    ax.set_ylabel("Projection speed (a.u.)")
    ax.legend(handles, ["Control", "Hering-Breuer"])
    ax.axvline(0, color="k", ls="--")

    fn = PC_FIGURES.joinpath(f"PCA_speed_{eid}.{ext}")
    plt.savefig(fn)


def plot_all_rec_pca_speeds_by_phase(all_pca_speeds, fs=(2, 2), ext="pdf"):
    p = (
        so.Plot(data=all_pca_speeds, x="tscale", y="speed", color="condition")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .layout(size=fs)
        .scale(color=CONDITION_COLORS)
    ).plot()
    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        clean_linear_radial_axis(ax)
        ax.set_xlabel("Phase (rads)")


def plot_all_pc_speed_PETH(all_pca_speed_peth, fs=(2, 3), ext="pdf"):
    p = (
        so.Plot(data=all_pca_speed_peth, x="t", y="mean", color="condition")
        .facet(row="event")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(alpha=0.3), so.Est(errorbar="se"), legend=False)
        .layout(size=fs)
        .scale(
            color=CONDITION_COLORS,
            y=so.Continuous().tick(count=3).label(like="{x:.2f}"),
        )
        .limit(y=(0, None), x=(-0.25, 0.5))
    ).plot()
    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        if ax.get_title() == "on_sec":
            ax.set_title("Breath onset")
        else:
            ax.set_title("Breath  offset")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("")
    p._figure.supylabel("Projection speed (a.u.)",x=0.1,fontsize='medium')
    p.save(f"PCA_speed_PETH_all.{ext}")


def plot_pca_phase_curve(DF, eid=None, fs=(2, 3), ext="pdf"):
    if eid is None:
        df = DF
    else:
        df = DF.query(f"eid == '{eid}'")
    p = (
        so.Plot(data=df, x="Phase (rads.)", y="val", color="condition")
        .facet(row="pc")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .layout(size=(2, 3))
        .scale(color=CONDITION_COLORS)
        .limit(x=(-np.pi, np.pi))
        .share(y=False)
    ).plot()

    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        clean_linear_radial_axis(ax)
        ax.set_xlabel("Phase (rads)")
        ax.set_ylabel(f"PC{ax.get_title()}")
        ax.set_title("")

    p._figure.supylabel("Projection (a.u.)", fontsize=8, x=0.1)

    # Save
    if eid is None:
        p.save(f"PCA_phase_curves_all.{ext}")
    else:
        save_fn = PC_FIGURES.joinpath(f"PCA_phase_curves_{eid}.{ext}")
        p.save(save_fn)


def plot_all_unit_phase_curves(all_phase_curves, fs=(2, 3), ext="pdf"):
    conditions = ["control", "hb"]
    dat = all_phase_curves.query("condition in @conditions")
    p = (
        so.Plot(data=dat, x="phase", y="rate", color="condition")
        .facet(row="category", order=["insp", "exp", "tonic"])
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .layout(size=fs)
        .scale(color=CONDITION_COLORS, y=so.Continuous().tick(every=20))
        .limit(x=(-np.pi, np.pi))
        .label(x="Phase (rads)")
        .share(y=False)
    ).plot()

    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        clean_linear_radial_axis(ax)
        category = ax.get_title()
        ax.set_ylabel(UNIT_TYPES[category], color=PHASE_MAP[category])
        ax.set_title("")
    p._figure.supylabel("Firing rate (sp/s)", fontsize=8, x=0.1)
    p.save(f"all_units_phase_curve.{ext}")


def plot_example_spiketrains(
    rec, n_per_phase=10, pre_time=3, post_time=5, fs=(4, 2), lw=0.5, ext="pdf"
):
    """Plot example spike trains for the top n_per_phase units in each phase

    Args:
        rec (Recording): Recording object
        n_per_phase (int, optional): Number of units to plot for each phase. Defaults to 10.
        pre_time (int, optional): Time before stimulus to plot. Defaults to 3.
        post_time (int, optional): Time after stimulus to plot. Defaults to 5.

    Returns:
        matplotlib.axes: Axes object

    """
    cluster_ids = []
    colors = []
    phase_map = PHASE_MAP
    phase_map["tonic"] = plt.rcParams["text.color"]

    # Get Unit ids
    for phase in ["insp", "exp", "tonic"]:
        _unit_ids = rec.get_phasic_unit_ids(phase)
        n_units = np.min([n_per_phase, len(_unit_ids)])
        if phase == "tonic":
            cluster_ids.append(_unit_ids[-n_units:])
        else:
            cluster_ids.append(_unit_ids[:n_units])
        colors += [PHASE_MAP[phase]] * n_units
    n_units_total = len(colors)

    cluster_ids = np.concatenate(cluster_ids)
    stims, intervals = rec.get_HB_stims(duration=5)
    t0, tf = stims[0, :]
    win_start = t0 - pre_time
    win_stop = tf + post_time
    raster = []
    for ii, cluster_id in enumerate(cluster_ids):
        spike_times = rec.spikes.times[rec.spikes.clusters == cluster_id]
        sub_spike_times = (
            spike_times[(spike_times > win_start) & (spike_times < win_stop)] - t0
        )
        raster.append(sub_spike_times)

    f = plt.figure(figsize=fs)
    ax = f.add_subplot(111)
    ax.eventplot(raster, linelengths=1, lineoffsets=1.1, colors=colors, linewidths=lw)
    ax.axvspan(0, 5, color=CONDITION_COLORS["hb"], alpha=0.2)
    s0, sf = rec.physiology.times.searchsorted([win_start, win_stop])
    dia_sub = rec.physiology.dia.copy()[s0:sf]
    dia_sub /= dia_sub.max()
    dia_sub *= n_units_total * 0.3
    dia_sub += n_units_total * 1.15
    ax.plot(rec.physiology.times[s0:sf] - t0, dia_sub, c=plt.rcParams["text.color"])

    for ii, phase in enumerate(["insp", "exp", "tonic"]):
        ax.text(
            -pre_time,
            n_per_phase * (ii + 0.5),
            phase.capitalize(),
            color=PHASE_MAP[phase],
            fontsize=6,
            rotation=90,
            ha="right",
            va="center",
        )

    ax.text(
        -pre_time,
        dia_sub.mean(),
        r"$\int{}$ Dia.",
        color=plt.rcParams["text.color"],
        fontsize=6,
        rotation=90,
        ha="right",
        va="bottom",
    )

    ax.set_xlim(-pre_time, post_time + 5)
    ax.set_ylim(0, dia_sub.max())
    replace_timeaxis_with_scalebar(ax, pad=0.05)
    # remove yaxis
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)

    save_fn = f"example_spiketrains_HB_{rec.prefix}.{ext}"
    plt.savefig(save_fn)


def plot_firing_rate_in_phase(all_firing_rate_in_phase, fs=(3.5, 2), ext="pdf"):
    df = all_firing_rate_in_phase.query("category != 'tonic'").copy()
    TINY = 1e-2
    delta = (df["hb"] + TINY) / (df["control"] + TINY)
    logdelta = np.log2(delta)
    df["delta"] = delta
    df["logdelta"] = logdelta

    fig, axs = plt.subplots(
        1,
        2,
        figsize=fs,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    # Scatter plot
    p1 = (
        (
            so.Plot(data=df, x="control", y="hb", color="category")
            .add(so.Dot(), alpha="respMod", pointsize="respMod")
            .scale(
                color=PHASE_MAP,
                pointsize=so.Continuous((0.1, 2), norm=(0, 1))
                .tick(count=3)
                .label(like="{x:.2f}"),
                alpha=so.Continuous((0, 0.5), norm=(0, 1))
                .tick(count=3)
                .label(like="{x:.2f}"),
                x="log",
                y="log",
            )
            .limit(y=(0, None), x=(0, None))
            .label(x="Control (sp/s)", y="HB (sp/s)", title="")
        )
        .on(axs[0])
        .plot()
    )
    axs[0].set_aspect("equal")
    axs[0].plot(axs[0].get_xlim(), axs[0].get_ylim(), color="k", ls="--")

    # Cumulative histogram
    p2 = (
        (
            so.Plot(data=df, x="logdelta", color="category")
            .add(
                so.Line(),
                so.Hist(cumulative=True, stat="percent", common_norm=False),
                legend=False,
            )
            .scale(
                color=PHASE_MAP,
                x=so.Continuous().tick(count=3).label(like="{x:.0f}"),
                y=so.Continuous().tick(count=3).label(like="{x:.0f}%"),
            )
            .limit(x=(-2, 2), y=(0, 100))
            .label(x="Log2(HB/Control)", y="Cumulative %", title="")
        )
        .on(axs[1])
        .plot()
    )
    # axs[1].set_aspect((axs[0].get_xlim()[1] - axs[0].get_xlim()[0]) / (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]))
    axs[1].axvline(0, color="k", ls="--")
    axs[1].axhline(50, color="k", ls="--")

    axs[1].set_xticks([-2, 0, 2])
    axs[1].set_xticklabels(["0.25x", "1x", "4x"])
    axs[1].set_xlabel(r"$\frac{FR_{HB}}{FR_{Control}}$")

    fig.suptitle("Phase-congruent firing rates")
    plt.savefig(f"firing_rate_in_phase.{ext}")


def plot_stim_aligned_speed_hb_only(all_pca_speed_stim_aligned, fs=(1.5, 1.5), ext="pdf"):
    df = all_pca_speed_stim_aligned.copy().query('stimulus_type == "hb"')
    df["t"] = df["t"].round(2)
    tbins = np.arange(-2, 7.1, 0.1)
    t_sub = tbins[np.digitize(df["t"], tbins) - 1]
    df["t_sub"] = t_sub
    df = pd.pivot_table(df, index=["t_sub"], values="speed", columns="eid")
    mm = df.mean(axis=1)
    lb = df.std(axis=1) / np.sqrt(df.shape[1])
    ub = df.std(axis=1) / np.sqrt(df.shape[1])

    f = plt.figure(figsize=fs)
    ax = f.add_subplot(111)

    # Normalize mm values for colormap
    norm = Normalize(vmin=mm.min(), vmax=mm.max())
    cmap = cm.get_cmap("viridis")

    # Create segments for LineCollection
    points = np.array([df.index, mm]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(mm)
    lc.set_linewidth(0.5)
    ax.add_collection(lc)

    ax.fill_between(
        df.index, mm - lb, mm + ub, alpha=0.1, color=plt.rcParams["text.color"]
    )
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylim((mm - lb).min(), (mm + ub).max())
    ax.axvline(0, color="k", ls="--")
    ax.axvspan(0, 5, color=CONDITION_COLORS["hb"], alpha=0.2)
    ax.set_ylabel("Projection speed (a.u.)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, None)
    plt.savefig(f"stim_aligned_speed_hb_only.{ext}")


def plot_stim_aligned_speed(all_pca_speed_stim_aligned, fs=(2, 1.75), ext="pdf"):
    df = all_pca_speed_stim_aligned.copy()
    df["t"] = df["t"].round(2)

    tbins = np.arange(-2, 7.1, 0.1)
    t_sub = tbins[np.digitize(df["t"], tbins) - 1]
    df["t_sub"] = t_sub

    df = df.groupby(['t_sub','eid','genotype','stimulus_type']).mean().reset_index()
    # Replace stimulus_type with genotype where stimulus_type is 'hold'
    df.loc[df['stimulus_type'] == 'hold', 'stimulus_type'] = df['genotype']
    colormap = GENOTYPE_COLORS.copy()
    colormap['hb'] = CONDITION_COLORS['hb']

    p = (
        so.Plot(data=df, x="t_sub", y="speed", color="stimulus_type")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .layout(size=fs)
        .scale(color=colormap, y=so.Continuous().tick(every=0.1))
        .limit(x=(-2, 7), y=(0, None))
        .label(x="Time (s)", y="Projection speed (a.u.)", title="")
    ).plot()
    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        ax.hlines(ax.get_ylim()[1]*0.95,0,2, color='k', lw=1)
        ax.text(0, ax.get_ylim()[1]*0.95, "Laser", ha='left', va='bottom', fontsize=6, color='k')
        ax.hlines(ax.get_ylim()[1]*0.85,0,5, color=HB_MAP[5], lw=1)
        ax.text(0, ax.get_ylim()[1]*0.85, "Hering-Breuer", ha='left', va='bottom', fontsize=6, color=HB_MAP[5])
        ax.set_xlabel("Time (s)")
    p.save(f"stim_aligned_pca_speed_hb_and_hold.{ext}")


def plot_stim_aligned_speed_summary(all_pca_speed_stim_aligned, fs=(2.5, 1.75), ext="pdf"):
    df = all_pca_speed_stim_aligned.copy()
    df.loc[df['stimulus_type'] == 'hold', 'stimulus_type'] = df['genotype']
    colormap = GENOTYPE_COLORS.copy()
    colormap['hb'] = CONDITION_COLORS['hb']
    stim = df.query("t>1 & t<2").reset_index()
    ctrl = df.query("t>-2 & t<-1").reset_index()

    stim = pd.pivot_table(stim, index="eid", values="speed",columns=['stimulus_type']).melt(value_name='speed',ignore_index=False).reset_index()
    ctrl = pd.pivot_table(ctrl, index="eid", values="speed",columns=['stimulus_type']).melt(value_name='speed',ignore_index=False).reset_index()

    df = pd.merge(stim, ctrl, on=['eid','stimulus_type'], suffixes=('_stim', '_ctrl'))
    df['delta_speed'] = (df['speed_stim'] - df['speed_ctrl'])/df['speed_ctrl']*100
    mapper = all_pca_speed_stim_aligned[['eid','genotype']].drop_duplicates().reset_index(drop=True)

    df = df.merge(mapper, on="eid", how='inner')
    
    p = (
        so.Plot(data=df, y="stimulus_type", x="delta_speed",color='stimulus_type')
        .add(so.Bar(width=0.5), so.Agg(), legend=False)
        .add(so.Range(color='k'), so.Est('mean','se'), legend=False)
        .add(so.Dots(pointsize=2),so.Jitter(),so.Shift(y=0.35),legend=False,color='genotype')
        .scale(color=colormap, x=so.Continuous().tick(every=50))
        .layout(size=fs)
        .label(x="Speed change (% diff.)", y="")
        
    ).plot()
    ax = p._figure.axes[0]
    ax.set_yticks(ax.get_yticks())
    yticks = ax.get_yticklabels()
    labels = GENOTYPE_LABELS.copy()
    labels['hb'] = 'Hering-Breuer'
    ax.set_yticklabels([labels[x.get_text()]for x in yticks])
    ax.axvline(0, color="k", ls="--")
    p.save(f"stim_aligned_speed_summary.{ext}")
    
    anova = pg.anova(data=df, dv='delta_speed', between='stimulus_type')
    pairwise = pg.pairwise_tukey(data=df, dv='delta_speed', between='stimulus_type')
    anova['sigstar'] = sig2star(anova['p_unc'].values)
    pairwise['sigstar'] = [sig2star(x) for x in pairwise['p_tukey'].values]
    pairwise.to_csv("stim_aligned_speed_summary_pairwise.csv")
    anova.to_csv("stim_aligned_speed_summary_anova.csv")


def plot_stim_aligned_speed_summary_hb(all_pca_speed_stim_aligned, sessions_to_include,fs=(1.35, 1.85), ext="pdf"):
    df = all_pca_speed_stim_aligned.copy()
    hb = df.query("t>1 & t<2")
    ctrl = df.query("t>-2 & t<1")

    ctrl = pd.pivot_table(ctrl, index="eid", values="speed")
    hb = pd.pivot_table(hb, index="eid", values="speed")
    ctrl["condition"] = "control"
    hb["condition"] = "hb"
    df = pd.concat([ctrl, hb]).reset_index()
    df = df.merge(sessions_to_include[['eid','genotype']], on="eid")

    p = (
        so.Plot(data=df, x="condition", y="speed")
        .add(so.Bar(width=0.5,color='k'), so.Agg(), alpha="condition", legend=False)
        .add(so.Range(color='k'), so.Est('mean','se'), legend=False)
        .add(so.Line(alpha=0.2),group="eid", legend=False,color='genotype')
        .layout(size=fs)
        .limit(y=(0, None))
        .label(x="", y="Projection speed (a.u.)", title="")
        .scale(
            y=so.Continuous().tick(every=0.1).label(like="{x:.2f}"),
            alpha = [0.25, 1],
            x = so.Nominal(['control', 'hb']),
        )
    ).plot()
    ax = p._figure.axes[0]
    ax.set_xticklabels(["Pre-stim", "Hering-Breuer"],rotation=45, ha='right',va='top')

    p.save(f"stim_aligned_speed_summary_hb_only.{ext}")
    x = df.query("condition == 'control'")["speed"]
    y = df.query("condition == 'hb'")["speed"]
    paired_ttest = pg.ttest(x,y,paired=True)
    paired_ttest['reject_null'] = paired_ttest['p_val'] < 0.05
    paired_ttest.to_csv("stim_aligned_speed_summary_stats_hb_only.csv")


def plot_stim_aligned_pca(all_pca_stim_aligned, fs=(2, 3), ext="pdf"):
    """NOT USEFUL

    Args:
        all_pca_stim_aligned (_type_): _description_
        fs (tuple, optional): _description_. Defaults to (2, 3).
        ext (str, optional): _description_. Defaults to "pdf".
    """
    df = all_pca_stim_aligned.pivot_table(
        index=["pc", "t", "eid"], values="mean"
    ).reset_index()
    p = (
        so.Plot(data=df, x="t", y="mean")
        .facet(row="pc")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .layout(size=fs)
        .scale(
            color=CONDITION_COLORS,
            y=so.Continuous().tick(count=3).label(like="{x:.2f}"),
        )
        .limit(x=(-5, 8))
        .label(x="Time (s)", y="Projection (a.u.)", title="")
    ).plot()
    p
    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls="--")
        ax.axvspan(0, 5, color=CONDITION_COLORS["hb"], alpha=0.2)
    p
    p.save(f"stim_aligned_pca.{ext}")


def plot_preI_rates(
    all_preI_spikerates,
    cluster_features,
    rec=None,
    pre_time=0.05,
    fs=(2, 1.75),
    ext="pdf",
):
    # Get mean breath_shape
    onsets = rec.breaths.on_sec[:100]
    breath_eta = get_eta(
        rec.physiology.dia, rec.physiology.times, onsets, pre_win=0.2, post_win=0.4
    )

    df = all_preI_spikerates.copy()
    df = pd.pivot_table(df, index=["uuid", "category", "condition"], values="preI_fr")
    df = df.reset_index()
    df["condition"] = df["condition"].map(LABELS)
    # Normalize
    df["mean_fr"] = pd.merge(
        df, cluster_features["firing_rate"], right_index=True, left_on="uuid"
    )["firing_rate"]
    df["preI_fr_norm"] = df["preI_fr"] / df["mean_fr"] * 100

    p = (
        so.Plot(data=df, x="condition", y="preI_fr", color="category")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Range(), so.Est(), legend=False)
        .scale(color=PHASE_MAP, y=so.Continuous().tick(count=4).label(like="{x:.0f}"))
        .limit(y=(0, None))
        .layout(size=fs)
        .label(x="", y="50ms pre-I firing rate\n(sp/s)", title="")
    ).plot()
    p.save(f"preI_rates.{ext}")

    p = (
        so.Plot(data=df, x="condition", y="preI_fr_norm", color="category")
        .add(so.Line(), so.Agg(), so.Shift(), legend=False)
        .add(so.Range(), so.Est(), so.Shift(), legend=False)
        .scale(
            color=PHASE_MAP,
            y=so.Continuous().tick(at=[50, 75, 100]).label(like="{x:.0f}%"),
        )
        .layout(size=fs)
        .limit(y=(40, 100))
        .label(x="", y="50ms pre-I firing rate\n(% of mean)", title="")
    ).plot()
    f = p._figure
    # Add axis inset in top right
    ax_inset = f.add_axes([0.8, 0.8, 0.2, 0.2])
    ax_inset.plot(breath_eta["t"], breath_eta["mean"], color=plt.rcParams["text.color"])
    ax_inset.axvspan(-pre_time, 0, color="k", alpha=0.1)
    ax_inset.axis("off")

    p.save(f"preI_rates_norm.{ext}")

    cats = ["insp", "exp", "tonic"]
    stats_df = pd.DataFrame()
    for cat in cats:
        x = df.query(f"category == @cat &  condition=='Control'")["preI_fr"]
        y = df.query(f"category == @cat &  condition=='Hering-Breuer'")["preI_fr"]
        t, p = ttest_rel(x, y)
        reject_null = p < 0.05
        _stats_df = pd.DataFrame(
            {"t": t, "p": p, "reject_null": reject_null}, index=[cat]
        )
        stats_df = pd.concat([stats_df, _stats_df])
    stats_df.to_csv("preI_rates_stats.csv")


def plot_phase_fr_ratio(all_unit_peth, fs=(2.5, 2), ext="pdf"):
    df = all_unit_peth.query("event == 'on_sec'").copy()
    df = pd.pivot_table(df, index="t", columns=["category", "condition"], values="mean")
    # df.columns = df.columns.droplevel()
    hb_EI_ratio = df["exp"]["hb"] / df["insp"]["hb"]
    control_EI_ratio = df["exp"]["control"] / df["insp"]["control"]

    hb_IE_ratio = df["insp"]["hb"] / df["exp"]["hb"]
    control_IE_ratio = df["insp"]["control"] / df["exp"]["control"]

    f, axs = plt.subplots(figsize=fs, nrows=2, sharex=True, constrained_layout=True)
    axs[0].plot(hb_EI_ratio, label="HB", color=CONDITION_COLORS["hb"])
    axs[0].plot(control_EI_ratio, label="Control", color=CONDITION_COLORS["control"])
    axs[0].set_ylabel("Exp/insp ratio")
    axs[0].axvline(0, color="k", ls="--")
    # axs[0].legend(['Hering-Breuer', 'Control'],loc='upper right')
    axs[0].set_ylim(0, None)

    axs[1].plot(hb_IE_ratio, label="HB", color=CONDITION_COLORS["hb"])
    axs[1].plot(control_IE_ratio, label="Control", color=CONDITION_COLORS["control"])
    axs[1].axvline(0, color="k", ls="--")
    axs[1].set_ylabel("Insp/exp ratio")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(0, None)
    axs[1].set_xlim(-0.25, 0.5)
    # Add inset zooming in on x between -0.1 and 0.0 and y between 0 and 1
    inset_ax = axs[1].inset_axes([0.7, 0.4, 0.4, 0.6])  # [x, y, width, height]
    inset_ax.plot(hb_IE_ratio, label="HB", color=CONDITION_COLORS["hb"])
    inset_ax.plot(control_IE_ratio, label="Control", color=CONDITION_COLORS["control"])
    inset_ax.axvline(0, color="k", ls="--")
    inset_ax.set_xlim(-0.1, 0.0)
    inset_ax.set_ylim(0.25, 0.75)
    axs[1].indicate_inset_zoom(inset_ax, edgecolor="black")

    # Align y-labels
    f.align_ylabels(axs)
    plt.savefig(f"phase_firing_ratios.{ext}")


def plot_phase_transition_fr_ratios(all_unit_peth, fs=(2, 2), ext="pdf"):
    df = all_unit_peth.query("event == 'on_sec'").copy()
    df = pd.pivot_table(
        df, index=["t", "eid"], columns=["category", "condition"], values="mean"
    )
    transitions = pd.DataFrame()
    for tt in [-0.01, 0.01]:
        transition_EI_ratio_control = (
            df.loc[tt]["exp"]["control"] / df.loc[tt]["insp"]["control"]
        )
        transition_EI_ratio_hb = df.loc[tt]["exp"]["hb"] / df.loc[tt]["insp"]["hb"]
        _df = pd.DataFrame()
        _df["control"] = transition_EI_ratio_control
        _df["hb"] = transition_EI_ratio_hb
        _df["t"] = tt
        _df["eid"] = transition_EI_ratio_control.index
        transitions = pd.concat([transitions, _df])
    transitions = transitions.melt(
        id_vars=["t", "eid"], value_name="EI_ratio", var_name="condition"
    )

    p = (
        so.Plot(data=transitions, x="t", y="EI_ratio", color="condition")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .add(so.Line(alpha=0.1), group="eid")
        .add(so.Dot(pointsize=2), so.Jitter(), group="eid")
        .scale(color=CONDITION_COLORS)
        .label(x="Inspiration time (s)", y="Exp/Insp ratio", title="")
        .layout(size=fs)
        .scale(x=so.Continuous().tick(at=[-0.01, 0, 0.01]))
        .limit(x=(-0.015, 0.015), y=(0, None))
    ).plot()

    f = p._figure
    f.legends[0].set_title("")
    for t in f.legends[0].texts:
        t.set_text(LABELS[t.get_text()])
    f.legends[0].set_bbox_to_anchor((0.75, 0.75))
    ax = f.axes[0]
    ax.axvline(0, color="k", ls="--")
    ax.axhline(1, color="k", ls="--")
    p
    p.save(f"phase_firing_ratios_transitions.{ext}")


def plot_distance_metrics_hb(fs=(1.5,2)):
    "Plots the mean distance during 0.5-1.5 in hb and at random points (see compute_pca_perturbations intervals)"
    df = pd.read_csv('../low_d/distance_results_full.csv')
    df = df.query('condition !="50ms" and ndims==2')
    df = pd.pivot_table(df,index=['eid','genotype','condition'],values=['exp_distance','dispersion']).reset_index()

    colors = GENOTYPE_COLORS.copy()
    colors['hb'] = CONDITION_COLORS['hb']
    colors['random'] = CONDITION_COLORS['control']
    labels = {
        'hb': 'HB',
        'random': 'Random',
        'exp_distance': 'Distance to I-off (a.u.)',
        'dispersion': 'Dispersion (a.u.)',
    }
    for depvar in ['exp_distance','dispersion']:
        p = (
            so.Plot(data=df, x="condition", y=depvar)
            .add(so.Dots(pointsize=2),so.Jitter(),so.Shift(0.35),legend=False,color='genotype')
            .add(so.Lines(alpha=0.1),so.Shift(0.35),group="eid", legend=False,color='genotype')
            .add(so.Bar(width=0.5,), so.Agg(), legend=False,color='condition',alpha='condition')
            .add(so.Range(color='k'), so.Est('mean','se'), legend=False)
            .scale(color=colors, x=so.Nominal(order=['random', 'hb']),y=so.Continuous().tick(every=0.5),alpha=[0.25, 1])
            .layout(size=fs)
            .label(x="", y=depvar, title="")
            .limit(y=(0, None))
        ).plot()
        ax = p._figure.axes[0]
        ax.set_xticklabels([labels[x.get_text()] for x in ax.get_xticklabels()])
        ax.set_ylabel(labels[depvar])
        p.save(f"HB_distance_metrics_{depvar}.pdf")
        
        anova = pg.mixed_anova(data=df,dv=depvar,between='genotype',within='condition',subject='eid')
        anova['sigstar'] = [sig2star(x) for x in anova['p_unc'].values]
        anova['dv'] = depvar
        anova.to_csv(f"HB_distance_metrics_{depvar}_anova.csv")

        
# ------------------ #
# Run functions
# ------------------ #
def run_compute():
    run_all_eids()

    # Breath Durations
    breath_durations, breath_durations_melt = get_breath_durations()
    breath_durations.to_csv("breath_durations.csv")
    breath_durations_melt.to_csv("breath_durations_melt.csv")


def run_plot():
    # ------------------ #
    #  Load data
    # ------------------ #

    # Load mean firing rates
    pct_diff, mean_frs, cluster_features = load_data()
    merged = pct_diff.reset_index().set_index("uuids")
    merged = merged.join(cluster_features["depthsVII"])
    all_firing_rate_in_phase = pd.read_parquet("all_firing_rate_in_phase.pqt")
    all_preI_spikerates = pd.read_parquet("all_preI_spikerates.pqt")

    # Load sinlge unit peths
    all_unit_peth = pd.read_parquet("all_unit_hb_peth.pqt")
    breath_durations = pd.read_csv("breath_durations.csv")

    # Load single unit phase curves
    all_phase_curves = VLAD_ROOT.joinpath("singlecell").joinpath("all_phase_curves.pqt")
    all_phase_curves = pd.read_parquet(all_phase_curves)

    # Load PCA phase curves
    all_pca_phase_curves = pd.read_parquet("all_pca_phase_curves.pqt")
    all_pca_peth = pd.read_parquet("all_pca_peth.pqt")
    all_pca_stim_aligned = pd.read_parquet("all_pca_stim_aligned.pqt")

    # Load PCA speed data
    all_pca_speed_peth = pd.read_parquet("all_pca_speed_peth.pqt")
    all_pca_speeds_phase = pd.read_parquet("all_pca_speed_phase.pqt")
    all_pca_speed_stim_aligned = pd.read_parquet("all_pca_speed_stim_aligned.pqt")
    all_pca_speed_stim_aligned = all_pca_speed_stim_aligned.merge(sessions_to_include[['eid','genotype']], on="eid")
    all_pca_stim_aligned = pd.read_parquet("all_pca_stim_aligned.pqt")

    # ------------------ #
    # Plot example spiketrains
    eid = EIDS_NEURAL[3]
    rec = Rec(one, eid)
    plot_example_spiketrains(rec, fs=(3, 2.25), lw=0.5, n_per_phase=10)
    rec.plot_pca_sweeps()
    rec.plot_average_dia()

    # ------------------ #
    # Plot mean firing rate data (scatter, CDF, anatomical distribution)
    plot_scatter_control_vs_hb(mean_frs)
    plot_CDF_control_vs_hb(pct_diff)
    plot_delta_FR_along_AP(merged, binwidth=100)
    plot_firing_rate_in_phase(all_firing_rate_in_phase)
    plot_preI_rates(all_preI_spikerates, cluster_features, rec)
    plot_phase_fr_ratio(all_unit_peth)
    plot_phase_transition_fr_ratios(all_unit_peth)

    # Plot multiple peths
    insp = rec.get_phasic_unit_ids("insp")
    exp = rec.get_phasic_unit_ids("exp")
    tonic = rec.get_phasic_unit_ids("tonic")
    cids = [insp[3], exp[1], tonic[1]]
    plot_peth(all_unit_peth, breath_durations, eid, wd=1.75, ht=1, cids=cids)
    plot_peth(all_unit_peth, breath_durations, eid, wd=1.75, ht=1, cids=cids, norm=True)

    # ------------------ #
    # Breath durations
    plot_breath_durations(breath_durations)

    # ------------------ #
    # Plot population phase curves
    plot_all_unit_phase_curves(all_phase_curves)

    # Population PETHs
    for eid in EIDS_NEURAL:
        plot_population_peth_rec(all_unit_peth, breath_durations, eid)
    plot_population_peth_all(all_unit_peth, breath_durations)

    # ------------------ #
    # PCA projection plots
    # PETHs
    for eid in EIDS_NEURAL:
        plot_PCA_PETH(all_pca_peth, eid=eid)

    # By phase
    plot_pca_phase_curve(all_pca_phase_curves)
    for eid in EIDS_NEURAL:
        plot_pca_phase_curve(all_pca_phase_curves, eid=eid)

    plot_stim_aligned_pca(all_pca_stim_aligned)
    # ------------------ #
    # PCA speed plots
    # PETHs
    plot_all_pc_speed_PETH(all_pca_speed_peth)
    plot_stim_aligned_speed_hb_only(all_pca_speed_stim_aligned)
    plot_stim_aligned_speed_summary_hb(all_pca_speed_stim_aligned, sessions_to_include)
    plot_stim_aligned_speed(all_pca_speed_stim_aligned)
    plot_stim_aligned_speed_summary(all_pca_speed_stim_aligned)


    # phase
    for eid in EIDS_NEURAL:
        plot_pca_speed_by_phase(all_pca_speeds_phase, eid)
    plot_all_rec_pca_speeds_by_phase(all_pca_speeds_phase)


    # Distance to I off and dispersion
    plot_distance_metrics_hb()
# --------------- #
# Main function
# --------------- #
@click.command()
@click.option("--compute", "-c", is_flag=True)
def main(compute):
    if compute:
        run_compute()
    else:
        run_plot()


if __name__ == "__main__":
    main()
