"""
May potentially want to use sqrt transform for all scatters. Currently only sqrt for hold-hb, 
rationale is that it is a different comparison, looking for the conditions which changes the FR the least
, and want to compare the correlations, which are better preserved by the sqrt trnsform
"""
import sys

sys.path.append("../")
sys.path.append("VLAD/")
from utils import (
    one,
    EIDS_NEURAL,
    Rec,
    PHASE_MAP,
    GENOTYPE_COLORS,
    GENOTYPE_LABELS,
    QC_QUERY,
    HB_MAP,
    set_style,
    sig2star
)
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.objects as so
import numpy as np
from cibrrig.analysis.singlecell import get_all_phase_curves
from cibrrig.utils.utils import time_to_interval
from cibrrig.plot import (
    plot_peth_and_raster,
    laser_colors,
    replace_timeaxis_with_scalebar,
    trim_yscale_to_lims,
)
from pathlib import Path
import one.alf.io as alfio
from brainbox.population.decode import get_spike_counts_in_bins
import brainbox.singlecell as bbsc
import click
from tqdm import tqdm
import scipy.signal as signal
from scipy.stats import linregress
import scipy.stats
from scipy.stats import ttest_rel
from itertools import product


CONDITION_COLORS = {
    "control": "k",
    "exp": PHASE_MAP["exp"],
    "insp": PHASE_MAP["insp"],
    "hold": laser_colors[473],
    "hb": HB_MAP[5],
}
CONDITION_TITLES = {
    "control": "None",
    "exp": "Expiration",
    "insp": "Inspiration",
    "hold": "Constant (2s)",
    "hb": "Hering-Breuer",
}

GENOTYPE_ORDER = ['vglut2ai32','vgatai32','vgatcre_ntschrmine']
PHASE_ORDER = ['exp','insp','tonic']

set_style()
EXT = ".pdf"

save_fn_phase_curves = Path("all_phase_curves.pqt")
save_fn_firing_rates = Path("all_firing_rates.pqt")

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)


def log2_formatter(x, y):
    """
    Format the x and y axis labels for log scale plots by mapping into log2
    """
    map = {
        -2: "0.25x",
        0: "1x",
        2: "4x",
    }
    return map[x]


def get_breaths_subset(breaths, stims):
    """Finds breaths that fall into a start and end time using the stims dataframe

    Args:
        breaths (AlfBunch): breaths object with variables "on_sec"
        stims (DataFrame): stims dataframe with columns "start_time" and "end_time"
    """
    starts, stops = stims[["start_time", "end_time"]].values.T
    breaths_out = breaths[np.isfinite(time_to_interval(breaths.on_sec, starts, stops))]
    return breaths_out


def compute_conditional_phase_modulation(rec, stims, condition_name, nbins=50):
    """return a dataframe of the phasic response curves of units during stimulus times

    Args:
        rec (Recording): recording object defined in utils. Has breaths attribute
        stims (DataFrame): dataframe that has columns 'start_time' and 'end_time'
        condition_name (str): what condition to label the outgoing dataframe with
        nbins (int, optional): number of bins to subdivide phase into. Defaults to 50.
    """
    breaths = rec.breaths.to_df()
    clusters = rec.clusters.copy()
    spikes = rec.spikes
    cluster_metrics = clusters.pop("metrics")
    clusters.pop("siMetrics", None)
    clusters.pop("waveforms")
    cluster_ids = rec.cluster_ids

    cluster_df = alfio.AlfBunch(
        clusters
    ).to_df()  # Have to map back to an AlfBunch for some reason
    cluster_df["cluster_ids"] = cluster_metrics.cluster_id.values

    breaths_subset = get_breaths_subset(breaths, stims)
    if len(breaths_subset) < 5:
        _log.warning(f"Less than 5 breaths in {condition_name} for {rec.prefix}")
        return pd.DataFrame(columns=["phase", "uuids", "rate", "condition"])
    # Phase
    bins, rate, sems_control, _ = get_all_phase_curves(
        spikes.times, spikes.clusters, cluster_ids, breaths_subset, nbins=nbins
    )
    # Output to dataframe
    df = (
        pd.DataFrame(rate, index=bins, columns=clusters.uuids[cluster_ids])
        .rename_axis("phase")
        .melt(ignore_index=False, var_name="uuids", value_name="rate")
        .reset_index()
        .merge(cluster_df, on="uuids")
    )
    df["condition"] = condition_name
    return df


def compute_phase_curves_all_conditions(rec):
    """Loop through the different stimulation types
    and compute the phase curves for all units in these situations

    Args:
        rec (Recording): Recording object
    """

    conditions = ["control", "exp", "insp", "hold", "hb"]
    all_data = []
    for cond in conditions:
        intervals, stims = rec.get_stims(cond)
        all_data.append(compute_conditional_phase_modulation(rec, stims, cond))
    df = pd.concat(all_data)

    df["genotype"] = rec.genotype
    df["eid"] = rec.eid

    return df


def compute_delta_phase_curves(phase_curves):
    """
    Compute the delta phase curves for each unit in each condition
    compared to the control condition
    """

    pc_pivot = phase_curves.pivot_table(
        index=["uuids", "genotype", "phase", "category"],
        columns="condition",
        values="rate",
        aggfunc="mean",
    ).reset_index()

    # Subtract the control condition from other conditions
    if "control" in pc_pivot.columns:
        for cond in pc_pivot.columns:
            if cond not in ["uuids", "genotype", "phase", "control", "category"]:
                pc_pivot[cond] = pc_pivot[cond] - pc_pivot["control"]

    # Melt into tidy form
    pc_melt = pc_pivot.melt(
        id_vars=["uuids", "genotype", "phase", "category"],
        var_name="condition",
        value_name="delta",
    )
    # Exclude control column since it's now the baseline
    pc_melt = pc_melt[pc_melt.condition != "control"]
    return pc_melt


def _compute_fr_epoch(rec, stims, condition_name):
    """Compute the mean firing rate of all neurons in a given stimulus condition

    Args:
        rec (Recording): recording obejct
        stims (DataFrame): stim dataframe
        condition_name (str): label of this condition
    """
    stim_times = stims[["start_time", "end_time"]].values
    pop_vector, cluster_ids = get_spike_counts_in_bins(
        rec.spikes.times, rec.spikes.clusters, stim_times
    )
    durs = np.diff(stim_times, 1).ravel().astype("f")
    rates = np.divide(pop_vector, durs).mean(1)
    df = pd.DataFrame()
    df["cluster_id"] = cluster_ids
    df["fr"] = rates
    df["condition"] = condition_name
    return df


def compute_mean_fr_epochs(rec):
    """
    Loop through all the stimulus conditions to compute the firing rate for
    Args:
        rec (Recording): recording object
    """
    conditions = ["control", "exp", "insp", "hold", "hb"]
    all_data = []
    for cond in conditions:
        intervals, stims = rec.get_stims(cond)
        all_data.append(_compute_fr_epoch(rec, stims, cond))

    # Assign cluster level information
    df = pd.concat(all_data).reset_index(drop=True)
    clusters = rec.clusters.copy()
    cluster_metrics = clusters.pop("metrics")
    clusters.pop("siMetrics", None)
    clusters.pop("waveforms")
    cluster_df = alfio.AlfBunch(clusters).to_df()
    cluster_df["cluster_id"] = cluster_metrics.cluster_id.values
    df = df.merge(cluster_df, on="cluster_id")

    # Assign session and subject level information
    df["genotype"] = rec.genotype
    df["subject"] = rec.subject
    df["eid"] = rec.eid
    return df


def compute_cluster_level_dataset():
    """Just concatenate all the cluster level information available.
    Does not do any filtering
    """
    all_clusters = pd.DataFrame()
    # _log.info('Computing cluster level data')
    for eid in EIDS_NEURAL:
        rec = Rec(one, eid)
        clusters = rec.clusters
        metrics = clusters.pop("metrics")
        clusters.pop("siMetrics", None)
        clusters.pop("waveforms", None)
        clusters = clusters.to_df()
        clusters = clusters.merge(metrics, left_index=True, right_on="cluster_id")
        clusters["eid"] = rec.eid
        clusters["subject"] = rec.subject
        clusters["genotype"] = rec.genotype
        all_clusters = pd.concat([all_clusters, clusters])

    all_clusters.reset_index(drop=True, inplace=True)

    all_clusters.to_parquet("cluster_features.pqt")


def plot_delta_phase_curves(df, save_fn, col_width=1, height=3):
    df_delta = compute_delta_phase_curves(df)

    # Set the y-axis label and limits
    yval = "delta"
    ylabel = "$FR_{(stim-control)}$ (sp/s/unit)"
    ylim = [-20, 20]
    yticks = [-20, 0, 20]
    n_conditions = df_delta["condition"].nunique()

    # Set the order of the conditions and phases
    condition_order = ["hold", "insp", "exp", "hb"]
    phase_order = ["tonic", "insp", "exp"]
    order = {"col": condition_order, "row": phase_order}

    # Set dimensions of the plot
    width = n_conditions * col_width
    col = "condition" if n_conditions > 1 else None
    p = (
        so.Plot(df_delta, x="phase", y=yval, color="genotype")
        .add(so.Line(), so.Agg(), legend=False)
        .add(so.Band(), so.Est(errorbar="se"), legend=False)
        .facet(row="category", col=col, order=order)
        .layout(size=(width, height))
        .scale(
            x=so.Continuous().tick(at=[-np.pi, 0, np.pi]),
            y=so.Continuous().tick(at=yticks),
            color=GENOTYPE_COLORS,
        )
        .label(x="".format, y="")
        .limit(y=ylim, x=[-np.pi, np.pi])
    ).plot()

    for ii, ax in enumerate(p._figure.axes):
        title = ax.get_title()
        condition, phase = title.split(" | ")
        if ii < n_conditions:
            ax.set_title(
                CONDITION_TITLES[condition],
                color=CONDITION_COLORS[condition],
                size=plt.rcParams["axes.labelsize"],
            )
        else:
            ax.set_title("")
        ax.set_ylabel(phase.capitalize() + " units", color=PHASE_MAP[phase])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.axvline(0, color="gray", linestyle="--")
        ax.axhline(0, color="gray", linestyle="--")

    p._figure.supylabel(ylabel, size=plt.rcParams["axes.labelsize"])
    p._figure.supxlabel("Phase (rads.)", size=plt.rcParams["axes.labelsize"])
    p._figure.subplots_adjust(wspace=0.2, hspace=0.2)
    p._figure.suptitle("Stimulation condition", size=plt.rcParams["axes.titlesize"])
    p._figure.tight_layout()
    p.save(save_fn)


def plot_delta_phase_curves_by_stim_type(df, save_fn, col_width=1, height=3):
    df_delta = compute_delta_phase_curves(df)

    # Set the y-axis label and limits
    yval = "delta"
    ylabel = "$FR_{(stim-control)}$ (sp/s/unit)"
    ylim = [-20, 20]
    yticks = [-20, 0, 20]
    n_conditions = df_delta["condition"].nunique()
    n_genotypes = df["genotype"].nunique()

    col = "genotype"
    color = "condition"
    row = "category"

    # Set the order of the conditions and phases
    condition_order = ["hold", "insp", "exp", "hb"]
    phase_order = ["tonic", "insp", "exp"]
    genotype_order = ["vgatai32", "vgatcre_ntschrmine", "vglut2ai32"]
    order = {"col": genotype_order, "row": phase_order}

    # Set dimensions of the plot
    width = n_genotypes * col_width

    p = (
        so.Plot(df_delta, x="phase", y=yval, color=color)
        .add(so.Line(), so.Agg())
        .add(so.Band(), so.Est(errorbar="se"))
        .facet(row=row, col=col, order=order)
        .layout(size=(width, height))
        .scale(
            x=so.Continuous().tick(at=[-np.pi, 0, np.pi]),
            y=so.Continuous().tick(at=yticks),
            color=CONDITION_COLORS,
        )
        .label(x="".format, y="")
        .limit(y=ylim, x=[-np.pi, np.pi])
    ).plot()

    for ii, ax in enumerate(p._figure.axes):
        title = ax.get_title()
        gg, category = title.split(" | ")
        if ii < n_genotypes:
            ax.set_title(
                GENOTYPE_LABELS[gg],
                color=GENOTYPE_COLORS[gg],
                size=plt.rcParams["axes.labelsize"],
            )
        else:
            ax.set_title("")
        ax.set_ylabel(category.capitalize() + " units", color=PHASE_MAP[category])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.axvline(0, color="gray", linestyle="--")
        ax.axhline(0, color="gray", linestyle="--")
    for handle, text in zip(
        p._figure.legends[0].legend_handles, p._figure.legends[0].texts
    ):
        text.set_text(CONDITION_TITLES[text.get_text()])

    p._figure.legends[0].set_title("Stim condition")
    p._figure.supylabel(ylabel, size=plt.rcParams["axes.labelsize"])
    p._figure.supxlabel("Phase (rads.)", size=plt.rcParams["axes.labelsize"])
    p._figure.subplots_adjust(wspace=0.2, hspace=0.2)
    # p._figure.suptitle("Stimulation condition", size=plt.rcParams["axes.titlesize"])
    p._figure.tight_layout()
    p.save(save_fn)


def plot_phase_curves(df, save_fn, col_width=1, height=3):
    """
    Plot the phase curves as returned by "compute_phase_curves_all_conditions"

    Args:
        df (DataFrame): Dataframe returned by compute_phase_curves_all_conditions
        save_fn (str or Path): Filename to save the plot to
        col_width (int, optional): Width of each column in the plot. Defaults to 1.
        height (int, optional): Height of each row in the plot. Defaults to 3.
        plot_delta (bool, optional): If True, plot the delta phase curves. Defaults to False.

    """
    # Get the number of genotypes and the order of the genotypes
    n_genotypes = df["genotype"].nunique()
    genotype_order = ["vgatai32", "vgatcre_ntschrmine", "vglut2ai32"]
    phase_order = ["tonic", "insp", "exp"]
    order = {"col": genotype_order, "row": phase_order}

    # Set the y-axis label and limits
    yval = "rate"
    ylabel = "Firing rate (sp/s/unit)"
    ylim = [0, 80]
    yticks = [0, 40, 80]

    # Set dimensions of the plot
    width = n_genotypes * col_width
    col = "genotype" if n_genotypes > 1 else None

    p = (
        so.Plot(df, x="phase", y=yval, color="condition")
        .add(so.Line(), so.Agg())
        .add(so.Band(), so.Est(errorbar="se"))
        .facet(row="category", col=col, order=order)
        .layout(size=(width, height))
        .scale(
            x=so.Continuous().tick(at=[-np.pi, 0, np.pi]),
            y=so.Continuous().tick(at=yticks),
            color=CONDITION_COLORS,
        )
        .label(x="".format, y="")
        .limit(y=ylim, x=[-np.pi, np.pi])
    ).plot()

    # Add labels to each row and column
    for ii, ax in enumerate(p._figure.axes):
        title = ax.get_title()
        if n_genotypes > 1:
            gg, phase = title.split(" | ")
        else:
            gg = df.genotype.unique()[0]
            phase = title

        if ii < n_genotypes:
            ax.set_title(
                GENOTYPE_LABELS[gg],
                color=GENOTYPE_COLORS[gg],
                size=plt.rcParams["axes.labelsize"],
            )
        else:
            ax.set_title("")
        ax.set_ylabel(phase.capitalize() + " units", color=PHASE_MAP[phase])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.axvline(0, color="gray", linestyle="--")
    # Update the figure legend to use the CONDITION_TITLES

    for handle, text in zip(
        p._figure.legends[0].legend_handles, p._figure.legends[0].texts
    ):
        text.set_text(CONDITION_TITLES[text.get_text()])
    p._figure.legends[0].set_title("Stim condition")
    p._figure.supylabel(ylabel, size=plt.rcParams["axes.labelsize"])
    p._figure.supxlabel("Phase (rads.)", size=plt.rcParams["axes.labelsize"])
    p._figure.subplots_adjust(wspace=0.2, hspace=0.2)
    p._figure.tight_layout()
    p.save(save_fn)


def plot_example_spiketrains(rec, n_per_phase=10, pre_time=3, post_time=5, stimulus='hold',ext=EXT,save_tgl=True,fs=(3,2),ax=None):
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
    stims, intervals = rec.get_stims(stimulus)
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
    if ax is None:
        f = plt.figure(figsize=fs)
        ax = f.add_subplot(111)
    ax.eventplot(raster, linelengths=1, lineoffsets=1.1, colors=colors, linewidths=0.25)
    for stim in stims:
        if stimulus == "hb":
            cc = HB_MAP[5]
        else:
            cc = rec.laser_color
        if stimulus  in ['10ms','50ms']:
            ax.axvline(stim[0] - t0, color=cc,alpha=0.5)
        else:
            ax.axvspan(stim[0] - t0, stim[1] - t0, color=cc, alpha=0.2)
    s0, sf = rec.physiology.times.searchsorted([win_start, win_stop])
    dia_sub = rec.physiology.dia.copy()[s0:sf]
    dia_sub /= dia_sub.max()
    dia_sub *= n_units_total * 0.1
    dia_sub += n_units_total * 1.15
    ax.plot(rec.physiology.times[s0:sf] - t0, dia_sub, c=plt.rcParams["text.color"])
    ax.set_xlim(-pre_time, post_time)
    replace_timeaxis_with_scalebar(ax)
    # remove yaxis
    ax.get_yaxis().set_visible(False)
    ax.spines["left"].set_visible(False)

    if save_tgl:
        plt.savefig(
            f"example_spike_train_{rec.genotype}_{rec.subject}_g{rec.sequence}{ext}"
        )

    return ax


def plot_all_peth():
    """
    Plot a peth and raster for every unit to inspect and verify respiratory related
    """
    print("=" * 20)
    print("Plotting all single cell PETHs")
    print("=" * 20)
    for eid in EIDS_NEURAL:
        rec = Rec(one, eid)
        breaths = rec.breaths.to_df()
        intervals, stims = rec.get_stims("control")
        breaths_subset = get_breaths_subset(breaths, stims)
        # Make a folder with recording name
        Path(rec.prefix).mkdir(exist_ok=True)

        for phase in ["insp", "exp", "tonic"]:
            print(f"'\tPlotting {phase} units")
            cluster_ids = rec.get_phasic_unit_ids(phase)
            for clu in tqdm(cluster_ids):
                # Compute PETH
                spike_times = rec.spikes.times[rec.spikes.clusters == clu]
                starts = breaths_subset["on_sec"].values
                stops = breaths_subset["off_sec"].values
                ax_rast, ax_peth = plot_peth_and_raster(
                    spike_times,
                    starts,
                    stops,
                    pre_time=0.5,
                    post_time=1,
                    subplot_ratio=(1, 2),
                    figsize=(3, 3),
                    raster_ylabel="Breath #",
                )
                respMod = rec.clusters.respMod[clu]
                ax_peth.set_title(f"Cluster {clu} | {phase} | respMod {respMod:0.2f}")
                #  TODO: Save to file in the recording folder
                plt.savefig(
                    f"{rec.prefix}/cluster_{respMod:0.2f}_{phase}_{clu}.png", dpi=150
                )
                plt.close("all")


def plot_scatters(mean_frs, conditions =None,fs=(3.25, 3.25), ext=EXT):
    ps = 1
    half_line = lambda x: x / 2
    double_line = lambda x: x * 2
    x = np.logspace(-2, 4, 100)

    # Make the tonic units a little darker
    phase_map = PHASE_MAP.copy()
    phase_map["tonic"] = plt.rcParams["text.color"]
    if conditions is None:
        conditions = ["insp", "exp", "hold", "hb"]
    elif isinstance(conditions, str):
        conditions = [conditions]

    for condition in conditions:
        p = (
            so.Plot(mean_frs, x="control", y=condition, color="category")
            .facet(col="category", row="genotype",order={"row": GENOTYPE_ORDER})
            .add(so.Dot(pointsize=ps, edgewidth=0, alpha=0.6), legend=False)
            .limit(x=[0, 2e2], y=[0, 2e2])
            .scale(
                x=so.Continuous(trans="symlog").label(base=None),
                y=so.Continuous(trans="symlog").label(base=None),
                color=phase_map,
            )
            .layout(size=fs, engine="constrained")
        ).plot()
        p

        for ii, ax in enumerate(p._figure.axes):
            ax.set_aspect("equal")
            ax.axline([1e-2, 1e-2], [1e2, 1e2], color="k", linestyle="--")
            ax.plot(x, double_line(x), color="gray", linestyle="--")
            ax.plot(x, half_line(x), color="gray", linestyle="--")

            category, gg = ax.get_title().split(" | ")
                        
            # Plot median
            mask = (mean_frs.index.get_level_values('category') == category) & (mean_frs.index.get_level_values('genotype') == gg)
            x_data = mean_frs.loc[mask, 'control']
            y_data = mean_frs.loc[mask, condition]
            
            x_median = x_data.median()
            y_median = y_data.median()
            
            # Plot centroid point
            centroid_color = phase_map[category]
            edge_color = 'white' if centroid_color == plt.rcParams["text.color"] else 'black'
            ax.plot(x_median, y_median, 'o', markersize=3, markerfacecolor=centroid_color, 
                   markeredgecolor=edge_color, markeredgewidth=0.5, zorder=10)
            
            
            ax.set_ylabel(f"{GENOTYPE_LABELS[gg]}", color=GENOTYPE_COLORS[gg])
            ax.set_xlabel("")
            if ii < 3:
                ax.set_title(
                    f"{category.capitalize()} units", color=PHASE_MAP[category]
                )
            else:
                ax.set_title("")
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

        p._figure.supxlabel(
            "Control firing rate (sp/s)",
            size=plt.rcParams["axes.labelsize"],
            color=plt.rcParams["axes.labelcolor"],
        )
        p._figure.supylabel(
            f"{CONDITION_TITLES[condition]} stim. firing rate (sp/s)",
            size=plt.rcParams["axes.labelsize"],
            color=plt.rcParams["axes.labelcolor"],
        )
        p
        p.save(f"fr_scatter_{condition}{ext}")
        plt.close("all")


def plot_cdf_stims_by_genotype(pct_diff, fs=(4, 3.25), ext=EXT):
    p = (
        so.Plot(pct_diff, x="deltalogFR", color="genotype")
        .facet(col="condition", row="category")
        .add(
            so.Line(),
            so.Hist("percent", cumulative=True, common_norm=False),
            legend=False,
        )
        .scale(color=GENOTYPE_COLORS)
        .label(y="% neurons")
        .layout(size=fs, engine="constrained")
        .limit(x=[-3.5, 3.5], y=[0, 100])
        .scale(
            x=so.Continuous().tick(at=[-2, -0, 2]),
            y=so.Continuous().tick(at=[0, 25, 50, 75, 100]),
        )
    ).plot()

    for ii, ax in enumerate(p._figure.axes):
        ax.axvline(0, color="k", linestyle="--")
        ax.axhline(50, color="k", linestyle="--")
        ax.grid(axis="y", ls="--", lw=0.25)

        condition, category = ax.get_title().split(" | ")
        ax.set_ylabel(f"{category.capitalize()} units", color=PHASE_MAP[category])
        ax.set_xlabel("")
        if ii < 4:
            ax.set_title(
                CONDITION_TITLES[condition],
                color=CONDITION_COLORS[condition],
                size=plt.rcParams["axes.labelsize"],
            )
        else:
            ax.set_title("")
    p._figure.supxlabel(
        "$\\Lleftarrow$(F.R.$\\downarrow$)      $log_2(\\frac{FR_{stim}}{FR_{control}})$      ($\\uparrow$F.R.)$\\Rrightarrow$",
        size=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
    )
    p._figure.supylabel(
        "Cumulative % of units ",
        size=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
    )
    p._figure.suptitle(
        "Stimulus type type",
        size=plt.rcParams["axes.titlesize"],
        color=plt.rcParams["axes.labelcolor"],
    )
    p.save(f"log2fold_fr_change_cdf_by_gentoype{ext}")


def plot_cdf_holds_stims_by_unit_type(pct_diff, fs=(1.75, 3), ext=EXT):
    # ------------------------------- #
    # Plot CDF of firing rate for different unit types only hold stimulus
    # ------------------------------- #
    p = (
        so.Plot(pct_diff.query('condition=="hold"'), x="deltalogFR", color="category")
        .facet(row="genotype",order=GENOTYPE_ORDER)
        .add(
            so.Line(),
            so.Hist("percent", cumulative=True, common_norm=False),
            legend=False,
        )
        .scale(color=PHASE_MAP)
        .label(x="$\\frac{FR_{stim}}{FR_{control}}$")
        .layout(size=fs, engine="constrained")
        .limit(x=[-3.5, 3.5], y=[0, 100])
        .scale(
            x=so.Continuous().tick(at=[-2, -0, 2]).label(like=log2_formatter),
            y=so.Continuous().tick(at=[0, 25, 50, 75, 100]),
        )
    ).plot()

    for ii, ax in enumerate(p._figure.axes):
        ax.axvline(0, color="k", linestyle="--")
        ax.axhline(50, color="k", linestyle="--")
        ax.grid(axis="y", ls="--", lw=0.25)

        gg = ax.get_title()
        ax.set_ylabel(f"{GENOTYPE_LABELS[gg]}", color=GENOTYPE_COLORS[gg])
        ax.set_title("")
    p._figure.supylabel(
        "Cumulative % of units ",
        size=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
    )
    p.save(f"log2fold_fr_change_hold_cdf{ext}")


def plot_cdf_hold_stims_by_genotype(pct_diff, fs=(1.75, 3), ext=EXT):
    # ------------------------------- #
    # Plot CDF of firing rate for different genotypes only hold stimulus
    # ------------------------------- #
    p = (
        so.Plot(pct_diff.query('condition=="hold"'), x="deltalogFR", color="genotype")
        .facet(row="category")
        .add(
            so.Line(),
            so.Hist("percent", cumulative=True, common_norm=False),
            legend=False,
        )
        .scale(color=GENOTYPE_COLORS)
        .label(x="$\\frac{FR_{stim}}{FR_{control}}$")
        .layout(size=fs, engine="constrained")
        .limit(x=[-3.5, 3.5], y=[0, 100])
        .scale(
            x=so.Continuous().tick(at=[-2, -0, 2]).label(like=log2_formatter),
            y=so.Continuous().tick(at=[0, 25, 50, 75, 100]),
        )
    ).plot()

    for ii, ax in enumerate(p._figure.axes):
        ax.axvline(0, color="k", linestyle="--")
        ax.axhline(50, color="k", linestyle="--")
        ax.grid(axis="y", ls="--", lw=0.25)

        category = ax.get_title()
        ax.set_ylabel(f"{category.capitalize()} units", color=PHASE_MAP[category])
        ax.set_title("")
    p._figure.supylabel(
        "Cumulative % of units ",
        size=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
    )
    p.save(f"log2fold_fr_change_hold_cdf_by_gentoype{ext}")


def plot_delta_fr_vs_depth(pct_diff, cluster_features, ext=EXT):
    # -------------------------------- #
    # Plot delta_fr as function of depth
    # -------------------------------- #
    pct_diff_with_depths = pd.merge(
        pct_diff.reset_index(), cluster_features[["uuids", "depthsVII"]], on="uuids"
    )
    _df = pct_diff_with_depths.query('condition=="hold" & category=="exp"')
    _df = _df.query("deltalogFR < 0.5 or deltalogFR > -0.5")
    p = (
        so.Plot(_df, x="depthsVII", y="deltalogFR", color="genotype")
        .facet(row="genotype")
        .add(so.Dot(pointsize=1))
        # .limit(y=[-3,3])
        .layout(size=(3, 3))
    ).plot()
    p.save(f"delta_fr_vs_depth{ext}")


def plot_cluster_features(cluster_features, ext=EXT):
    """
    Plot distributions of cluster-level  features
    """
    # Plot scatter of preferred phase vs max firing rate phase for each type of unit (I/E/Tonic)
    (
        so.Plot(cluster_features.query('category!="qc_fail"'))
        .pair(x=["preferredPhase"], y=["maxFiringRatePhase"])
        .add(so.Dots(), color="category", pointsize="respMod", alpha="respMod")
        .scale(color=PHASE_MAP)
        .limit(x=[-np.pi, np.pi], y=[-np.pi, np.pi])
        .label(x="Preferred phase", y="Max firing rate phase")
        .layout(size=(3, 3))
        .save(f"preferred_phase_vs_maxphase_fr{ext}")
    )

    # Plot CDF of respiratory modulation by genotype
    (
        so.Plot(
            cluster_features.query('category!="qc_fail"'), "respMod", color="genotype"
        )
        .add(so.Area(), so.Hist("percent", cumulative=True, common_norm=False))
        .scale(color=GENOTYPE_COLORS)
        .label(y="Percent")
        .save(f"respMod_by_genotype{ext}")
    )

    # Plot histogram of number of neurons by probe depth
    (
        so.Plot(cluster_features, "depths")
        .add(so.Area(), so.Hist("count", binwidth=100), color="category")
        .scale(color=PHASE_MAP)
        .label(y="# neurons")
        .save(f"num_neurons_by_probe_depth{ext}")
    )

    # Plot scatter of respiratory modulation vs probe depth, colored by preferred phase
    (
        so.Plot(cluster_features.query('category!="qc_fail"'), x="depths", y="respMod")
        .add(so.Dot(), color="preferredPhase", pointsize="respMod", alpha="respMod")
        .scale(color="RdBu_r")
        .save(f"respMod_and_preferred_phase_by_probe_depth{ext}")
    )

    # Plot CDF of firing rate by phase
    (
        so.Plot(cluster_features, x="firing_rate")
        .add(
            so.Line(),
            so.Hist("percent", cumulative=True, binwidth=0.2, common_norm=False),
            color="category",
            common_norm=True,
        )
        .add(
            so.Line(color="k", linewidth=2, linestyle="--"),
            so.Hist("percent", cumulative=True, common_norm=False),
        )
        .scale(color=PHASE_MAP, x=so.Continuous(trans="log"))
        .label(y="Percent")
        .save(f"firing_rate_cumulative{ext}")
    )

    # Plot CDF of presence ratio by I/E/Tonic category
    (
        so.Plot(cluster_features, x="presence_ratio")
        .add(
            so.Line(),
            so.Hist("percent", cumulative=True, common_norm=False),
            color="category",
            common_norm=True,
        )
        .scale(color=PHASE_MAP)
        .save(f"presence_ratio_cumulative{ext}")
    )

    # Plot CDF of respiratory modulation by I/E/Tonic category
    (
        so.Plot(cluster_features, x="respMod")
        .add(
            so.Line(),
            so.Hist("percent", cumulative=True, common_norm=False),
            color="category",
        )
        .add(
            so.Line(color="k", linewidth=2, linestyle="--"),
            so.Hist("percent", cumulative=True, common_norm=False),
        )
        .scale(color=PHASE_MAP)
        .label(y="Percent")
        .save(f"respMod_cumulative{ext}")
    )


def plot_hb_vs_hold_scatter(mean_frs, ext=EXT):
    # Plot  log-log scatter of firing rate during opto hold (2s) and hering breuer
    # Add LM
    # Compute slopes and R and compare.
    # # Expect Vgat/NTS to be close to 1 and high r, vglut less so. For all.

    half_line = lambda x: x / 2
    double_line = lambda x: x * 2
    x = np.linspace(0, 200, 100)

    ps = 1
    phase_map = PHASE_MAP.copy()
    phase_map["tonic"] = plt.rcParams["text.color"]
    p = (
        so.Plot(mean_frs, "hb", "hold", color="category")
        .facet(col="category", row="genotype",order  = {"row": GENOTYPE_ORDER, "col": PHASE_ORDER})
        .add(so.Dot(pointsize=ps, edgewidth=0, alpha=0.7), legend=False)
        .limit(x=[0, 151], y=[0, 151])
        .scale(
            x=so.Continuous(trans="sqrt").tick(at=[0,10,50,150]).label(),
            y=so.Continuous(trans="sqrt").tick(at=[0,10,50,150]).label(),
            color=phase_map,
        )
        .layout(size=(3.5, 3),engine='constrained')
    ).plot()

    for ii, ax in enumerate(p._figure.axes):
        title = ax.get_title()
        phase, gg = title.split(" | ")
                
        # Plot median
        mask = (mean_frs.index.get_level_values('category') == phase) & (mean_frs.index.get_level_values('genotype') == gg)
        x_data = mean_frs.loc[mask, 'hb']
        y_data = mean_frs.loc[mask, 'hold']
        
        x_median = x_data.median()
        y_median = y_data.median()

        # Plot centroid point
        centroid_color = phase_map[phase]
        edge_color = 'white' if centroid_color == plt.rcParams["text.color"] else 'black'
        ax.plot(x_median, y_median, 'o', markersize=3, markerfacecolor=centroid_color, 
               markeredgecolor=edge_color, markeredgewidth=0.5, zorder=10)
    
        if ii < 3:
            ax.set_title(phase.capitalize() + " units", color=phase_map[phase],size='small')
        else:
            ax.set_title("")
        ax.set_ylabel(GENOTYPE_LABELS[gg], color=GENOTYPE_COLORS[gg])
        ax.set_xlabel("")
        ax.axhline(0, color="gray", linestyle="--")
        ax.axvline(0, color="gray", linestyle="--")
        ax.axline((-1.2, -1.2), (1.2, 1.2), color="k", ls=":")
        ax.plot(x, double_line(x), color="gray", linestyle="--")
        ax.plot(x, half_line(x), color="gray", linestyle="--")
        if ii==0:
            ax.text(100,50,'0.5x',color='gray',fontsize='xx-small',rotation=30,va='top',ha='left')
            ax.text(50,100,'2x',color='gray',fontsize='xx-small',rotation=60,va='bottom',ha='right')




    ax.set_aspect("equal")
    p._figure.supylabel("$FR_{opto-hold}$ (sp/s)", size='small')
    p._figure.supxlabel(
        "$FR_{Hering-Breuer}$ (sp/s)", size='small'
    )
    p

    p.save(f"hb_vs_hold_scatter{ext}")
    plt.close("all")



    # Compute means and covariances
    df = np.sqrt(mean_frs.copy()[['hb','hold']])
    means = df.groupby(['genotype','category']).mean()
    means['diff_mean_sqrt'] = means['hold'] - means['hb']
    corrs = df.groupby(['genotype','category']).corr().unstack()['hb']['hold']
    corrs.name= 'Corr'
    df = pd.merge(means['diff_mean_sqrt'], corrs, left_index=True, right_index=True)

    markers = {
        'insp': 'v',
        'exp': '^',
        'tonic': 'o',
    }

    p = (
        so.Plot(
            df.reset_index(), x="diff_mean_sqrt", y="Corr", color="genotype"
        )
        .add(so.Dot(edgewidth=0.25,edgecolor='k'), pointsize='Corr',marker='category',edgestyle='category')
        .scale(color=GENOTYPE_COLORS,
               marker=markers,
               x=so.Continuous().tick(at=[-2,0,2]),
               y=so.Continuous().tick(at=[0,0.5,1]),
               pointsize=so.Continuous((3,6)),
               )    
            
        .layout(size=(1.75,2.25))
        .limit(x=[-2.5,2.5],y=[0,1])
        .label(x=r"$\bar{\sqrt{FR_{opto}}} - \bar{\sqrt{FR_{Hering-Breuer}}}$",
               y=r"$\rho_{\sqrt{HB}, \sqrt{opto}}$")
    ).plot()
    ax = p._figure.axes[0]
    ax.set_axisbelow(True)
    ax.grid(ls='--', lw=0.25, color='k')
    p.save("hb_vs_hold_scatter_correlation.pdf")
    plt.close('all')


def load_and_plot_phase_curves(ext):
    # Load and plot phase curves
    _log.info("Loading phase curves")
    phase_curves = pd.read_parquet(save_fn_phase_curves)
    plot_phase_curves(phase_curves, f"all_population_phase_curves{ext}")
    plot_delta_phase_curves(phase_curves, f"all_population_delta_phase_curves{ext}")
    plot_delta_phase_curves_by_stim_type(
        phase_curves, f"all_population_delta_phase_curves_by_stim_type{ext}"
    )


def load_mean_frs():
    # Load and aggregate mean firing rates so that each row is a unit and each column is a condition
    # Mean FRs are only for units that pass QC_QUERY
    _log.info("Loading mean firing rates")
    mean_frs_raw = pd.read_parquet(save_fn_firing_rates)
    mean_frs = mean_frs_raw.pivot(
        values="fr", columns=["condition"], index=["genotype", "category", "uuids"]
    )
    return mean_frs, mean_frs_raw


def load_cluster_features():
    # Load cluster level features
    cluster_features = pd.read_parquet("cluster_features.pqt")
    # Identify units that did not pass QC and set those categorues to qc_fail
    idx = cluster_features.query(f"~({QC_QUERY})").index
    cluster_features.loc[idx, "category"] = "qc_fail"
    return cluster_features


def compute_delta_FRs(mean_frs):
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
    return pct_diff


def cdf_stats(pct_diff):
    # Compute CDF stats for each unit type and condition

    from itertools import product
    def p2sig(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "n.s."

    TINY = 1e-2
    def _run_wilcoxon(vals,stats,index):
        res  = scipy.stats.wilcoxon(vals,np.zeros_like(vals))
        _stats = pd.DataFrame(
            {
                "wilcoxon": res.statistic,
                "p": res.pvalue,
                "reject": res.pvalue < 0.05,
                "signifigance": p2sig(res.pvalue),
                "category": category,
                "condition": condition,
                "genotype": genotype,
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(),
            },index=[index])
        stats = pd.concat([stats, _stats])
        return(stats)

    conditions = ["hold", "exp", "insp", "hb"]
    categories = ["exp", "insp", "tonic"]
    genotypes = GENOTYPE_COLORS.keys()
    stats = pd.DataFrame()
    df = pct_diff.copy().reset_index()
    ii=0

    for condition,category,genotype in product(conditions,categories,genotypes):
        ii+=1
        _df = df.query(f'condition==@condition & category==@category & genotype==@genotype')
        vals = np.log(_df["FR_pct_change"]+TINY).values
        stats = _run_wilcoxon(vals,stats,ii)
    stats = stats.set_index(['condition','category','genotype'])
    stats.to_csv('cdf_stats.csv')



    stats = pd.DataFrame()
    ii=0
    for category in categories:
        _df = df.query('condition=="hb" & category==@category')
        vals = np.log(_df["FR_pct_change"]+TINY)
        stats = _run_wilcoxon(vals,stats,ii)
        stats = stats.drop(columns=['genotype','condition'])
        ii+=1
    stats.to_csv('cdf_stats_hb_agg_genotypes.csv')


def main_compute():
    """
    Computes mean firing rates and phase curves
    for all stimulus conditions and all subjects
    and saves to parquet files
    """

    _log.info("Computing phase curves")
    all_phase_curves = pd.DataFrame()
    all_mean_fr_epochs = pd.DataFrame()

    # Loop through all recordings
    for eid in EIDS_NEURAL:
        rec = Rec(one, eid)
        # -------------------- #
        # Compute the phase curves
        # -------------------- #
        try:
            phase_curves = compute_phase_curves_all_conditions(rec)
        except:
            _log.error(f"Error computing phase curves for {rec.prefix}")
        all_phase_curves = pd.concat([all_phase_curves, phase_curves])

        # -------------------- #
        # Compute the mean firing rates
        # -------------------- #
        try:
            mean_fr_epochs = compute_mean_fr_epochs(rec)
        except:
            _log.error(f"Error computing mean firing rates for {rec.prefix}")
        all_mean_fr_epochs = pd.concat([all_mean_fr_epochs, mean_fr_epochs])

    # Save the data
    _log.info("saving phase curves")
    all_phase_curves.to_parquet(save_fn_phase_curves)
    _log.info("saving average firing rates")
    all_mean_fr_epochs.to_parquet(save_fn_firing_rates)



# ============================ #
# Main entry point
# ============================ #
@click.command()
@click.option("--compute", "-c", is_flag=True)
@click.option("--all_peth", "-p", is_flag=True)
@click.option("--ext", "-e", default="pdf")
def main(compute, all_peth, ext):
    assert ext in ["pdf", "png", "svg"]
    ext = "." + ext
    if compute:
        main_compute()
        compute_cluster_level_dataset()

    # Phase Curves
    load_and_plot_phase_curves(ext)

    # Load mean firing rates
    mean_frs, mean_frs_raw = load_mean_frs()

    # Load cluster level features
    cluster_features = load_cluster_features()

    # Compute delta firing rates
    pct_diff = compute_delta_FRs(mean_frs)

    # -------------------- #
    # Plotting
    # -------------------- #

    plot_scatters(mean_frs,conditions='hold', ext=ext)

    plot_cdf_holds_stims_by_unit_type(pct_diff, fs=(1.75, 3), ext=ext)
    plot_cdf_hold_stims_by_genotype(pct_diff, fs=(1.75, 3), ext=ext)
    cdf_stats(pct_diff)

    plot_delta_fr_vs_depth(pct_diff, cluster_features, ext=ext)

    # Cluster level features
    plot_cluster_features(cluster_features, ext=ext)

    # Plot scatter of hold vs HB firing rate
    plot_hb_vs_hold_scatter(mean_frs, ext=ext)

    # Plot example trains for long holds:
    subjects = [("m2024-40", 2), ("m2024-34", 0), ("m2024-28", 0)]
    for subject, number in subjects:
        eid = one.search(subject=subject, number=number)[0]
        rec = Rec(one, eid)
        plot_example_spiketrains(rec, n_per_phase=15)

    if all_peth:
        plot_all_peth()


if __name__ == "__main__":
    main()
