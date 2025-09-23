import pandas as pd
import seaborn.objects as so
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("../")
sys.path.append("VLAD/")
from cibrrig.plot import (
    plot_laser,
    plot_sweeps,
    laser_colors,
    trim_yscale_to_lims,
    replace_timeaxis_with_scalebar,
)
from brainbox.singlecell import calculate_peths
from utils import (
    PHASE_MAP,
    HB_MAP,
    Rec,
    GENOTYPE_LABELS,
    GENOTYPE_COLORS,
    one,
    EIDS_PHYSIOL,
    sessions_to_include,
    set_style,
    sig2star,
)
import numpy as np
import pingouin as pg
import statsmodels.formula.api as smf

set_style()

EXT = ".pdf"
GENOTYPES = GENOTYPE_COLORS.keys()
DASH_LW = 1.5
PS = 2
genotype_order = ["vglut2ai32", "vgatai32", "vgatcre_ntschrmine"]


def rebound_analyses(dv, label, fs=(2.0, 1.5), ext=EXT):
    df = pd.read_csv("rebound_latencies.csv")
    df = pd.pivot_table(df, index=["eid", "genotype"], values=dv, aggfunc="median")

    if dv in ["latency", "duration"]:
        df[dv] = df[dv] * 1000

    p = (
        so.Plot(df, y="genotype", x=dv, color="genotype")
        .add(so.Bar(width=0.5), so.Agg(), legend=False)
        .add(
            so.Range(color="k", linewidth=1),
            so.Est("mean", "se"),
            legend=False,
        )
        .add(
            so.Dot(pointsize=PS, alpha=0.5, color="w", edgecolor="k"),
            so.Jitter(),
            group="eid",
            legend=False,
        )
        .scale(color=GENOTYPE_COLORS, y=so.Nominal(order=genotype_order))
        .layout(size=fs)
        .label(y="", x=label)
    ).plot()
    ax = p._figure.axes[0]
    xticklabels = [GENOTYPE_LABELS[x.get_text()] for x in ax.get_yticklabels()]

    ax.set_yticklabels(xticklabels)
    p.save(f"{dv}_from_hold_off_to_breath{ext}")

    kruskal = pg.kruskal(df.reset_index(), dv=dv, between="genotype", detailed=True)
    kruskal["significant"] = kruskal["p-unc"] < 0.05
    kruskal["sigstars"] = kruskal["p-unc"].apply(sig2star)
    kruskal.to_csv(f"{dv}_kruskal.csv", index=False)

    pairwise = pg.pairwise_tests(
        df.reset_index(), dv=dv, between="genotype", padjust="holm", parametric=False
    )
    pairwise["significant"] = pairwise["p-corr"] < 0.05
    pairwise["sigstars"] = pairwise["p-corr"].apply(sig2star)
    pairwise.to_csv(f"{dv}_pairwise.csv", index=False)


def plot_rebound(fs=(2.0, 1.5), ext=EXT):
    vars = [
        ("latency", "Latency (ms)"),
        ("amp", "Amplitude (norm)"),
        ("duration", "Duration (ms)"),
    ]
    for dv, label in vars:
        rebound_analyses(dv, label, fs=fs, ext=ext)


def process_intercept(reset_curve_intercept):
    """
    Process the reset curve intercept dataframe to get the delta intercept for each genotype and comparison
    """
    # Remove vglut2ai32 from the reset curve intercept dataframe
    reset_curve_intercept = reset_curve_intercept.query('genotype!="vglut2ai32"')
    reset_curve_intercept["d_peak"] = (
        reset_curve_intercept["intercept"] - reset_curve_intercept["peak_duty_cycle"]
    )
    reset_curve_intercept["d_insp"] = (
        reset_curve_intercept["intercept"] - reset_curve_intercept["duty_cycle"]
    )
    # Melt d_peak and d_insp
    reset_curve_intercept = reset_curve_intercept.melt(
        id_vars=["genotype", "eid"],
        value_vars=["d_peak", "d_insp"],
        var_name="comparison",
        value_name="delta_intercept",
    )
    return reset_curve_intercept


def add_laterality(df, sessions_to_include):
    # Add laterality (i.e., if stims are unilateral or bilateral)
    df = df.merge(sessions_to_include[["eid", "stim_laterality"]], on="eid")
    return df


def plot_reset_curve_intercept(reset_curve_intercept, ps=PS, ext=EXT, fs=(1.5, 1.5)):
    """
    Plot the phase at which the reset curve crosses 1 WRT diaphragm peak and end times

    Saves the figure and the stats
    Args:
        reset_curve_intercept (pd.DataFrame): Dataframe containing the reset curve intercepts
        ps (int): point size for the dots
        ext (str): file extension for saving the plot
        fs (tuple): figure size for the plot
    """
    p = (
        so.Plot(
            reset_curve_intercept, x="comparison", y="delta_intercept", color="genotype"
        )
        .add(so.Bar(), so.Agg(), so.Dodge(), legend=False)
        .add(
            so.Range(color="k", linewidth=1),
            so.Est("mean", "se"),
            so.Dodge(),
            legend=False,
        )
        .add(
            so.Dot(alpha=0.75, pointsize=ps, edgewidth=0),
            so.Jitter(),
            so.Dodge(),
            group="eid",
            marker="stim_laterality",
            legend=False,
        )
        .scale(color=GENOTYPE_COLORS, marker=["x", "o"])
        .label(y="$Reset_{intercept} - Dia._{feature}$\n(norm.)", x="")
        .layout(size=fs)
    ).plot()

    ax = p._figure.axes[0]
    label_map = {"d_peak": "Peak", "d_insp": "End"}
    ax.set_xticklabels([label_map[x.get_text()] for x in ax.get_xticklabels()])
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.axhline(0, color="k", ls="-")
    ax.grid(axis="y", ls="--")
    p.save(f"reset_curve_intercept{ext}", dpi=300)

    # Stats
    stat_rez = pd.DataFrame()
    for gg in reset_curve_intercept.genotype.unique():
        for comp in ["d_peak", "d_insp"]:
            _df = reset_curve_intercept.query(
                "genotype==@gg & comparison==@comp"
            ).dropna()
            rez = pg.ttest(_df["delta_intercept"], 0)
            rez["genotype"] = gg
            rez["metric"] = comp
            rez["n"] = _df.shape[0]
            stat_rez = pd.concat([stat_rez, rez])
    stat_rez["reject_null"] = stat_rez["p-val"] < 0.05
    stat_rez.to_csv("reset_curve_intercept_stats.csv", index=False)


def plot_opto_holds(long_opto_pivot, fs=(2, 1.65), ext=EXT):
    """
    Plot the respiratory rate for the long opto holds (2s)

    """
    # ---------------------------
    # Opto holds
    # ---------------------------
    p = (
        so.Plot(long_opto_pivot, x="comparison", y="resp_rate")
        .facet(col="genotype", order=GENOTYPES)
        .add(
            so.Dash(width=0.5, linewidth=DASH_LW),
            so.Agg(),
            color="comparison",
            legend=False,
        )
        .add(
            so.Range(color=plt.rcParams["text.color"], linewidth=1),
            so.Est("mean", "se"),
            legend=False,
        )
        .add(
            so.Line(color=plt.rcParams["text.color"], alpha=0.5),
            group="eid",
            legend=False,
            linestyle="stim_laterality",
        )
        .add(
            so.Dot(pointsize=PS, alpha=0.5, edgewidth=0),
            group="eid",
            color="comparison",
            legend=False,
        )
        .scale(color=["silver", laser_colors[473]], linestyle=[":", "-"])
        .label(y="Resp. rate (Hz)", x="")
        .limit(y=(0, None))
        .layout(size=fs)
    ).plot()
    for ax in p._figure.axes:
        ticklabels = [x.get_text().capitalize() for x in ax.get_xticklabels()]
        ax.set_xticklabels(ticklabels, rotation=45, ha="right", va="top")
        ax.set_title(
            GENOTYPE_LABELS[ax.get_title()],
            fontsize=plt.rcParams["axes.labelsize"],
            color=GENOTYPE_COLORS[ax.get_title()],
        )

    p.save(f"2s_stims{ext}", dpi=300)


def plot_opto_phasic(phasic_opto_pivot, fs=(2, 1.65), ext=EXT):
    """
    Plot the respiratory rate for the phasic opto holds (0.5s)

    """
    # ---------------------------
    # Opto phasic
    # ---------------------------
    p = (
        so.Plot(phasic_opto_pivot, x="comparison", y="resp_rate", color="phase")
        .facet(col="genotype", order=GENOTYPES)
        .add(
            so.Line(alpha=0.5, linewidth=0.5),
            group="eid",
            legend=False,
            linestyle="stim_laterality",
        )
        .add(so.Dash(width=0.5, linewidth=DASH_LW), so.Agg(), so.Dodge(), legend=False)
        .add(
            so.Range(color="k", linewidth=1),
            so.Est("mean", "se"),
            so.Dodge(),
            legend=False,
        )
        # .add(so.Dot(pointsize=2,alpha=0.5),group='eid',marker='comparison',legend=False)
        .scale(color=PHASE_MAP, marker=["o", "s"], linestyle=[":", "-"])
        .label(y="Resp. rate (Hz)", x="")
        .limit(y=(0, None))
        .layout(size=fs)
    ).plot()
    for ax in p._figure.axes:
        ax.xaxis.set_tick_params(rotation=45)
        ax.set_title(
            GENOTYPE_LABELS[ax.get_title()],
            fontsize=plt.rcParams["axes.labelsize"],
            color=GENOTYPE_COLORS[ax.get_title()],
        )
    p.save(f"phasic_stims{EXT}", dpi=300)


def plot_opto_phasic_amps(phasic_amps_pivot, fs=(3, 2), ext=EXT):
    df = pd.pivot_table(
        phasic_amps_pivot,
        index=["genotype", "eid", "phase", "stim_laterality"],
        columns="comparison",
        values=["amp_mean"],
    )
    normed = df["amp_mean"]["stim"] / df["amp_mean"]["control"] * 100
    label = r"Amplitude (% of baseline)"
    normed.name = label
    normed = normed.reset_index()
    p = (
        so.Plot(normed, x="phase", y=label, color="phase")
        .facet(col="genotype", order=GENOTYPES)
        .add(so.Bar(baseline=100, width=0.5), so.Agg(), legend=False)
        .add(so.Range(color="k", linewidth=1), so.Est("mean", "se"), legend=False)
        .add(
            so.Dot(pointsize=PS, color="k", alpha=0.75),
            so.Jitter(),
            so.Shift(0.25),
            group="eid",
            marker="stim_laterality",
            legend=False,
        )
        .scale(color=PHASE_MAP, marker=["x", "o"])
        .layout(size=fs)
        .label(x="")
    ).plot()
    for ax in p._figure.axes:
        ax.set_title(
            GENOTYPE_LABELS[ax.get_title()],
            fontsize=plt.rcParams["axes.labelsize"],
            color=GENOTYPE_COLORS[ax.get_title()],
        )
        ticklabels = [x.get_text().capitalize() for x in ax.get_xticklabels()]
        ax.set_xticklabels(ticklabels)
        ax.axhline(100, color="k", ls="--")
    p.save(f"phasic_stims_amps{ext}", dpi=300)


def plot_opto_phasic_normalized(phasic_opto_normed, fs=(2, 1.65), ext=EXT):
    """
    Plot the within animal normalized respiratory rate for the phasic opto stims
    """
    p = (
        so.Plot(phasic_opto_normed, x="phase", y="resp_rate_norm", color="phase")
        .facet(col="genotype", order=GENOTYPES)
        # .add(so.Dash(),so.Agg(),so.Dodge())
        .add(so.Bar(baseline=100, width=0.5), so.Agg(), legend=False)
        .add(so.Range(color="k", linewidth=1), so.Est("mean", "se"), legend=False)
        .add(
            so.Dot(
                alpha=0.75, color=plt.rcParams["text.color"], pointsize=PS, edgewidth=0
            ),
            so.Jitter(),
            so.Shift(0.25),
            group="eid",
            legend=False,
            marker="stim_laterality",
        )
        .scale(color=PHASE_MAP, marker=["x", "o"])
        .label(y="Resp. rate. (% of baseline)", x="")
        .layout(size=fs)
    ).plot()
    for ax in p._figure.axes:
        ax.set_xticks(ax.get_xticks())
        ticklabels = [x.get_text().capitalize() for x in ax.get_xticklabels()]
        ax.set_xticklabels(ticklabels)
        ax.axhline(100, color="k", ls="--")
        ax.set_title(
            GENOTYPE_LABELS[ax.get_title()],
            fontsize=plt.rcParams["axes.labelsize"],
            color=GENOTYPE_COLORS[ax.get_title()],
        )

    p.save(f"phasic_stims_percent_delta_{EXT}", dpi=300)


def plot_plot_opto_phasic_normalized_talk(phasic_opto_normed, fs=(3, 3), ext=EXT):
    # For talks
    """
    Plot the respiratory rate for the phasic opto holds (0.5s)

    """
    # ---------------------------
    # Opto phasic
    # ---------------------------
    with plt.style.context("../VLAD_Talks.mplstyle"):
        so.Plot.config.theme.update(matplotlib.rcParams)
        (
            so.Plot(
                phasic_opto_normed.query('genotype!="vglut2ai32"'),
                x="phase",
                y="resp_rate_norm",
                color="phase",
            )
            .facet(col="genotype", order=GENOTYPES)
            .add(so.Bar(baseline=100, width=0.5), so.Agg(), legend=False)
            .add(so.Range(color="k", linewidth=1), so.Est("mean", "se"), legend=False)
            .add(
                so.Dot(
                    alpha=0.75,
                    color=plt.rcParams["text.color"],
                    pointsize=3,
                    edgewidth=0,
                ),
                so.Jitter(),
                so.Shift(0.35),
                group="eid",
                legend=False,
            )
            .scale(color=PHASE_MAP)
            .label(y="Resp Rate. (% Change)", x="")
            .layout(size=fs)
            .save(f"phasic_stims_percent_delta_talk.svg", dpi=300)
        )
    so.Plot.config.theme.update(matplotlib.rcParams)


def plot_hering_breuer(HB_pivot, fs=(2.5, 2), ext=EXT):
    """
    Plot the respiratory rate for the Hering Breuer holds (2s and 5s)
    """
    p = (
        so.Plot(HB_pivot, x="comparison", y="resp_rate", color="duration")
        .facet(col="genotype", row="duration")
        .add(so.Line(alpha=0.5, linewidth=0.5), group="eid", legend=False)
        .add(so.Dash(width=0.5, linewidth=DASH_LW), so.Agg(), legend=False)
        .add(so.Range(color="k", linewidth=1), so.Est("mean", "se"), legend=False)
        # .add(so.Dot(),group='eid',marker='comparison',legend=False)
        .scale(color=HB_MAP, y=so.Continuous().tick(at=[0, 2, 4]))
        .limit(y=(0, 4.25))
        .layout(size=fs)
    ).plot()
    for ax in p._figure.axes:
        ax.xaxis.set_tick_params(rotation=45)
        gg, duration = ax.get_title().split(" | ")
        if duration == "2.0":
            ax.set_title(
                GENOTYPE_LABELS[gg],
                fontsize=plt.rcParams["axes.labelsize"],
                color=GENOTYPE_COLORS[gg],
            )
        else:
            ax.set_title("")
            ax.set_xlabel("")
        ax.set_ylabel(f"{float(duration):0.0f}s")
    p._figure.supylabel(
        "Resp. rate (breaths/s)",
        fontsize=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
        x=0.1,
    )
    p.save(f"HB_stims{EXT}", dpi=300)


def plot_heart_rate_opto_stims(hr_normed):
    """
    Plot the change in heart rate for the 2s opto stims
    """
    ps = 1
    hr_normed_pivot = hr_normed.groupby(["genotype", "eid"]).mean().reset_index()
    (
        so.Plot(hr_normed_pivot, x="genotype", y="delta_hr_pct", color="genotype")
        # .add(so.Dash(),so.Agg(),so.Dodge())
        .add(so.Bar(baseline=100, width=0.75), so.Agg(), legend=False)
        .add(so.Range(color="k", linewidth=1), so.Est("mean", "se"), legend=False)
        .add(
            so.Dot(alpha=0.75, color=plt.rcParams["text.color"], pointsize=ps),
            so.Jitter(),
            group="eid",
            legend=False,
        )
        .scale(color=GENOTYPE_COLORS)
        .label(y="Heart rate (%)", x="")
        .layout(size=(1.5, 1.5))
        .save(f"heart_rate_percent_change{EXT}", dpi=300)
    )

    p = (
        so.Plot(hr_normed_pivot, x="genotype", y="delta_hr_bpm", color="genotype")
        # .add(so.Dash(),so.Agg(),so.Dodge())
        .add(so.Bar(baseline=0, width=0.75), so.Agg(), legend=False)
        .add(so.Range(color="k", linewidth=1), so.Est("mean", "se"), legend=False)
        .add(
            so.Dot(alpha=0.75, color=plt.rcParams["text.color"], pointsize=ps),
            so.Jitter(),
            group="eid",
            legend=False,
        )
        .scale(color=GENOTYPE_COLORS)
        .label(y=r"$\Delta$ Heart rate (bpm)", x="")
        .layout(size=(2, 1.75))
    ).plot()

    for ax in p._figure.axes:
        ax.set_xticklabels(
            [GENOTYPE_LABELS[x.get_text()] for x in ax.get_xticklabels()],
            rotation=45,
            ha="right",
            va="top",
        )
        ax.axhline(0, color="k", ls="--")
    p
    p.save(f"heart_rate_bpm_change{EXT}", dpi=300)

    stat_rez = pd.DataFrame()
    for gg in hr_normed_pivot.genotype.unique():
        _df = hr_normed_pivot.query("genotype==@gg")

        # Delta percent
        rez = pg.ttest(_df["delta_hr_pct"], 100)
        rez["genotype"] = gg
        rez["metric"] = "delta_hr_pct"
        stat_rez = pd.concat([stat_rez, rez])

        # Delta bpm
        rez = pg.ttest(_df["delta_hr_bpm"], 0)
        rez["genotype"] = gg
        rez["metric"] = "delta_hr_bpm"
        stat_rez = pd.concat([stat_rez, rez])
    stat_rez["reject_null"] = stat_rez["p-val"] < 0.05
    stat_rez.to_csv("heart_rate_stats.csv", index=False)


def plot_all_reset_curves(reset_curves, duty_cycle, duration_reset, fs=(2, 2), ext=EXT):
    """
    Plot the reset curves for all genotypes separately, one plot with the subjects identified and one with them grouped

    Args:
        reset_curves (pd.DataFrame): Dataframe containing the reset curves
    """
    for gg in GENOTYPES:
        _duty_cycle = duty_cycle.query("genotype==@gg")["duty_cycle"].values[0]
        _pk_duty_cycle = duty_cycle.query("genotype==@gg")["peak_duty_cycle"].values[0]

        f = plt.figure(figsize=fs)
        ax = f.add_subplot(aspect=0.6)
        control = reset_curves.query("comparison=='control' & genotype==@gg")
        stim = reset_curves.query("comparison=='stim' & genotype==@gg")

        # sns.scatterplot(control, x="Stim Phase", y="Cycle Duration", color="silver",s=ms)
        sns.scatterplot(
            stim,
            x="Stim Phase",
            y="Cycle Duration",
            color=GENOTYPE_COLORS[gg],
            s=PS,
            legend=False,
        )
        # sns.lineplot(control, x="Stim Phase Bins", y="Cycle Duration", color='silver')
        sns.lineplot(
            stim,
            x="Stim Phase Bins",
            y="Cycle Duration",
            color=GENOTYPE_COLORS[gg],
            legend=False,
        )

        plt.axhline(1, color=plt.rcParams["text.color"], ls=":")
        plt.axvline(_duty_cycle, color="C4")
        plt.axvline(_pk_duty_cycle, color="C4", ls=":")
        plt.xlabel("Stim phase (normalized)")
        plt.ylabel("Cycle duration (normalized)")
        yy = 0, 2
        plt.plot(yy, yy, color=plt.rcParams["text.color"], ls="--")
        plt.xlim(0, 1.5)
        plt.ylim(yy[0], yy[1])
        sns.despine()
        plt.title(f"{gg} ({duration_reset * 1000:0.0f}ms stims)")
        plt.savefig(f"reset_curves_{duration_reset * 1000:0.0f}_{gg}{EXT}")

        # ------------------------------ #
        # Plot 50ms reset curve individual mice
        f = plt.figure(figsize=fs)
        ax = f.add_subplot(aspect=0.6)
        # sns.scatterplot(control, x="Stim Phase", y="Cycle Duration", color="silver",s=ms)
        sns.scatterplot(
            stim, x="Stim Phase", y="Cycle Duration", hue="subject", s=PS, legend=False
        )
        plt.axhline(1, color=plt.rcParams["text.color"], ls=":")
        plt.axvline(_duty_cycle, color="C4")
        plt.axvline(_pk_duty_cycle, color="C4", ls=":")
        plt.xlabel("Stim phase (normalized)")
        plt.ylabel("Cycle duration (normalized)")
        yy = 0, 2
        plt.plot(yy, yy, color=plt.rcParams["text.color"], ls="--")
        plt.xlim(0, 1.5)
        plt.ylim(yy[0], yy[1])
        sns.despine()
        plt.title(f"{gg} ({duration_reset * 1000:0.0f}ms stims)")
        plt.savefig(f"reset_curves_{duration_reset * 1000:0.0f}_subject_{gg}{EXT}")


def plot_reset_curves_all_genotypes(
    reset_curves, duty_cycle, duration_reset, fs=(2.25, 1.9), ext=EXT
):
    """
    Plot the average reset curves with shaded error for all genotypes
    Args:
        reset_curves (pd.DataFrame): Dataframe containing the reset curves
        duty_cycle (pd.DataFrame): Dataframe containing the duty cycle
        fs (tuple): figure size for the plot
        ext (str): file extension for saving the plot
    """
    _duty_cycle = duty_cycle["duty_cycle"].mean()
    _pk_duty_cycle = duty_cycle["peak_duty_cycle"].mean()

    f = plt.figure(figsize=fs)
    sns.lineplot(
        reset_curves.query('comparison=="stim"'),
        x="Stim Phase Bins",
        y="Cycle Duration",
        hue="genotype",
        palette=GENOTYPE_COLORS,
        errorbar="se",
        legend=False,
    )
    # ax = sns.lineplot(
    #     reset_curves.query('comparison=="control"'),
    #     x="Stim Phase Bins",
    #     y="Cycle Duration",
    #     color=plt.rcParams["text.color"],
    #     legend=False,
    # )
    plt.axhline(1, color=plt.rcParams["text.color"], ls=":")
    plt.axvline(_duty_cycle, color="C4")
    plt.axvline(_pk_duty_cycle, color="C4", ls=":")
    plt.text(
        _pk_duty_cycle,
        2,
        "Dia. peak",
        ha="right",
        va="top",
        fontsize="xx-small",
        rotation=90,
    )
    plt.text(
        _duty_cycle,
        2,
        "Dia. end",
        fontsize="xx-small",
        ha="right",
        va="top",
        rotation=90,
    )
    plt.xlabel("Stim phase (normalized)")
    plt.ylabel("Cycle duration (normalized)")
    yy = 0, 2
    plt.plot(yy, yy, color=plt.rcParams["text.color"], ls="--")
    plt.xlim(0, 1.5)
    plt.ylim(yy[0], yy[1])
    sns.despine()
    # sns.move_legend(ax,loc='upper left',bbox_to_anchor=(1,1))
    plt.title(f"Phase reset (random {duration_reset * 1000:0.0f}ms stims)")
    plt.savefig(f"reset_curves_all_genotypes{duration_reset * 1000:0.0f}{EXT}")
    plt.close("all")


def plot_all_HB():
    f = plt.figure(figsize=(2, 1.5))
    # compute histogram to get means (PETH)
    all_peth2_means = []
    all_peth5_means = []
    eids = EIDS_PHYSIOL
    for eid in eids:
        rec = Rec(one, eid, load_spikes=False)
        breaths = rec.breaths
        intervals, stims = rec.get_HB_stims(duration=5)
        if stims.shape[0] == 0:
            continue

        peth5, binned = calculate_peths(
            breaths.times,
            np.ones_like(breaths.times),
            [1],
            stims["start_time"],
            smoothing=0.25,
            pre_time=10,
            post_time=15,
            bin_size=0.25,
        )
        all_peth5_means.append(peth5.means[0])

        intervals, stims = rec.get_HB_stims(duration=2)
        peth2, binned = calculate_peths(
            breaths.times,
            np.ones_like(breaths.times),
            [1],
            stims["start_time"],
            smoothing=0.25,
            pre_time=10,
            post_time=15,
            bin_size=0.25,
        )
        all_peth2_means.append(peth2.means[0])

    all_peth_means = np.array(all_peth5_means)
    m = all_peth_means.mean(0)
    lb = m - np.std(all_peth_means) / np.sqrt(all_peth_means.shape[0])
    ub = m + np.std(all_peth_means) / np.sqrt(all_peth_means.shape[0])
    plt.plot(peth5.tscale, m, color=HB_MAP[5])
    plt.fill_between(peth5.tscale, lb, ub, color=HB_MAP[5], lw=0, alpha=0.25)

    all_peth_means = np.array(all_peth2_means)
    m = all_peth_means.mean(0)
    lb = m - np.std(all_peth_means) / np.sqrt(all_peth_means.shape[0])
    ub = m + np.std(all_peth_means) / np.sqrt(all_peth_means.shape[0])
    plt.plot(peth2.tscale, m, color=HB_MAP[2])
    plt.fill_between(peth5.tscale, lb, ub, color=HB_MAP[2], lw=0, alpha=0.25)
    plt.hlines(3.8, 0, 2, color=HB_MAP[2], lw=4)
    plt.hlines(3.9, 0, 5, color=HB_MAP[5], lw=4)
    plt.ylabel("Breaths/s")
    plt.xlabel("Time (s)")
    plt.ylim(0, 4)
    plt.xlim(-5, 15)
    plt.axvline(0, color=plt.rcParams["text.color"], ls=":")
    sns.despine(trim=True)
    plt.savefig(f"HB_example_peth{EXT}")
    plt.close("all")


def plot_resp_rate_vs_fiber_position(fs=(4, 1.65), ext="pdf"):
    """
    Plot the respiratory rate change vs fiber position for the 2s opto stims

    Args:
        fs (tuple, optional): Figure size. Defaults to (4, 1.5).
        ext (str, optional): File extension for saving the plot. Defaults to 'pdf'.
    """

    # Load data
    df = pd.read_csv("2s_opto_stims_raw.csv").merge(
        sessions_to_include[["eid", "fiber_position_um"]]
    )

    # Process data
    rr_by_fiber = pd.pivot_table(
        df,
        index=["eid", "genotype", "fiber_position_um"],
        values=["resp_rate"],
        columns=["comparison", "trial"],
    )["resp_rate"]
    
    delta_rr = 100 * (rr_by_fiber["stim"] - rr_by_fiber["control"]) / rr_by_fiber["control"]
    delta_rr = (
        delta_rr.melt(ignore_index=False)
        .reset_index()
        .rename(
            {
                "fiber_position_um": "Fiber Position ($\mu$m from VII)",
                "value": "Resp. Rate (% diff.)",
            },
            axis=1,
        )
        .query('genotype!="vgatcre_ntschrmine"')
    )

    # Make plot
    p = (
        so.Plot(
            data=delta_rr,
            x="Fiber Position ($\mu$m from VII)",
            y="Resp. Rate (% diff.)",
            color="genotype",
        )
        .facet(col="genotype", order=genotype_order[:-1])
        .add(
            so.Range(alpha=0.5, linewidth=2),
            so.Est("mean", "se"),
            group="eid",
            legend=False,
        )
        .add(so.Dash(linewidth=1, width=2), so.Agg(), group="eid", legend=False)
        .add(so.Dots(pointsize=PS, alpha=0.2), so.Jitter(), legend=False)
        .scale(color=GENOTYPE_COLORS)
        .label(x="")
        .limit(y=(-110, None))
        .layout(size=fs)
    ).plot()

    for ax in p._figure.axes:
        ax.set_title(
            GENOTYPE_LABELS[ax.get_title()],
            fontsize=plt.rcParams["axes.labelsize"],
            color=GENOTYPE_COLORS[ax.get_title()],
        )
        ax.axhline(0, lw=0.5, ls=":", color=plt.rcParams["text.color"])
    p._figure.supxlabel(
        "Fiber Position ($\mu$m from VII)",
        y=0.1,
        color=plt.rcParams["text.color"],
        fontsize=plt.rcParams["axes.labelsize"],
    )
    p
    p.save(f"2s__vs_fiber_pos{ext}", dpi=300)

    # Stats
    for gg in ['vglut2ai32','vgatai32']:
        model = smf.mixedlm(
            'Q("Resp. Rate (% diff.)") ~ Q("Fiber Position ($\mu$m from VII)")',
            data=delta_rr.query('genotype==@gg').dropna(axis=0),
            groups='eid'
        )
        result = model.fit().summary()
        with open(f"resp_rate_vs_fiber_pos_{gg}.txt", "w") as f:
            f.write(result.as_text())


def make_all_summary_plots():
    """
    Make all the plots for the paper
    """
    # Load data
    long_opto_pivot = pd.read_csv("2s_opto_stims.csv")
    phasic_opto_pivot = pd.read_csv("phasic_opto_stims.csv")
    phasic_opto_normed = pd.read_csv("phasic_opto_normed.csv")
    hr_normed = pd.read_csv("heartrate_delta.csv")
    reset_curve_intercept = pd.read_csv("reset_intercept.csv")
    phasic_amps_pivot = pd.read_csv("phasic_amps.csv")

    HB_pivot = pd.read_csv("HB_stims.csv")
    reset_curves = pd.read_csv("reset_curves.csv")
    duty_cycle = pd.read_csv("duty_cycle.csv", index_col=0)
    duration_reset = reset_curves["duration_pulse"].mean()

    reset_curve_intercept = process_intercept(reset_curve_intercept)
    reset_curve_intercept = add_laterality(reset_curve_intercept, sessions_to_include)
    long_opto_pivot = add_laterality(long_opto_pivot, sessions_to_include)
    phasic_opto_pivot = add_laterality(phasic_opto_pivot, sessions_to_include)
    phasic_opto_normed = add_laterality(phasic_opto_normed, sessions_to_include)
    phasic_amps_pivot = add_laterality(phasic_amps_pivot, sessions_to_include)

    plot_opto_holds(long_opto_pivot, fs=(2, 1.65), ext=EXT)
    plot_opto_phasic(phasic_opto_pivot, fs=(2, 1.65), ext=EXT)
    plot_opto_phasic_normalized(phasic_opto_normed, fs=(2.25, 1.85), ext=EXT)
    plot_opto_phasic_amps(phasic_amps_pivot, fs=(2.25, 1.85), ext=EXT)
    plot_plot_opto_phasic_normalized_talk(phasic_opto_normed, fs=(3, 3), ext=EXT)
    plot_hering_breuer(HB_pivot, fs=(2.5, 2), ext=EXT)
    plot_heart_rate_opto_stims(hr_normed)
    plot_all_HB()
    plot_all_reset_curves(
        reset_curves, duty_cycle, duration_reset, fs=(2, 1.5), ext=EXT
    )
    plot_reset_curves_all_genotypes(
        reset_curves, duty_cycle, duration_reset, fs=(2, 1.5), ext=EXT
    )
    plot_reset_curve_intercept(reset_curve_intercept, ps=PS, ext=EXT, fs=(1.5, 1.75))

    plot_rebound()
    plot_resp_rate_vs_fiber_position(fs=(4, 1.65), ext=EXT)





class Rec(Rec):
    def plot_HB_example(self):
        figsize = (2, 1)
        log = self.log
        HB = log.to_df().query('label=="hering_breuer"& duration==5')
        physiology = self.physiology
        f = plt.figure(figsize=figsize, constrained_layout=True)
        ax = f.add_subplot()
        plot_sweeps(
            physiology.times,
            physiology.dia,
            HB["start_time"].values[0:1],
            5,
            15,
            ax=ax,
            color=plt.rcParams["text.color"],
            lw=0.5,
        )
        ax.set_xlim(-5, 15)
        ax.set_ylim(-0.55, None)
        plt.hlines(ax.get_ylim()[1] * 0.9, 0, 5, color=HB_MAP[5], lw=2)
        plt.axis("off")
        plt.hlines(-0.5, -2, 0, color=plt.rcParams["text.color"], lw=1)
        plt.text(-2, -0.5, "2s", va="top", ha="left")
        plt.savefig(f"HB_example_{self.subject}_{self.genotype}{EXT}")
        plt.close("all")

    def plot_example_phasic_opto(self):
        physiology = self.physiology
        t_insp = self.log.to_df().query('phase=="insp"')["start_time"].iloc[3]
        t_exp = (
            self.log.to_df().query('phase=="exp" & mode=="hold"')["start_time"].iloc[0]
        )

        f, ax = plt.subplots(
            nrows=2, sharey=True, sharex=True, figsize=(2, 1.5), constrained_layout=True
        )
        t0 = np.array([t_insp, t_exp]) - 3
        tf = (
            np.array([t_insp, t_exp]) + 3.75
        )  # Choosing this because there is a sigh in almost all of the Vglut2 stims and this throws off the ylim
        for ii, (_t0, _tf) in enumerate(zip(t0, tf)):
            # t0, tf = [t_insp - 5, t_insp + 15]
            s0, sf = np.searchsorted(physiology.times, [_t0, _tf])
            idx = np.logical_and(
                self.laser["intervals"][:, 0] >= _t0,
                self.laser["intervals"][:, 1] <= _tf,
            )
            laser_times = self.laser.intervals[idx, :]
            ax[ii].plot(
                physiology.times[s0:sf] - _t0,
                physiology.dia[s0:sf],
                color=plt.rcParams["text.color"],
            )
            plot_laser(laser_times - _t0, ax=ax[ii], color=self.laser_color)
            ax[ii].axis("off")
            ax[ii].text(
                0,
                0,
                r"$\int{Dia}$",
                va="bottom",
                ha="right",
                fontsize=plt.rcParams["axes.labelsize"],
                rotation=90,
            )
            ax[ii].set_ylim(-1, None)
        ax[1].hlines(-0.7, 0, 2, color=plt.rcParams["text.color"], lw=1)
        ax[1].text(0, -0.8, "2s", va="top", ha="left", fontsize="small")
        ax[0].text(
            0,
            1,
            GENOTYPE_LABELS[self.genotype],
            transform=ax[0].transAxes,
            va="top",
            ha="left",
            fontsize="small",
            color=GENOTYPE_COLORS[self.genotype],
        )

        plt.savefig(f"phasic_example_{self.subject}_{self.genotype}{EXT}")
        plt.close("all")

    def plot_example_opto_hold(self):
        physiology = self.physiology
        t_holds = (
            self.log.to_df()
            .query('label=="opto_pulse" & duration==2')["start_time"]
            .values[0]
        )

        f = plt.figure(figsize=(2, 0.5))
        ax = f.add_subplot()
        t0, tf = [t_holds - 3, t_holds + 6]
        s0, sf = np.searchsorted(physiology.times, [t0, tf])
        ax.plot(
            physiology.times[s0:sf],
            physiology.dia[s0:sf],
            color=plt.rcParams["text.color"],
            lw=0.5,
        )
        plot_laser(self.laser, ax=ax, color=self.laser_color)
        ax.set_xlim(t0, tf)
        ax.set_ylim(0, 5)
        ax.hlines(-1, t0, t0 + 2, color=plt.rcParams["text.color"], lw=1)
        ax.axis("off")

        plt.savefig(f"2s_stim_example_{self.subject}_{self.genotype}{EXT}")
        plt.close("all")

    def plot_long_stims_all_physiol(self):
        """
        Plot diaphragm and heart rate for the long opto holds (2s)
        """
        physiology = self.physiology.copy()
        t_holds = (
            self.log.to_df()
            .query('label=="opto_pulse" & duration==2')["start_time"]
            .values[0]
        )

        f, ax = plt.subplots(nrows=2, ncols=1, figsize=(3, 1.5))
        t0, tf = [t_holds - 5, t_holds + 95]
        s0, sf = np.searchsorted(physiology.times, [t0, tf])
        for ii, vv in enumerate(["dia", "hr_bpm"]):
            ax[ii].plot(
                physiology.times[s0:sf],
                physiology[vv][s0:sf],
                color=plt.rcParams["text.color"],
                lw=0.5,
            )
            ax[ii].set_xlim(t0, tf)
            plot_laser(self.laser, ax=ax[ii])

        ax[0].axis("off")
        ax[0].set_ylabel("$\\int$Dia")
        ax[1].set_ylabel("HR (bpm)")

        plt.tight_layout()
        plt.savefig(f"2s_stim_example_all_physiol_{self.subject}_{self.genotype}{EXT}")
        plt.close("all")

    def plot_post_stim_traces(self):
        physiology = self.physiology
        log = self.log
        long_pulses = log.to_df().query('duration==2.0 & label=="opto_pulse"')

        f = plt.figure(figsize=(2, 1))
        ax = f.add_subplot()
        pre = 0.2
        post = 0.5
        plot_sweeps(
            physiology.times,
            physiology.dia,
            long_pulses["end_time"].values,
            pre,
            post,
            ax=ax,
            color=plt.rcParams["text.color"],
            lw=0.5,
        )
        ax.text(
            0,
            1,
            GENOTYPE_LABELS[self.genotype],
            va="bottom",
            ha="left",
            fontsize="x-small",
            color=GENOTYPE_COLORS[self.genotype],
            transform=ax.transAxes,
        )
        ax.axvspan(-pre, 0, ymin=0.15, color=self.laser_color, alpha=0.4, lw=0)
        ax.set_xlim(-pre, post)
        ax.set_ylim(-1, 5)
        ax.axis("off")
        ax.hlines(-pre, -pre + 0.05, -0.5, color=plt.rcParams["text.color"], lw=1)
        ax.text(-pre, -0.5 * 1.1, "50ms", va="top", ha="left", fontsize="x-small")
        plt.savefig(f"rebound_{self.subject}_{self.genotype}{EXT}")


# -------------------------
# RUN PLOTS #
# -------------------------

make_all_summary_plots()

# Make example figs
eids = EIDS_PHYSIOL
# Make Example phasic plots
example_subjects = [
    "m2024-29",
    "m2024-40",
    "m2024-36",
    "m2024-59",
    "m2024-60",
    "m2025-01",
]
example_sequences = [0, 1, 0, 1, 0, 0]
for subject, sequence in zip(example_subjects, example_sequences):
    eid = one.search(subject=subject, number=sequence)[0]
    rec = Rec(one, eid, load_spikes=False)
    rec.plot_example_phasic_opto()
    rec.plot_example_opto_hold()
    rec.plot_long_stims_all_physiol()
    rec.plot_HB_example()
    rec.plot_post_stim_traces()
