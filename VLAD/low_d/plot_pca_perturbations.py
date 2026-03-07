"""
Plot summary and statistical data that comes from "compute_pca_perturbations.py"
"""

import sys

sys.path.append("../")
sys.path.append("VLAD/")

import cibrrig.plot as cbp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import seaborn.objects as so
from pathlib import Path
from utils import GENOTYPE_COLORS, GENOTYPE_LABELS, PHASE_MAP,set_style,sig2star

# Styling
set_style()
ext = ".pdf"
FIGSIZE = (2.5, 1.75)
PS = 3
STIM_MAP = {"50ms": "Pulse", "hb": "HB","random": "Control"}
ORDER = ['vglut2ai32','vgatai32','vgatcre_ntschrmine']
STATS_DEST = Path('./stats')
STATS_DEST.mkdir(exist_ok=True)
# ========================
# PLOTTING MODIFIERS
# ========================
def modify_facet(p, ticklabels=None, rotate=45):
    rotate = rotate if rotate is not None else 0

    for ax in p._figure.axes:
        if ticklabels is None:
            ticklabels = ax.get_xticklabels()
            ticklabels = [x.get_text().capitalize() for x in ticklabels]
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ticklabels, rotation=rotate, ha="right", va="top")
        try:
            ax.set_title(
                GENOTYPE_LABELS[ax.get_title()], color=GENOTYPE_COLORS[ax.get_title()]
            )
        except KeyError:
            pass
    return p

def modify_multi_facet(p, ticklabels=None, rotate=45):
    rotate = rotate or 0

    for ii, ax in enumerate(p._figure.axes):
        stim, gg = ax.get_title().split(" | ")

        if ticklabels is None:
            ticklabels = ax.get_xticklabels()
            ticklabels = [x.get_text().capitalize() for x in ticklabels]
        ax.set_xticklabels(ticklabels, rotation=rotate, ha="right", va="top")
        if ii < 3:
            ax.set_title(STIM_MAP[stim], color=plt.rcParams["axes.labelcolor"])
        else:
            ax.set_title("")
        if ii % 3 == 0:
            ax.set_ylabel(GENOTYPE_LABELS[gg], color=GENOTYPE_COLORS[gg])
    return p


# ========================
# LOAD DATA
# ========================
def load_data():
    """ 
    Load data from CSV files.
    """
    df_D = pd.read_csv("median_distances.csv")

    hold_stim_pca = pd.read_csv("hold_stim_pca.csv")
    speed_rez = pd.read_csv("speed_by_phase.csv")
    speed_rez.set_index(["genotype", "subject", "sequence", "eid"], inplace=True)
    closest_phi = pd.read_csv("closest_phi.csv")

    # Includes gridsearch over post-stim time and ndims
    closest_phi_full = pd.read_csv("closest_phi_full.csv")
    rez_full = pd.read_csv("distance_results_full.csv")

    # Convert post_time to ms
    rez_full["post_time"] = 1000 * rez_full["post_time"]
    # ========================
    # Compare to Hering breuer location
    # ========================
    return df_D,speed_rez, closest_phi, closest_phi_full, rez_full,hold_stim_pca


# ========================
# PLOTTING
# ========================

def plot_hold_stims_speeds(fs=(2.5,1.5)):
    df = pd.read_csv("hold_stim_speeds.csv")
    df = pd.pivot_table(
        data=df, index=["genotype","subject",'sequence','eid','phase'], columns="condition", values="speed"
    )
    df['delta_speed'] = (df['stim_speed'] - df['control_speed']) / df['control_speed'] * 100
    
    p = (
        so.Plot(df, x="phase", y="delta_speed", color="genotype")
        .add(so.Line(), so.Agg('mean'),legend=False)
        .add(so.Band(), so.Est("mean", "se"),legend=False)
        .layout(size=fs,engine='constrained')
        .scale(
            y=so.Continuous().tick(every=50),
            x=so.Continuous().tick(at=[-np.pi, 0, np.pi]),
            color = GENOTYPE_COLORS
        )
        .label(x=r'Phase (rads)',y=r'$\Delta$ Speed (% diff.)',title='2s stimulations')
        .limit(x=(-np.pi,np.pi))
    ).plot()
    p=modify_facet(p, ticklabels=[r"$-\pi$", "0", r"$\pi$"],rotate=0)
    ax = p._figure.axes[0]
    ax.axhline(0, color="k", ls=":")
    ax.axvline(0, color="k", ls=":")
    ax.hlines(0.95,0,0.5,transform=ax.transAxes, color=PHASE_MAP['exp'])
    ax.hlines(0.95,0.5,1,transform=ax.transAxes, color=PHASE_MAP['insp'])
    ax.text(0.25, 0.95, "E", va="bottom", color=PHASE_MAP["exp"],transform=ax.transAxes,fontsize='x-small')
    ax.text(0.75, 0.95, "I", va="bottom", color=PHASE_MAP["insp"],transform=ax.transAxes,fontsize='x-small')
    for ii,gg in enumerate(ORDER):
        ax.text(0.6, 0.82-ii*0.07, GENOTYPE_LABELS[gg], va="bottom", color=GENOTYPE_COLORS[gg],transform=ax.transAxes,fontsize='xx-small')
    p.save(f"hold_stim_speeds_phasic.pdf")

def plot_distances_to_hb_2d(df_D):
    """
    Plot distances to Hering Breuer for 2D projection at 25ms post-stimulus time.

    Args:
        df_D_melted (pd.DataFrame): DataFrame containing the melted distances to Hering Breuer.
    """
    df = df_D.query('ndims==3 & comparison!="hold_control"')
    p = (
        so.Plot(
            df, x="comparison", y="distance", color="genotype"
        )
        .add(so.Bar(width=0.5), so.Agg(), alpha="comparison", legend=False)
        .facet(col="genotype",order=ORDER)
        .add(so.Range(color="k"), so.Est("mean", "se"), legend=False)
        .add(so.Line(alpha=0.5, linewidth=0.5), so.Shift(0.35), group="eid", legend=False)
        .add(
            so.Dot(color="w", edgewidth=0.5, edgecolor="k", pointsize=PS),
            so.Shift(0.35),
            legend=False,
        )
        .scale(
            color=GENOTYPE_COLORS,
            x=so.Nominal(order=["hb_hold", "hb_control"]),
            y=so.Continuous().tick(count=3),
            alpha=[0.25, 1],
        )
        .layout(size=FIGSIZE)
        .label(x="", y="Distance to HB (a.u.)")
    ).plot()
    p = modify_facet(p)
    p.save(f"distance_to_hb_2d{ext}")

def plot_distances_to_hb_dimsweep(df_D):
    """
    Plot distances to Hering Breuer for all genotypes and dimensions.
    At 25ms post-stimulus time.

    Args:
        df_D_melted (pd.DataFrame): DataFrame containing the melted distances to Hering Breuer.
    """
    df = df_D.copy().query('comparison!="hold_control"')
    p = (
        so.Plot(
            df,
            x="ndims",
            y="distance",
            color="genotype",
            linestyle="comparison",
        )
        .add(so.Line(), so.Agg())
        .add(so.Band(), so.Est("mean", "se"), group="comparison", legend=False)
        .facet(col="genotype",order=ORDER)
        .layout(size=FIGSIZE)
        .limit(x=(2, 10), y=(0, None))
        .scale(
            color=GENOTYPE_COLORS,
            y=so.Continuous().tick(count=3),
            x=so.Continuous().tick(at=[2, 4, 6, 8, 10]),
        )
        .label(x="# Dims", y="Distance to HB (a.u.)")
    ).plot()
    p = modify_facet(p, rotate=False)
    p.save(f"distance_to_hb_all_dims{ext}")

def plot_metrics_by_genotype_2d_25ms(rez, depvar):
    """
    Plot metrics by genotype for 2D projection at 25ms post-stimulus time.
    only for 50ms and random conditions.

    Saves the plot as a PDF file.

    Args:
        rez (pd.DataFrame): DataFrame containing the results.
        depvar (str): The dependent variable to plot. Can be either "exp_distance" or "dispersion".

    """
    ylabel = "Distance to I-off" if depvar == "exp_distance" else "Dispersion"

    df = rez.query('ndims==2 & post_time==25 & condition!="hb"')
    df = df.copy()
    df['condition'] = df['condition'].map(STIM_MAP)
    p = (
        so.Plot(
            df,
            x="condition",
            y=depvar,
            color="genotype",
        )
        .add(so.Bar(width=0.5), so.Agg(), alpha="condition", legend=False)
        .facet(col="genotype",order=ORDER)
        .add(so.Range(color="k"), so.Est("mean", "se"), legend=False)
        .add(
            so.Line(alpha=0.5, linewidth=0.5), so.Shift(0.35), group="eid", legend=False
        )
        .add(
            so.Dot(color="w", edgewidth=0.5, edgecolor="k", pointsize=PS),
            so.Shift(0.35),
            legend=False,
        )
        .scale(
            color=GENOTYPE_COLORS,
            x=so.Nominal(order=["Control", "Pulse"]),
            alpha=[0.25, 1],
            y=so.Continuous().tick(count=3),
        )
        .layout(size=FIGSIZE)
        .label(x="", y=ylabel)
    ).plot()
    p = modify_facet(p)
    p
    p.save(f"{depvar}_by_condition_by_genotype_50ms_2d{ext}")

def plot_metrics_by_genotype_dimsweep(rez, depvar):
    """
    Plot metrics by genotype for all dimensions at 25ms post-stimulus time.
    only for 50ms and random conditions.
    
    Saves the plot as a PDF file.
    
    Args:
        rez (pd.DataFrame): DataFrame containing the results.
        depvar (str): The dependent variable to plot. Can be either "exp_distance" or "dispersion".

    """
    ylabel = "Distance to I-off" if depvar == "exp_distance" else "Dispersion"
    df = rez.query('post_time==25 & condition!="hb"')
    p = (
        so.Plot(df, x="ndims", y=depvar, color="genotype", linestyle="condition")
        .add(so.Line(), so.Agg())
        .add(so.Band(), so.Est("mean", "se"), group="condition")
        .facet(col="genotype",order=ORDER)
        .layout(size=FIGSIZE)
        .limit(x=(2, 10), y=(0, None))
        .scale(
            color=GENOTYPE_COLORS,
            y=so.Continuous().tick(count=3),
            x=so.Continuous().tick(at=[2, 4, 6, 8, 10]),
        )
        .label(x="", y=ylabel)
    ).plot()
    p = modify_facet(p, rotate=False)
    p._figure.supxlabel(
        "# Dims",
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
        y=0.2
    )
    p.save(f"{depvar}_all_dims_poststim_25ms{ext}")

def plot_metrics_by_genotype_2d_timesweep(rez, depvar):
    """
    Plot metrics by genotype for 2D projection at all post-stimulus times.
    only for 50ms and random conditions.
    
    Saves the plot as a PDF file.
    
    Args:
        rez (pd.DataFrame): DataFrame containing the results.
        depvar (str): The dependent variable to plot. Can be either "exp_distance" or "dispersion".

    """
    ylabel = "Distance to I-off" if depvar == "exp_distance" else "Dispersion"
    df = rez.query('ndims==2 & condition!="hb"')
    p = (
        so.Plot(
            df,
            x="post_time",
            y=depvar,
            color="genotype",
            linestyle="condition",
        )
        .add(so.Line(), so.Agg(),legend=False)
        .add(so.Band(), so.Est("mean", "se"), group="condition",legend=False)
        .facet(col="genotype",order=ORDER)
        .layout(size=(2.75,1.75))
        .limit(x=(0, 100), y=(0, None))
        .scale(
            color=GENOTYPE_COLORS,
            y=so.Continuous().tick(count=3),
            x=so.Continuous().tick(at=[0, 50, 100]),
        )
        .label(x="", y=ylabel)
    ).plot()
    p = modify_facet(p, rotate=False)
    p._figure.supxlabel(
        "Post-stim time (ms)",
        ha="center",
        va="top",
        fontsize=plt.rcParams["axes.labelsize"],
        color=plt.rcParams["axes.labelcolor"],
        y=0.2
    )
    for ax in p._figure.axes:
        ax.axvline(25, color="k", ls=":")
    p.save(f"{depvar}_all_post_time_2d{ext}")

def plot_metrics_gridsearch(rez):
    df = rez.copy()
    fs = (5, 1.5)
    var_label = {"dispersion": "Dispersion", "exp_distance": "Distance to I-off"}
    cmap = {"dispersion": "cool", "exp_distance": "magma"}
    conditions = ['50ms']
    cond = '50ms'
    for depvar in ["dispersion", "exp_distance"]:
        f, ax = plt.subplots(len(conditions), 3, sharex=True, sharey=True, figsize=fs,constrained_layout=True)
        # Get min and max of all data for a given depvar
        all_var_data = df[depvar]
        vlims = [0, np.percentile(df[depvar], 95)]
        for ii, gg in enumerate(ORDER):
            temp = df.query("genotype==@gg and condition==@cond")
            temp = pd.pivot_table(
                data=temp, index="ndims", columns="post_time", values=depvar
            )
            # sns.heatmap(temp,ax=ax[rr,ii],cbar=False,vmin=0,vmax=1,xticklabels=4,yticklabels=4,fmt='.0f')
            ax[ii].pcolormesh(
                temp.columns,
                temp.index,
                temp.values,
                vmin=vlims[0],
                vmax=vlims[1],
                cmap=cmap[depvar],
            )
            # Plot the minimum values across all rows
            min_time = temp.columns[np.argmin(temp.values, axis=1)]

            ax[ii].plot(min_time, temp.index, "w.-", lw=0.5)
            ax[ii].set_title(GENOTYPE_LABELS[gg], color=GENOTYPE_COLORS[gg])
            ax[ii].set_yticks([2, 5, 10])
            if ii == 1:
                ax[ii].set_xlabel("Post-stim time (ms)")
            else:
                ax[ii].set_xlabel("")
            ax[ii].set_ylabel("")
        ax[0].set_ylabel('$n_{dims}$')
        ax[ii].set_xlim([0, 100])
        ax[ii].set_ylim(2, 10)
        ax[ii].invert_yaxis()

        # Add a colorbar outside the rightmost subplot
        cb = plt.colorbar(ax[2].collections[0], ax=ax, orientation="vertical")
        # Remove lines and labels
        cb.outline.set_visible(False)
        cb.set_label(var_label[depvar])
        plt.savefig(f"{depvar}_gridsearch_post_time_ndims{ext}")

def plot_metrics(rez_full, df_D):
    plot_distances_to_hb_dimsweep(df_D)
    plot_distances_to_hb_2d(df_D)
    plot_metrics_gridsearch(rez_full)
    for depvar in ["exp_distance", "dispersion"]:
        plot_metrics_by_genotype_2d_25ms(rez_full, depvar)
        plot_metrics_by_genotype_dimsweep(rez_full, depvar)
        plot_metrics_by_genotype_2d_timesweep(rez_full, depvar)

def plot_phasic_stim_speeds(speed_rez):
    """
    Plot speed by phase for each genotype for inspiratory and expiratory stims
    """
    genotypes = set(x[0] for x in speed_rez.index[:])
    fs = (FIGSIZE[0] / 2, FIGSIZE[1] / 2)
    for gg in genotypes:
        aa = speed_rez.loc[gg, :, :].reset_index()
        f = plt.figure(figsize=fs)
        ax = f.add_subplot(111)
        sns.lineplot(aa, x="phase", y="insp_speed_pct_diff", color=PHASE_MAP["insp"])
        sns.lineplot(aa, x="phase", y="exp_speed_pct_diff", color=PHASE_MAP["exp"])
        ax.axhline(0, color=plt.rcParams["text.color"], ls=":")
        ax.axvline(0, color=plt.rcParams["text.color"], ls=":")
        ax.set_ylim(-200, 400)
        cbp.clean_linear_radial_axis(ax)
        plt.ylabel(r"$\Delta$ speed (% diff.)")
        plt.xlabel("Phase (rads)")
        ymax = ax.get_ylim()[1]
        plt.hlines(ymax * 0.8, -np.pi, 0, color=PHASE_MAP["exp"])
        plt.hlines(ymax * 0.8, 0, np.pi, color=PHASE_MAP["insp"])
        plt.text(-np.pi / 2, ymax * 0.8, "E", va="bottom", color=PHASE_MAP["exp"])
        plt.text(np.pi / 2, ymax * 0.8, "I", va="bottom", color=PHASE_MAP["insp"])
        sns.despine(trim=True)
        ax.set_title(GENOTYPE_LABELS[gg], color=GENOTYPE_COLORS[gg])
        plt.savefig(f"delta_speed_by_phase_{gg}{ext}")

def plot_closest_phi_to_stim(closest_phi_full):
    """
    Plot histograms of the nearest phase on the eupnic cycle that is closes to the post-stim
    phase for each genotype and condition.

    """
    order_dict = {'col': ['hb','50ms','random'],
                'row':ORDER}
    df_use = closest_phi_full.query('ndims==2').copy()
    
    idx = df_use.query('condition=="hb"').index
    df_use.loc[idx,'post_time']-=df_use.query('condition=="hb"')['post_time'].min()

    p = (
        so.Plot(df_use.query("ndims==2 & post_time<0.026 & post_time>0.024"), "closest_phi", color="genotype")
        .facet(row="genotype", col="condition",order=order_dict)
        .add(
            so.Bars(edgewidth=0, alpha=0.75, edgecolor="k"),
            so.Hist(common_norm=False, stat="percent", bins=20),
            legend=False,
        )
        .scale(
            color=GENOTYPE_COLORS,
            x=so.Continuous().tick(at=[-np.pi, 0, np.pi]).label(like="0.2f"),
        )
        .layout(size=(3.5,3))
        .limit(x=(-np.pi, np.pi))
        .label(x="", y="")
    ).plot()
    p = modify_multi_facet(p,ticklabels=[r"$-\pi$", "0", r"$\pi$"],rotate=0)
    for ax in p._figure.axes:
        ax.axvline(0, color="k", ls=":")
    p._figure.supxlabel(r"Closest $\phi$ to post-stim", ha="center", va="bottom", fontsize=plt.rcParams["axes.labelsize"], color=plt.rcParams["axes.labelcolor"],y=0.05)
    p._figure.supylabel("Percent of stims", ha="right", va="center", fontsize=plt.rcParams["axes.labelsize"], color=plt.rcParams["axes.labelcolor"],x=0.05)
    p.save(f"closest_phi_to_post_stim{ext}")

# ========================
# STATS
# ========================

def run_metric_stats(rez_full,ndims=2,post_time=25):
    """
    Run ANOVA and pairwise tests for each genotype and dependent variable.
    """
    stats_rez_full = pd.DataFrame()
    genotypes = rez_full["genotype"].unique()
    for depvar in ["exp_distance", "dispersion"]:
        for genotype in genotypes:
            test_df = rez_full.query("ndims==@ndims and genotype==@genotype & post_time==@post_time")
            stats_rez = pg.ttest(
                test_df.query("condition=='50ms'")[depvar],
                test_df.query("condition=='random'")[depvar],
                paired=True,
            )

            stats_rez["genotype"] = genotype
            stats_rez["depvar"] = depvar
            stats_rez["significant"] = stats_rez["p_val"] < 0.05
            stats_rez["sig_string"] = stats_rez["p_val"].apply(sig2star)

            stats_rez_full = pd.concat([stats_rez_full, stats_rez])

    fn = STATS_DEST.joinpath(f"ttest_distance_metrics.csv")
    stats_rez_full.to_csv(fn,index=False)

def run_HB_distance_stats(df_D):
    """
    Run ANOVA for distances to Hering Breuer for 2D projection at 25ms post-stimulus time.
    """
    df = df_D.query("ndims==2 & comparison!='hold_control'")
    mdl = pg.mixed_anova(df, dv="distance", 
                         between='genotype',
                         within="comparison", subject="eid")
    mdl["significant"] = mdl["p_unc"] < 0.05
    mdl["sig_string"] = mdl["p_unc"].apply(sig2star)

    fn = STATS_DEST.joinpath("anova_distance_to_hb.csv")
    mdl.to_csv(fn)

    wilcoxon_full = pd.DataFrame()
    for genotype in df["genotype"].unique():
        _df = df.query("genotype==@genotype")
        x = _df.query("comparison=='hb_hold'")["distance"]
        y = _df.query("comparison=='hb_control'")["distance"]
        wilcoxon = pg.wilcoxon(
            x,y)
        wilcoxon['A'] = '50ms'
        wilcoxon['B'] = 'random'
        wilcoxon["genotype"] = genotype
        wilcoxon["significant"] = wilcoxon["p_val"] < 0.05
        wilcoxon["sig_string"] = wilcoxon["p_val"].apply(sig2star)
        wilcoxon_full = pd.concat([wilcoxon_full, wilcoxon])
    fn = STATS_DEST.joinpath("wilcoxon_distance_to_hb.csv")
    wilcoxon_full.to_csv(fn, index=False)

# ========================
# Run
# ========================
if __name__ == "__main__":
    df_D, speed_rez, closest_phi, closest_phi_full, rez_full,hold_stim_pca = load_data()
    plot_metrics(rez_full, df_D)
    plot_phasic_stim_speeds(speed_rez)
    plot_closest_phi_to_stim(closest_phi_full)

    run_metric_stats(rez_full)
    run_HB_distance_stats(df_D)
    plot_hold_stims_speeds()



    p = so.Plot(
        hold_stim_pca,x=0,y=1,color='genotype'
    )