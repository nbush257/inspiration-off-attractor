import sys
import gc

sys.path.append("../")
sys.path.append("VLAD/")
from pathlib import Path
import logging

import brainbox.plot as bbp
import cibrrig.plot as cbp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import coherence
from cibrrig.analysis.population import Population
from cibrrig.utils.utils import get_eta, weighted_histogram
from scipy.spatial.distance import cdist
from sklearn.metrics import DistanceMetric
from utils import (
    EIDS_NEURAL,
    HB_MAP,
    Rec,
    one,
)

plt.style.use("../VLAD.mplstyle")
from itertools import product,combinations

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

NDIMS = 5
EXAMPLE_PATH = Path("example_figs")
EXAMPLE_PATH.mkdir(exist_ok=True)



def plot_lowd_stim_response(rec, pop, intervals, pre_time=0.1, post_time=0.1, ax=None):
    """
    Plot the baseline pca trajectories and stimulations with some lead in and lead out
    """
    cc = ["k", rec.laser_color, "tab:green"]

    intervals_pre = np.vstack([intervals[:, 0] - pre_time, intervals[:, 0]]).T
    intervals_post = np.vstack([intervals[:, 1], intervals[:, 1] + post_time]).T
    n_stims = intervals.shape[0]
    colors = [cc[1]] * n_stims
    colors_pre = [cc[0]] * n_stims
    colors_post = [cc[2]] * n_stims

    dims = [0, 1]
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    t0 = intervals[0, 0] - 30  # 600
    tf = intervals[0, 0]  # 550
    cbp.plot_projection_line_multicondition(
        pop.projection,
        pop.tbins,
        np.array([[t0, tf]]),
        dims=dims,
        colors=["k"],
        ax=ax,
        alpha=0.1,
    )
    cbp.plot_projection_line_multicondition(
        pop.projection, pop.tbins, intervals_pre, colors=colors_pre, ax=ax, dims=dims
    )
    cbp.plot_projection_line_multicondition(
        pop.projection, pop.tbins, intervals, colors=colors, ax=ax, dims=dims
    )
    cbp.plot_projection_line_multicondition(
        pop.projection, pop.tbins, intervals_post, colors=colors_post, ax=ax, dims=dims
    )
    plt.xlim([-2, 4])
    plt.ylim([-2, 4])
    return pop


def plot_lowd_stim_response_1D(
    rec, pop, intervals, pre_time=0.1, post_time=0.1, figsize=(3, 7)
):
    """
    Plot PETH of the stimulation evoked response in low-D space
    """
    starts = intervals[:, 0]
    starts_control = starts + np.random.uniform(-0.5, 0.5, starts.shape[0])
    pulse_dur = np.mean(np.diff(intervals, 1))

    n_dim = 5
    f, axs = plt.subplots(
        nrows=n_dim, ncols=2, sharey=True, sharex=True, figsize=figsize
    )
    pre_win = pre_time
    post_win = post_time
    for dim in range(n_dim):
        ax = axs[dim, 0]
        eta = get_eta(
            pop.projection[:, dim],
            pop.tbins,
            starts,
            pre_win=pre_win,
            post_win=post_win,
        )
        eta_control = get_eta(
            pop.projection[:, dim],
            pop.tbins,
            starts_control,
            pre_win=pre_win,
            post_win=post_win,
        )
        ax.plot(eta["t"], eta["mean"])
        ax.fill_between(eta["t"], eta["lb"], eta["ub"], alpha=0.3)
        ax.plot(eta_control["t"], eta_control["mean"], color="k")
        ax.fill_between(
            eta_control["t"], eta_control["lb"], eta_control["ub"], alpha=0.3, color="k"
        )
        ax.axvspan(0, pulse_dur, color=rec.laser_color, alpha=0.2)

        ax_sweep = axs[dim, 1]
        cbp.plot_sweeps(
            pop.tbins,
            pop.projection[:, dim],
            starts,
            pre=pre_win,
            post=post_win,
            color="C0",
            ax=ax_sweep,
            lw=0.1,
        )
        ax_sweep.axvspan(0, pulse_dur, color=rec.laser_color, alpha=0.2)
        ax.set_ylabel(f"Dim {dim}")
    sns.despine(trim=True)
    plt.tight_layout()


def compute_post_stim_dispersion(
    rec, pop, intervals, post_time=0.05, ndims=10, mode="trace"
):
    """
    Compute the post-stimulus locations and their dispersion in low-dimensional space.

    Args:
        rec (object): Recording object containing experimental data.
        pop (object): Population object containing neural data projections.
        intervals (array-like): Array of time intervals (start, end) for stimulus events.
        post_time (float, optional): Time window after the stimulus to consider for analysis (default is 0.05 seconds).
        ndims (int, optional): Number of dimensions to consider in the low-dimensional space (default is 10).
        mode (str, optional): Mode of distance computation. Supported modes are:
            - 'mahalanobis': Compute Mahalanobis distance.
            - 'pairwise': Compute pairwise Euclidean distances.
            - 'trace': Compute the trace of the covariance matrix.
            - 'det': Compute the determinant of the covariance matrix.
            (default is 'pairwise').

    Returns:
        tuple: A tuple containing:
            - post_stim (ndarray): Array of post-stimulus locations in the low-dimensional space.
            - result (float or ndarray): Depending on the mode, this can be:
                - The trace of the covariance matrix (if mode is 'trace').
                - The determinant of the covariance matrix (if mode is 'det').
                - Pairwise distances between post-stimulus locations (if mode is 'pairwise').

    """
    supported_modes = ["mahalanobis", "pairwise", "trace", "det"]
    assert mode in supported_modes, f"Mode must be in {supported_modes}"

    post_stim = []

    for t0, tf in intervals:
        sf2 = np.searchsorted(pop.tbins, tf + post_time)
        post_stim.append([pop.projection[sf2, :ndims]])
    post_stim = np.vstack(post_stim)

    # Compute the covariance matrix of post-stim
    if mode == "trace":
        cov = np.cov(post_stim, rowvar=False)
        trace = np.trace(cov)
        return (post_stim, trace)

    if mode == "det":
        cov = np.cov(post_stim, rowvar=False)
        det = np.linalg.det(cov)
        return (post_stim, det)

    if mode == "pairwise":
        D = cdist(post_stim, post_stim)

        rr, cc = np.triu_indices_from(D, k=1)
        D = D[rr, cc]
        return (post_stim, np.median(D))

    if mode == "mahalanobis":
        raise ValueError("Mahalonobis distance is potentially not appropriate")
        cov_matrix = np.cov(post_stim, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)

        # Compute the mean of the dataset
        mean = np.mean(post_stim, axis=0)

        # Initialize Mahalanobis distance metric with the inverse covariance matrix
        mahalanobis_metric = DistanceMetric.get_metric("mahalanobis", VI=inv_cov_matrix)

        # Calculate Mahalanobis distances from the mean for all points in the dataset
        D = mahalanobis_metric.pairwise(post_stim, [mean]).ravel()

        return (post_stim, np.median(D))
    return None


def compute_expiratory_attractor(rec, pop, ndims=5, binsize=np.pi / 25, t0=300, tf=600):
    """
    Compute the location in latent space of the expiratory attractor
    i.e., the point in space where phi is near to pi/-pi transition

    Explicitly finds the location of the points taht occur when phi is between -pi and -pi+binsize

    Args:
        rec (object): Recording object containing experimental data.
        pop (object): Population object containing neural data projections.
        intervals (array-like): Array of time intervals (start, end) for stimulus events.
        binsize (float, optional): Size of the bins for phase computation. Defaults to np.pi/50.

    Returns:
        tuple: A tuple containing:
            - attractor (ndarray): Array representing the computed expiratory attractor.
            - phases (ndarray): Array of phases corresponding to the attractor.

    """

    # Find the subset of points to compute on
    s0, sf = np.searchsorted(pop.tbins, [t0, tf])
    X = pop.projection[s0:sf, :]

    # Map the phase time basis to the same time basis as the projection
    phi2 = pop.sync_var(rec.phi, rec.phi_t)
    phi2 = phi2[s0:sf]

    # Find the points that are in the post inspiratory bin (i.e., between -pi and -pi+binsize)
    idx = np.logical_and(phi2 > -np.pi, phi2 < (-np.pi + binsize))

    # Return the mean of those points in lowD space
    attractor = X[idx, :].mean(0)
    return attractor[:ndims]


def compute_distance_to_attractor(
    rec, pop, test_points, ndims=5, binsize=np.pi / 50, t0=300, tf=600
):
    """
    Compute the latent distanct of the points in test_points from the expiratory attractor

    Returns:
        float: The mean distance of the points in test_points from the expiratory attractor
    """
    attractor = compute_expiratory_attractor(
        rec, pop, ndims=ndims, binsize=binsize, t0=300, tf=600
    )
    D = cdist(test_points, attractor[np.newaxis, :])
    return np.mean(D)


def compute_nearest_phase(rec, pop, test_points, ndims=5, bins=50, t0=300, tf=600):
    """
    Compute the nearest phase to each point in test_points

    Returns:
        np.array: The phase of the nearest point in the population space, for each test point
        np.array: The distance to the nearest point in the population space, for each test point
    """
    s0, sf = np.searchsorted(pop.tbins, [t0, tf])
    X = pop.projection[s0:sf, :]
    phi2 = pop.sync_var(rec.phi, rec.phi_t)
    likli = []
    for ii in range(ndims):
        _bins, _likli = weighted_histogram(
            phi2[s0:sf], X[:, ii], bins=np.linspace(-np.pi, np.pi, bins)
        )
        likli.append(_likli)

    # Turn likli into a matrix
    likli = np.vstack(likli).T

    # Compute the distance between each test point and the liklihood so that we get a distance for each point in test points
    D = cdist(test_points, likli[:, :ndims])
    closest_point = np.argmin(D, 1)

    return (_bins[closest_point], np.min(D, 1))


def compute_speed_by_phase_holds(rec,pop):
    """
    Compute the speed of the projection points as a function of phase specifically for the phasic stimulations

    """

    # Map the phase time basis to the same time basis as the projection
    phi2 = pop.sync_var(rec.phi, rec.phi_t)
    speed = pop.projection_speed
    bins = np.linspace(-np.pi, np.pi, 50)
    bins = np.round(bins, 3)

    intervals, stims = rec.get_stims("hold")
    pre_time = 5
    intervals = np.hstack([intervals[:,0][:,np.newaxis]-pre_time,intervals])

    df = pd.DataFrame()
    ii=0
    for tc,t0,tf in intervals:
        sc,s0, sf = np.searchsorted(pop.tbins, [tc,t0, tf])
        _,_ctrl = weighted_histogram(phi2[sc:s0],speed[sc:s0],bins=bins,wrap=True)
        _,_stim = weighted_histogram(phi2[s0:sf],speed[s0:sf],bins=bins,wrap=True)
        _df = pd.DataFrame()
        _df['control_speed'] = _ctrl
        _df['stim_speed'] = _stim
        _df['phase'] = bins
        _df['replicate'] = ii
        # _df = _df.fillna(0)
        df = pd.concat([df,_df])
        ii+=1

    df = df.melt(id_vars=['phase','replicate'], value_vars=['control_speed','stim_speed'],var_name='condition',value_name='speed')
    df = df.fillna(0)

    df = pd.pivot_table(df, index=['phase','condition'],  values='speed')


    df["subject"] = rec.subject
    df["genotype"] = rec.genotype
    df["sequence"] = rec.sequence
    df["eid"] = rec.eid
    df = df.reset_index()
    return df

# Speed by phase
def get_speed_by_phase(rec, pop):
    """
    Compute the speed of the projection points as a function of phase specifically for the phasic stimulations
    Should be rewritten and split up!

    Makes and saves a plot too.
    NB: Very hardcoded.
    """

    # Map the phase time basis to the same time basis as the projection
    phi2 = pop.sync_var(rec.phi, rec.phi_t)

    f = plt.figure(figsize=(3, 2))

    # Inspiratory Stims
    intervals, stims = rec.get_phasic_stims("insp")
    ax = f.add_subplot(121, projection="polar")
    _, _, insp_speeds, polar_bins = cbp.plot_polar_average(
        phi2,
        pop.projection_speed,
        pop.tbins,
        t0=stims["start_time"],
        tf=stims["end_time"],
        color=rec.laser_color,
        multi="std",
        ax=ax,
    )
    cbp.plot_polar_average(
        phi2,
        pop.projection_speed,
        pop.tbins,
        t0=stims["start_time"] - 20,
        tf=stims["end_time"] - 10,
        color="k",
        multi="std",
        ax=ax,
    )
    ax.set_title("Insp. stims", fontsize="small")

    # Expiratory Stims
    intervals, stims = rec.get_phasic_stims("exp")
    ax = f.add_subplot(122, projection="polar")
    _, _, exp_speeds, polar_bins = cbp.plot_polar_average(
        phi2,
        pop.projection_speed,
        pop.tbins,
        t0=stims["start_time"],
        tf=stims["end_time"],
        color=rec.laser_color,
        multi="std",
        ax=ax,
    )
    _, _, ctrl_speeds, polar_bins = cbp.plot_polar_average(
        phi2,
        pop.projection_speed,
        pop.tbins,
        t0=stims["start_time"] - 20,
        tf=stims["end_time"] - 10,
        color="k",
        multi="std",
        ax=ax,
    )
    ax.set_title("Exp. stims", fontsize="small")

    plt.tight_layout()
    fn = EXAMPLE_PATH.joinpath(f"{rec.subject}_g{rec.sequence}_{rec.genotype}_speed_by_phase.svg")
    plt.savefig(fn)
    plt.close("all")

    # Output dataframe
    df = pd.DataFrame()
    df["insp_speed"] = insp_speeds.mean(0)
    df["exp_speed"] = exp_speeds.mean(0)
    df["ctrl_speed"] = ctrl_speeds.mean(0)
    df["insp_speed_pct_diff"] = (
        100 * (insp_speeds.mean(0) - ctrl_speeds.mean(0)) / ctrl_speeds.mean(0)
    )
    df["exp_speed_pct_diff"] = (
        100 * (exp_speeds.mean(0) - ctrl_speeds.mean(0)) / ctrl_speeds.mean(0)
    )
    df["phase"] = np.round(np.linspace(-np.pi, np.pi, polar_bins.shape[0]), 3)
    df["subject"] = rec.subject
    df["genotype"] = rec.genotype
    df["sequence"] = rec.sequence
    df["eid"] = rec.eid
    df = df.set_index(["genotype", "subject", "sequence", "eid", "phase"])
    return df


def plot_rasters(rec):
    """
    Plot the rasters for the different stimulation types
    """
    win = 10
    f, axs = plt.subplots(nrows=5, figsize=(6, 15), sharey=True)
    t0 = 500
    tf = t0 + win
    idx = np.logical_and(rec.spikes.times > t0, rec.spikes.times < tf)
    bbp.driftmap(rec.spikes.times[idx], rec.spikes.depths[idx], ax=axs[0])
    axs[0].set_xlim(t0, tf)

    pulse_dur = 0.01
    intervals, _ = rec.get_pulse_stims(pulse_dur)
    t0 = np.min(intervals)
    tf = t0 + win
    idx = np.logical_and(rec.spikes.times > t0, rec.spikes.times < tf)
    bbp.driftmap(rec.spikes.times[idx], rec.spikes.depths[idx], ax=axs[1])
    cbp.plot_laser(rec.laser, mode="bar", ax=axs[1], lw=10, wavelength=rec.wavelength)
    axs[1].set_xlim(t0, tf)

    pulse_dur = 0.05
    intervals, _ = rec.get_pulse_stims(pulse_dur)
    t0 = np.min(intervals)
    tf = t0 + win
    idx = np.logical_and(rec.spikes.times > t0, rec.spikes.times < tf)
    bbp.driftmap(rec.spikes.times[idx], rec.spikes.depths[idx], ax=axs[2])
    cbp.plot_laser(rec.laser, mode="bar", ax=axs[2], lw=10, wavelength=rec.wavelength)
    axs[2].set_xlim(t0, tf)

    intervals, stims = rec.get_pulse_stims(2)
    t0 = np.min(intervals) - win / 2
    tf = t0 + win
    idx = np.logical_and(rec.spikes.times > t0, rec.spikes.times < tf)
    bbp.driftmap(rec.spikes.times[idx], rec.spikes.depths[idx], ax=axs[3])
    cbp.plot_laser(rec.laser, mode="bar", ax=axs[3], wavelength=rec.wavelength)
    axs[3].set_xlim(t0, tf)

    intervals, stims = rec.get_HB_stims(duration=2)
    t0 = np.min(intervals) - win / 2
    tf = t0 + win
    idx = np.logical_and(rec.spikes.times > t0, rec.spikes.times < tf)
    bbp.driftmap(rec.spikes.times[idx], rec.spikes.depths[idx], ax=axs[4])
    cbp.plot_laser(intervals, mode="bar", ax=axs[4], color=HB_MAP[5])
    axs[4].set_xlim(t0, tf)

    titles = ["control", "10ms", "50ms", "2s", "15cmH$_2$O"]
    for ax, title in zip(axs, titles):
        ax.plot(rec.physiology.times, rec.physiology.dia * 100 + 3200, lw=1)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{rec.subject}_g{rec.sequence}_{rec.genotype}_raster.png")


def compute_mean_distances(pop, times_A, times_B, ndims):
    """
    Compute the mean distances between two sets of time intervals in a low-dimensional space.

    Args:
        pop (object): Population object containing neural data projections.
        times1 (array-like): Array of time points for the first set of intervals.
        times2 (array-like): Array of time points for the second set of intervals.
        ndims (int, optional): Number of dimensions to consider in the low-dimensional space (default is 10).

    Returns:
        float: The mean distance between the two sets of time intervals in the low-dimensional space.
    """

    sA = np.searchsorted(pop.tbins, times_A)
    sB = np.searchsorted(pop.tbins, times_B)

    test_points_A = pop.projection[sA, :ndims]
    test_points_B = pop.projection[sB, :ndims]

    meanA = np.mean(test_points_A, 0)
    meanB = np.mean(test_points_B, 0)

    D = np.linalg.norm(meanA - meanB)

    return D


def compute_median_stim_projection(rec,pop,mode='hb'):
    """
    Compute the mean projection of the population for a given stimulation type.

    Args:
        rec (object): Recording object containing experimental data.
        pop (object): Population object containing neural data projections.
        mode (str): Stimulation type. Options are 'hb','hold','50ms', or 'control'

    Returns:
        np.array: The median projection of the population for the specified stimulation type.
    """

    assert mode in ['hb','hold','control'], "Mode must be one of 'hb','hold','control'"
    intervals,stims = rec.get_stims(mode)


    X = np.full([pop.cbins.shape[0], intervals.shape[0]], np.nan)
    for ii, (t0, tf) in enumerate(intervals):
        if mode in ['hb','hold']:
            t0 += 1
            tf = t0+2

        s0, sf = np.searchsorted(pop.tbins, [t0, tf])
        x = pop.projection[s0:sf, :]
        x = np.median(x, 0)
        X[:, ii] = x
    y = np.nanmedian(X,1)

    return y,X

def compute_hb_comparisons(rec,pop,ndims_to_compute):
    conditions = ['hb','hold','control']
    df = pd.DataFrame()
    df_pos = pd.DataFrame()
    max_dim = np.max(ndims_to_compute)
    for condition in conditions:
        y,X = compute_median_stim_projection(rec,pop,mode=condition)
        df[condition] = y[:max_dim]
        X = X[:max_dim,:]
        _,yy = np.meshgrid(np.arange(X.shape[1]),np.arange(X.shape[0]))

        _df_pos = pd.DataFrame(X.T)
        _df_pos.index.name='replicate'
        _df_pos['condition'] = condition
        _df_pos = _df_pos.set_index('condition', append=True)
        df_pos = pd.concat([df_pos,_df_pos])
    
    df_pos.reset_index(drop=False)
    df_pos['genotype'] = rec.genotype
    df_pos['eid'] = rec.eid
    df_pos['subject'] = rec.subject
    df_pos = pd.pivot_table(df_pos, index=['genotype','eid','subject','condition','replicate'])
    df_pos.reset_index(drop=False, inplace=True)


    df = df.T
    # Compute the distance between the median positions
    D = pd.DataFrame()
    for a,b in combinations(conditions,2):
        for ndims in ndims_to_compute:
            x = df.loc[a].values[:ndims]
            y = df.loc[b].values[:ndims]
            d = np.linalg.norm(x - y)
            _D = pd.DataFrame()
            _D['comparison'] = [f"{a}_{b}"]
            _D['ndims'] = [ndims]
            _D['distance'] = [d]
            D = pd.concat([D, _D])
    D['genotype'] = rec.genotype
    D['eid'] = rec.eid
    D['subject'] = rec.subject

    return D,df_pos


def get_intervals(rec):
    # Determine the timepoints that demarcate the different stimulations

    starts = np.random.uniform(500, 600, 100)
    intervals_control = np.vstack([starts, starts]).T   
    intervals_50ms = rec.get_stims("50ms")[0]

    intervals_hb = rec.get_HB_stims(duration=5)[0]
    intervals_hb = np.c_[intervals_hb[:, 0] + 0.5, intervals_hb[:, 0] + 1.5]

    intervals = {}
    intervals["50ms"] = intervals_50ms
    intervals["hb"] = intervals_hb
    intervals["random"] = intervals_control
    return intervals


# ============== #
# MAIN LOOP
# ============== #
rez = pd.DataFrame()
speed_rez = pd.DataFrame()
df_D = pd.DataFrame()
closest_phi_DF = pd.DataFrame()

rez_full_fn = Path(r'distance_results_full.csv')
rez_fn = Path(r'distance_results.csv')
median_distances_fn = Path(r'median_distances.csv')
closest_phi_full_fn = Path(r'closest_phi_full.csv')
closest_phi_fn = Path(r'closest_phi.csv')
speed_fn = Path(r'speed_by_phase.csv')
hold_stim_pca_fn = Path(r'hold_stim_pca.csv')
hold_stim_speeds = Path(r'hold_stim_speeds.csv')

def remove_files():
    if closest_phi_full_fn.exists():
        closest_phi_full_fn.unlink()
    if rez_full_fn.exists():
        rez_full_fn.unlink()
    if median_distances_fn.exists():
        median_distances_fn.unlink()
    if closest_phi_fn.exists():
        closest_phi_fn.unlink()
    if speed_fn.exists():
        speed_fn.unlink()
    if rez_fn.exists():
        rez_fn.unlink()
    if hold_stim_pca_fn.exists():
        hold_stim_pca_fn.unlink()
    if hold_stim_speeds.exists():
        hold_stim_speeds.unlink()
remove_files()

for eid in EIDS_NEURAL:
    # Load and compute projections
    rec = Rec(one, eid)
    pop = Population(rec.spikes.times, rec.spikes.clusters, t0=300, tf=600)
    pop.compute_projection()
    pop.compute_projection_speed()
    ndims_to_compute = range(2, 11)

    df_hold_speed = compute_speed_by_phase_holds(rec,pop)
    df_hold_speed.to_csv(hold_stim_speeds, mode="a", header=not hold_stim_speeds.exists(), index=False)

    # Compute pairwise median distance between Hold, HB, and control stims
    D,pos = compute_hb_comparisons(rec, pop, ndims_to_compute)
    D.to_csv(median_distances_fn, mode="a", header=not median_distances_fn.exists(), index=False)
    pos.to_csv(hold_stim_pca_fn, mode="a", header=not closest_phi_full_fn.exists(), index=False)

    intervals = get_intervals(rec)
 

    # Compute dispersion, distance to attractor, and closest phase
    conditions = ["50ms", "hb", "random"]
    post_times = np.arange(0, 0.1, 0.005)
    rez = pd.DataFrame()
    closest_phi_DF = pd.DataFrame()
    for condition,ndims,post_time in product(conditions, ndims_to_compute, post_times):
        if condition=='hb':
            post_time += 1
        # Dispersion
        post_stim, D = compute_post_stim_dispersion(
            rec,
            pop,
            intervals[condition],
            post_time=post_time,
            ndims=ndims,
            mode="trace",
        )

        # Distance to attractor
        D_attractor = compute_distance_to_attractor(
            rec, pop, post_stim, ndims=ndims, t0=500, tf=600
        )
        # Put the dispersion and distance to attractor in a dataframe
        _rez = pd.DataFrame()
        _rez["dispersion"] = [D]
        _rez["condition"] = condition
        _rez["exp_distance"] = [D_attractor]
        _rez["genotype"] = rec.genotype
        _rez["ndims"] = ndims
        _rez["post_time"] = post_time
        _rez["eid"] = rec.eid
        _rez["subject"] = rec.subject
        rez = pd.concat([rez, _rez])

        # Closest phase
        closest_phi, D_to_manifold = compute_nearest_phase(
            rec, pop, post_stim[:, :ndims], ndims=ndims
        )
        # Put the closest phi points for all stims in a dataframe
        _closest_phi_DF = pd.DataFrame()
        _closest_phi_DF["closest_phi"] = closest_phi
        _closest_phi_DF["genotype"] = rec.genotype
        _closest_phi_DF["ndims"] = ndims
        _closest_phi_DF["post_time"] = post_time
        _closest_phi_DF["eid"] = rec.eid
        _closest_phi_DF["subject"] = rec.subject
        _closest_phi_DF["condition"] = condition
        closest_phi_DF = pd.concat([closest_phi_DF, _closest_phi_DF])


    rez.to_csv(rez_full_fn, mode="a", header=not rez_full_fn.exists(), index=False)
    closest_phi_DF.to_csv(
        closest_phi_full_fn, mode="a", header=not closest_phi_full_fn.exists(), index=False
    )
    speed = get_speed_by_phase(rec, pop)
    speed.to_csv(speed_fn, mode="a", header=not speed_fn.exists())

    # Garbage collection
    del rec,pop,speed,closest_phi_DF,rez
    gc.collect()

rez_full = pd.read_csv(rez_full_fn)
rez = rez_full.query("post_time == 0.025")
rez.reset_index(drop=True, inplace=True)
rez.to_csv("distance_results.csv", index=False)

closest_phi_full = pd.read_csv(closest_phi_full_fn)
closest_phi = closest_phi_full.query("post_time == 0.025")
closest_phi.reset_index(drop=True, inplace=True)
closest_phi.to_csv("closest_phi.csv", index=False)


def plot_clearer_stim_example(eid,figsize=(2.5,2.5),lw=1,ms=2,ext="png",dpi=600):
    """ 
    Makes a step by step plot of the stimulation effect on LowD space
    """
    rec = Rec(one, eid)
    pop = Population(rec.spikes.times, rec.spikes.clusters, t0=300, tf=600)
    pop.compute_projection()
    phi2 = pop.sync_var(rec.phi, rec.phi_t)
    pulse_dur = 0.05
    intervals, _ = rec.get_pulse_stims(pulse_dur)
    dims = [0, 1]
    post_time = 0.025
    t0,tf = 570,600
    fn = f"example_stim_{rec.prefix}"
    fn = EXAMPLE_PATH.joinpath(fn)
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot()
    pop.plot_projection_line(
        dims=dims,
        t0=t0,
        tf=tf,
        cvar=phi2,
        cmap="RdBu_r",
        colorbar_title=r"$\phi$ (rads.)",
        alpha=0.5,
        ax=ax,
    )
    ax.autoscale()
    ax.set_aspect("equal")
    # Modify colorbar ticks
    f.get_children()[2].set_xticks([-np.pi, 0, np.pi])
    f.get_children()[2].set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    plt.savefig(f"{fn}_a.{ext}",dpi=dpi)

    cbp.plot_projection_line_multicondition(
        pop.projection,
        pop.tbins,
        intervals,
        colors=[rec.laser_color for _ in range(intervals.shape[0])],
        ax=ax,
        dims=dims,
        lw=lw,
        use_arrow=True,
        multi_arrow=True,
        mutation_scale=5
    )
    plt.savefig(f"{fn}_b.{ext}",dpi=dpi)
    intervals_post = np.vstack([intervals[:, 1], intervals[:, 1] + post_time]).T
    cbp.plot_projection_line_multicondition(
        pop.projection,
        pop.tbins,
        intervals_post,
        colors=["k" for _ in range(intervals_post.shape[0])],
        ax=ax,
        dims=dims,
        lw=lw,
    )

    plt.savefig(f"{fn}_c.{ext}",dpi=dpi)
    post_stim, D = compute_post_stim_dispersion(rec, pop, intervals, post_time, ndims=2)
    ax.plot(post_stim[:, 0], post_stim[:, 1], "o", mec="k", mew=lw/2,ms=ms, mfc="silver")
    plt.savefig(f"{fn}_d.{ext}",dpi=dpi)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.close("all")

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)
    pop.plot_projection(
        dims=dims,
        t0=t0,
        tf=tf,
        cvar=phi2,
        cmap="RdBu_r",
        colorbar_title=r"$\phi$ (rads.)",
        alpha=0.5,
        ax=ax,
        s=0,
    )
    ax.autoscale()
    ax.set_aspect("equal")
    n = post_stim.shape[0]
    for ii in range(n):
        for jj in range(ii + 1, n):
            ax.plot(
                [post_stim[ii, 0], post_stim[jj, 0]],
                [post_stim[ii, 1], post_stim[jj, 1]],
                "k-",
                lw=0.25,
                alpha=0.1,
            )
    f.get_children()[2].set_xticks([-np.pi, 0, np.pi])
    f.get_children()[2].set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.plot(post_stim[:, 0], post_stim[:, 1], "o", mec="k", mew=lw, ms=ms, mfc="silver")
    plt.savefig(f"{fn}_e.{ext}",dpi=dpi)

    # Speed
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)
    pop.compute_projection_speed()
    pop.plot_projection_line(
        dims=dims,
        t0=t0,
        tf=tf,
        cvar=pop.projection_speed,
        cmap="viridis",
        colorbar_title="Speed (a.u.)",
        alpha=0.5,
        ax=ax,
    )
    ax.autoscale()
    ax.set_aspect("equal")
    clims = f.get_children()[2].get_xticks()
    f.get_children()[2].set_xticklabels([f"{clim:.2f}" for clim in clims])
    plt.savefig(f"{fn}_f.{ext}",dpi=dpi)

exts = ['pdf'] #  ['pdf','png']
for ext in exts:
    eid = one.search(subject="m2024-40", datasets="spikes.times.npy")[0]
    plot_clearer_stim_example(eid,ext=ext)

    eid = one.search(subject="m2024-34", datasets="spikes.times.npy")[0]
    plot_clearer_stim_example(eid,ext=ext)

    eid = one.search(subject="m2024-30", datasets="spikes.times.npy")[0]
    plot_clearer_stim_example(eid,ext=ext)

    eid = one.search(subject="m2025-01", datasets="spikes.times.npy")[0]
    plot_clearer_stim_example(eid,ext=ext)
