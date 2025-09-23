"""
Analyze RSLDS to learned stimulus responses
"""

import sys
sys.path.append("../")
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
from sklearn.svm import SVR
import matplotlib.colors as mcolors
from itertools import product
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
import autograd.numpy.random as npr
import seaborn.objects as so
import click
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
from cibrrig.utils.utils import remap_time_basis
from cibrrig.preprocess.physiology import burst_stats_dia
from cibrrig.plot import plot_reset_curve, AlfBunch, laser_colors
from cibrrig.analysis.population import rasterize, Population
import copy
from utils import Rec, one, EIDS_REGEN, PHASE_MAP, EIDS_REGEN_NOSTIM, set_style
from joblib import Parallel, delayed
from ssm_helpers import (
    _get_stim_vector,
)

npr.seed(0)
set_style()
rcParams["animation.writer"] = "ffmpeg"
import os
N_JOBS = os.environ.get("SLURM_CPUS_PER_TASK", -1)

def sample_x_with_sink(
    model, z, xhist, V, sink, input=None, tag=None, with_noise=False
):
    assert not with_noise, "Not implemented yet"
    obs = model.dynamics
    D, As, bs, Vs = obs.D, obs.As, obs.bs, obs.Vs

    if xhist.shape[0] < obs.lags:
        # Sample from the initial distribution
        S = np.linalg.cholesky(obs.Sigmas_init[z]) if with_noise else 0
        return obs.mu_init[z] + np.dot(S, npr.randn(D))
    else:
        # Sample from the autoregressive distribution
        mu = Vs[z].dot(input[: obs.M]) + bs[z]
        for l in range(obs.lags):
            Al = As[z][:, l * D : (l + 1) * D]
            mu = mu + Al.dot(xhist[-l - 1]) + V.dot(xhist[-l - 1] - sink)

        S = np.linalg.cholesky(obs.Sigmas[z]) if with_noise else 0
        return mu + np.dot(S, npr.randn(D))


def sample_x_partition_distance(
    model,
    z,
    xhist,
    k,
    stim_vector,
    input=None,
    tag=None,
    with_noise=False,
    scaling="sigmoid",
    slope=1.0,
):
    # k is the index of the partition we want to use to scale to
    # Stim vector is the supplied input
    assert not with_noise, "Not implemented yet"
    obs = model.dynamics
    D, As, bs, Vs = obs.D, obs.As, obs.bs, obs.Vs

    # Scale the input according to distance to partition (sigmoid, valley, uniform)
    v = xhist[-1].dot(model.transitions.Rs.T) + model.transitions.r
    _v = np.diff(v, 1)
    if scaling == "sigmoid":
        f = 2 * k - 1
        scaler = apply_sigmoid(f * _v, slope=slope)
    elif scaling == "valley":
        # Scale the vector according to the distance to the partition
        scaler = _v
    elif scaling == "mountain":
        scaler = -1 * _v
    elif scaling == "uniform":
        # Do not scale the vector
        scaler = 1
    elif scaling in ["insp", "exp"]:
        _z = np.argmax(v)
        if _z == k:
            scaler = 1
        else:
            scaler = 0
    input *= scaler

    if xhist.shape[0] < obs.lags:
        # Sample from the initial distribution
        S = np.linalg.cholesky(obs.Sigmas_init[z]) if with_noise else 0
        return obs.mu_init[z] + np.dot(S, npr.randn(D))
    else:
        # Sample from the autoregressive distribution
        mu = bs[z] + stim_vector * input
        for l in range(obs.lags):
            Al = As[z][:, l * D : (l + 1) * D]
            mu = mu + Al.dot(xhist[-l - 1])

        S = np.linalg.cholesky(obs.Sigmas[z]) if with_noise else 0
        return mu + np.dot(S, npr.randn(D))


def sample_with_stim(rec, T, stimulus, input=None, with_noise=False, prefix=None):
    """
    Sample the RSLDS model with a stimulus field that is a function of X
    x_t = A^k*x_{t-1} + V_{t-1}*x_{t-1} + vb_t + b

    Modified from ssm.lds.sample

    """

    valid_modes = ["sink", "sigmoid", "valley", "uniform", "insp", "exp"]
    mode = stimulus.mode
    assert mode in valid_modes, f"mode must be one of {valid_modes}"

    if mode == "sink":
        assert stimulus.sink is not None, "sink must be provided for sink mode"
        sink = stimulus.sink
        # V does not have to be provided for sink mode - it will be assumed to be -I
    elif mode in ["sigmoid", "valley"]:
        stim_vector = stimulus.stim_vector
        k = stimulus.k
    elif mode in ["uniform", "insp", "exp"]:
        stim_vector = stimulus.stim_vector
        if mode in ["insp", "exp"]:
            k = rec.k_phase.index(stimulus.mode)
        else:
            k = stimulus.k

    # Get the model parameters
    model = rec.rslds
    N = model.N
    M = (model.M,)
    D = (model.D,)

    # Define the sink
    if stimulus.V is None:
        V = np.eye(model.D) * -1
    else:
        V = stimulus.V

    blank = np.zeros((T + 1,) + M)
    if prefix is None:
        pad = 1
        z = np.zeros(T + 1, dtype=int)
        x = np.zeros((T + 1,) + D)
        if input is None:
            input = np.zeros((T + pad,) + M)
        else:
            input = np.concatenate([np.zeros((pad,) + M), input], axis=0)
        xmask = np.ones((T + 1,) + D, dtype=bool)

        # Sample the first state from the initial distribution

        x0 = x[0][np.newaxis, :]
        z[0] = np.argmax(x0.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)[
            0
        ]
        x[0] = model.dynamics.sample_x(z[0], x[:0], tag=None, with_noise=with_noise)
    else:
        zhist, xhist, yhist = prefix
        pad = len(zhist)
        assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < model.K
        assert yhist.shape == (pad, N)

        z = np.concatenate((zhist, np.zeros(T, dtype=int)))
        x = np.concatenate((xhist, np.zeros((T,) + D)))

        input = (
            np.zeros((T + pad,) + M)
            if input is None
            else np.concatenate((np.zeros((pad,) + M), input))
        )
        xmask = np.ones((T + pad,) + D, dtype=bool)
        blank = np.zeros((T + pad,) + M)

    for t in range(pad, T + pad):
        Pt = np.exp(
            model.transitions.log_transition_matrices(
                x[t - 1 : t + 1],
                input[t - 1 : t + 1],
                mask=xmask[t - 1 : t + 1],
                tag=None,
            )
        )[0]
        z[t] = npr.choice(model.K, p=Pt[z[t - 1]])
        # Input is applied to V, not to the learned Vs
        if mode == "sink":
            x[t] = sample_x_with_sink(
                model,
                z[t],
                x[:t],
                V * input[t],
                sink,
                input=blank[t],
                with_noise=with_noise,
            )
        else:
            x[t] = sample_x_partition_distance(
                model,
                z[t],
                x[:t],
                k,
                stim_vector,
                input=input[t],
                with_noise=with_noise,
                scaling=mode,
                slope=stimulus.slope,
            )

    return z[pad:], x[pad:], input[pad:]


def fit_sim_dia_SVR(X, y):
    dia_model = SVR()
    dia_model.fit(X, y)
    return dia_model


def fit_sim_dia_relu(X, y):
    transformer = TransformedTargetRegressor(
        regressor=Lasso(),
        func=lambda x: x,  # identity
        inverse_func=lambda x: np.maximum(x, 0),  # ensures non-negative predictions
    )

    transformer.fit(X, y)
    return transformer


def get_stim_field(rec, stimulus, xy=None, plot_tgl=False):
    """
    Compute the stimulus field for a given stimulus and xy coordinates
    Args:
        rec: Rec object
        stimulus: Stimulus object
        xy: (N,2) array of x,y coordinates
    Returns:
        field: (N,) array of stimulus field values
    """
    if xy is None:
        x = np.linspace(-30, 30, 10)
        y = np.linspace(-30, 30, 10)
        X, Y = np.meshgrid(x, y)
        xy = np.c_[X.ravel(), Y.ravel()]

    model = rec.rslds
    p = xy.dot(model.transitions.Rs.T) + model.transitions.r

    if stimulus.mode == "sink":
        xy -= stimulus.sink
        u, v = xy.dot(stimulus.V).T
    else:
        _p = np.diff(p, 1).ravel()
        if stimulus.mode == "sigmoid":
            k = stimulus.k
            f = k * 2 - 1
            scaled = f * _p
            scaled = apply_sigmoid(scaled, stimulus.slope)
        elif stimulus.mode == "valley":
            scaled = _p
        elif stimulus.mode == "mountain":
            scaled = -1 * _p
        elif stimulus.mode in ["insp", "exp"]:
            k = rec.k_phase.index(stimulus.mode)
            scaled = np.where(np.argmax(p, axis=1) == k, 1, 0)
        elif stimulus.mode == "uniform":
            scaled = np.ones(p.shape[0])
        u = np.repeat(stimulus.stim_vector[0], xy.shape[0]) * scaled
        v = np.repeat(stimulus.stim_vector[1], xy.shape[0]) * scaled

    if plot_tgl:
        plt.quiver(X, Y, u, v)

    return X, Y, u, v


def rotate_vector(v, angle):
    angle = np.deg2rad(angle)
    new_angle = np.arctan2(v[1], v[0]) + angle
    return np.array([np.cos(new_angle), np.sin(new_angle)])


def plot_summary_amp_sweep(df, stim_units, fs=(2.5, 2)):
    df = df.rename(columns={"amp": "Dia. amplitude"})
    df = df.query("stim_amp >= 0")
    p = (
        so.Plot(df, x="stim_amp", y="resp_rate", color="Dia. amplitude")
        .add(so.Dot(), pointsize="Dia. amplitude")
        .label(
            x=f"Stim Amplitude ({stim_units})",
            y="Breath Rate (Hz)",
            legend="Dia. amp (a.u.)",
        )
        .layout(size=fs)
        .scale(color="winter")
        .limit(x=(0, None), y=(0, None))
    ).plot()
    return p


def plot_sim_phasic_stims(
    phasic_stim_output, binsize=0.01, ax=None, laser_color=laser_colors[473]
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    dia_predicted = phasic_stim_output["dia_predicted"]
    ks = phasic_stim_output["ks"]
    xs = phasic_stim_output["xs"]
    ys = phasic_stim_output["ys"]
    triggered = phasic_stim_output["triggered"]

    # Plot dia predicted
    tt = np.arange(len(dia_predicted)) * binsize
    ax.plot(tt, dia_predicted, color="k", label="Predicted DIA")
    ax.fill_between(tt, 0, dia_predicted, alpha=0.2, color="k")

    # Plot triggered events
    # Turn boolean triggered into onsets and offsets
    triggered = np.concatenate([triggered, np.zeros(1)])
    onsets = np.where(np.diff(triggered.astype(int)) == 1)[0]
    offsets = np.where(np.diff(triggered.astype(int)) == -1)[0]

    for onset, offset in zip(onsets, offsets):
        ax.axvspan(
            onset * binsize,
            offset * binsize,
            color=laser_color,
            alpha=0.5,
        )

    # Set labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\int{Dia.}$ Predicted")


def apply_sigmoid(v, slope=1):
    """
    Apply a sigmoid function to the input vector v
    """
    return 1 / (1 + np.exp(-slope * v))


@dataclass
class Stimulus:
    mode: str = None
    sink: np.ndarray = None
    V: np.ndarray = None
    stim_vector: np.ndarray = None
    k: int = None
    slope: float = 0.2

    def __post_init__(self):
        if self.mode == "sink":
            assert self.sink is not None, "sink must be provided for sink mode"
            self.sink = np.array(self.sink)
            self.V = np.eye(self.sink.shape[0]) * -0.1
            self.slope = None
        elif self.mode in ["sigmoid"]:
            assert (
                self.stim_vector is not None
            ), "stim_vector must be provided for sigmoid/valley mode"

        elif self.mode in ["uniform", "insp", "exp", "valley", "mountain"]:
            assert (
                self.stim_vector is not None
            ), "stim_vector must be provided for uniform mode"
            self.k = None
            self.slope = None
        if self.stim_vector  is not None:
            self.stim_vector = np.real(self.stim_vector)


class Rec(Rec):
    def load_rslds(self, fn=None, suffix=""):
        assert suffix in ["", "_norm", "_nostim"]
        self.norm_stim_amp = suffix != ""
        self.model_path = Path("./").joinpath(
            f"rslds_{self.subject}_g{self.sequence}{suffix}"
        )
        if fn is None:
            flist = list(self.model_path.glob("*.pkl"))

            if len(flist) > 1:
                raise ValueError("Multiple rslds files found")
            elif len(flist) == 0:
                raise ValueError("No rslds files found")
            else:
                fn = flist[0]
        with open(fn, "rb") as fid:
            dat = pickle.load(fid)

        print(f"Loading model from {fn}")
        self.rslds = dat["rslds"]
        self.original_Vs = self.rslds.dynamics.Vs.copy()
        self.tbins = dat["tbins"]
        self.q = dat["q"]
        self.binsize = dat["binsize"]
        self.elbos = dat["elbos"]
        self.assign_phase_to_k()
        self.pop = Population(
            self.spikes.times, self.spikes.clusters, binsize=self.binsize
        )
        self.load_projection()
        self.get_stimulus_magnitude_observed()
        self.stim_units = "a.u." if self.norm_stim_amp else "mW"

    def get_stimulus_magnitude_observed(self):
        if self.norm_stim_amp:
            self.stimulus_magnitude_observed = 1
        else:
            self.stimulus_magnitude_observed = self.laser.amplitudesMilliwatts.mean()

    def load_projection(self):
        fn = self.model_path.joinpath("projection.values.npy")
        if not fn.exists():
            print("Projection file not found")
            return None

        X = np.load(self.model_path.joinpath("projection.values.npy"))
        tbins = np.load(self.model_path.joinpath("projection.times.npy"))
        self.observed_latent = X
        self.observed_latent_times = tbins
        return X, tbins

    def generate_wellspaced_amps(self):
        if self.stimulus_magnitude_observed == 1:
            vals = [0, 0.2, 0.5, 1, 1.2, 1.3]
        elif self.stimulus_magnitude_observed < 12:
            vals = [0, 1, 2, 5, 10]
        else:
            vals = [0, 2, 5, 10, 20]
        return vals

    def generate_high_density_amps(self):
        if self.stimulus_magnitude_observed == 1:
            vals = np.linspace(0, 1.5, 100)
        elif self.stimulus_magnitude_observed < 12:
            vals = np.linspace(0, 15, 100)
        else:
            vals = np.linspace(0, 20, 100)
        return vals

    def plot_stim_field(
        self,
        ax=None,
        xlim=30,
        ylim=30,
        npts=10,
        input_strength=1,
        colors=None,
        fs=(2, 2),
        arrowscale=10,
    ):
        if colors is None:
            colors = self.k_colors
        model = copy.deepcopy(self.rslds)
        if ax is None:
            f = plt.figure(figsize=(fs), constrained_layout=True)
            ax = f.add_subplot(111)

        x = np.linspace(-xlim, xlim, npts)
        y = np.linspace(-ylim, ylim, npts)
        X, Y = np.meshgrid(x, y)
        xy = np.c_[X.ravel(), Y.ravel()]
        z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

        for k in range(model.K):
            v = model.dynamics.Vs[k]
            if np.all(v == 0):
                continue
            v = v.dot(np.ones(model.M) * input_strength)
            V = np.repeat(v[np.newaxis, :], xy.shape[0], axis=0)
            v_amp = np.linalg.norm(v)

            ax.quiver(
                xy[z == k, 0],
                xy[z == k, 1],
                V[z == k, 0],
                V[z == k, 1],
                color=colors[k],
                alpha=0.5,
                scale=v_amp * arrowscale,
            )
        ax.set_xlabel("$x_{1}$")
        ax.set_ylabel("$x_{2}$")
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_aspect("equal")

    def project_full_recording(self):
        raster, tbins, cbins = rasterize(
            self.spikes.times, self.spikes.clusters, self.binsize
        )
        raster = raster.T.astype(int)
        v = _get_stim_vector(
            self.laser.intervals, tbins, self.laser.amplitudesMilliwatts
        )
        print("Projecting full recording")
        elbos, q = self.rslds.approximate_posterior(raster, inputs=v)
        X = q.mean_continuous_states[0]
        return (X, tbins)

    def _preprocess_for_dia(self, subset_size=None):
        physiology = self.physiology
        if not hasattr(self, "observed_latent"):
            print("Precomputed full projection not found, using only fitted latent")
            X = self.q.mean_continuous_states[0]
            projection_times = self.tbins
        else:
            X = self.observed_latent
            projection_times = self.observed_latent_times
        y = remap_time_basis(physiology.dia, physiology.times, projection_times)
        # Speed up the fitting by using a subset of the data
        if subset_size is None or subset_size > len(y):
            return X, y

        idx = np.random.choice(np.arange(len(y)), size=subset_size, replace=False)
        return X[idx], y[idx]

    def fit_sim_dia(self, mode="relu", subset_size=10000):
        X, y = self._preprocess_for_dia(subset_size=subset_size)
        if mode == "SVR":
            dia_model = fit_sim_dia_SVR(X, y)
        elif mode == "relu":
            dia_model = fit_sim_dia_relu(X, y)
        else:
            raise ValueError("mode must be 'SVR' or 'relu'")
        self.dia_model = dia_model
        return dia_model

    def plot_amp_sweep(
        self, applied=None, amps=None, pre_time=5, post_time=5, stim_dur=20
    ):
        if amps is None:
            amps = self.generate_wellspaced_amps()
        binsize = self.binsize
        rslds = self.rslds
        max_dur = pre_time + stim_dur + post_time

        pre_bins = int(pre_time / binsize)
        post_bins = int(post_time / binsize)
        max_bins = int(max_dur / binsize)
        dur_bins = int(stim_dur / binsize)

        v = np.concatenate([np.zeros(pre_bins), np.ones(dur_bins), np.zeros(post_bins)])
        v = v[:, np.newaxis]
        f, axs = plt.subplots(
            nrows=len(amps),
            sharex=True,
            sharey=True,
            figsize=(2.5, len(amps) * 0.5),
            constrained_layout=True,
        )
        ylim = 0
        for ii, amp in enumerate(amps):
            try:
                if applied is not None:
                    a, b, c = sample_with_stim(self, max_bins, applied, input=v * amp)
                else:
                    a, b, c = rslds.sample(max_bins, with_noise=False, input=v * amp)
            except ValueError:
                print("ValueError: sample_with_stim")
                b = np.full((max_bins,), np.nan)
                axs[ii].axis("off")
                continue

            t = np.arange(len(b)) * binsize

            axs[ii].set_prop_cycle(color=["#bb521f", "#a59586"])
            axs[ii].plot(t, b)
            _predicted_dia = self.predict_dia(b)
            ax = axs[ii].twinx()
            ax.set_ylim([-15, 5])
            ax.plot(t, _predicted_dia, color="k")
            axs[ii].axis("off")
            ax.axis("off")
            ylim = np.max([np.max(np.abs(b)), ylim])
            axs[ii].text(
                0,
                0,
                f"{amp} {self.stim_units}",
                weight="bold",
                rotation=90,
                fontsize="xx-small",
                va="center",
                ha="right",
            )

        axs[0].set_ylim(-ylim, ylim * 1.5)
        axs[0].set_xlim(0, max_dur)
        axs[0].hlines(
            ylim * 1.4, pre_time, pre_time + stim_dur, color=self.laser_color, lw=2
        )

        axs[-1].hlines(-ylim * 0.9, 0, 2, color="k")
        axs[-1].text(0, -ylim * 0.9, "2s", fontsize="xx-small", va="top", ha="left")

    def compute_fixedpoints(self, input_strength=0):
        """
        Compute the fixedpoints of each k state

        Args:
            model: ssm.SLDS object
        Returns:
            n: [k x model.D] solutions to the nullclines
        """
        model = copy.deepcopy(self.rslds)
        fp = np.zeros((model.K, model.D))
        stability = np.empty((model.K,), dtype="object")
        for k in range(model.K):
            A = model.dynamics.As[k]
            b = model.dynamics.bs[k]
            v = model.dynamics.Vs[k]
            v = v.dot(np.ones(model.M) * input_strength)
            fp[k] = np.linalg.solve(np.eye(model.D) - A, b + v)
            # Check stability
            eigs = np.linalg.eigvals(A)
            if np.all(np.abs(eigs) < 1):
                stability[k] = "stable"
            elif np.all(np.abs(eigs) > 1):
                stability[k] = "unstable"
            else:
                stability[k] = "saddle"

        return fp, stability

    def predict_dia(self, projection):
        predicted_dia = self.dia_model.predict(projection)
        return predicted_dia


    def compute_amp_sweep(self, applied=None, amps=None,dia_thresh=1,mean_sub=False):
        if amps is None:
            amps = self.generate_high_density_amps()
        print("Computing amplitude sweep")
        binsize = self.binsize
        rslds = self.rslds

        pre_time = 5
        post_time = 5
        pad_time = 2
        dur = 20 + (pad_time * 2)
        max_dur = pre_time + dur + post_time

        pre_bins = int(pre_time / binsize)
        post_bins = int(post_time / binsize)
        max_bins = int(max_dur / binsize)
        dur_bins = int(dur / binsize)
        pad_bins = int(pad_time / binsize)

        v = np.concatenate([np.zeros(pre_bins), np.ones(dur_bins), np.zeros(post_bins)])
        v = v[:, np.newaxis]

        # Initialize
        predicted_dia_obs = self.predict_dia(
            self.q.mean_continuous_states[0][:max_bins]
        )
        all_breaths = burst_stats_dia(predicted_dia_obs, 1.0 / binsize,dia_thresh=dia_thresh)
        all_breaths["stim_amp"] = -1

        drop_amps = []

        def process_amp(amp):
            try:
                if applied is not None:
                    a, b, c = sample_with_stim(self, max_bins, applied, input=v * amp)
                else:
                    a, b, c = rslds.sample(max_bins, with_noise=False, input=v * amp)
                _predicted_dia = self.predict_dia(b)
                if mean_sub:
                    _predicted_dia = _predicted_dia - np.mean(_predicted_dia)
            except ValueError:
                drop_amps.append(amp)
                return None, amp

            s0 = pre_bins + pad_bins
            sf = pre_bins + dur_bins - pad_bins
            try:
                breaths = burst_stats_dia(_predicted_dia[s0:sf], sr=1 / binsize, dia_thresh=dia_thresh)
            except ValueError:
                breaths = pd.DataFrame()    
            breaths["stim_amp"] = amp
            return breaths, None

        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_amp)(amp) for amp in tqdm(amps)
        )

        for breaths, dropped_amp in results:
            if dropped_amp is not None:
                drop_amps.append(dropped_amp)
            elif breaths is not None:
                all_breaths = pd.concat([all_breaths, breaths])

        stim_amp_values = np.concatenate([[-1], amps])
        df = all_breaths.pivot_table(index="stim_amp")
        df["n_breaths"] = all_breaths.groupby("stim_amp")["on_sec"].count()
        df = df.reindex(stim_amp_values, fill_value=0)
        df["resp_rate"] = df["n_breaths"] / (dur - 2 * pad_time)

        df.drop(
            columns=["on_sec", "off_sec", "on_samp", "off_samp", "pk_samp", "pk_time"],
            inplace=True,
        )
        df["eid"] = self.eid
        df["subject"] = self.subject
        df["sequence"] = self.sequence
        df["genotype"] = self.genotype
        df.reset_index(inplace=True)
        cols = df.columns.tolist()
        # cols = [x for x in cols if x != "stim_amp"]
        cols = list(set(cols)-set(['stim_amp','eid','subject','sequence','genotype']))
        # Set all values to nan if stim amp in drop_amps
        for amp in drop_amps:
            df.loc[df["stim_amp"] == amp, cols] = 0

        return df

    def sim_reset_curve(
        self,
        applied=None,
        nreps=100,
        stim_dur=0.05,
        ISI=3,
        stim_amplitude=1,
        plot_tgl=False,
    ):
        binsize = self.binsize
        rslds = self.rslds
        pre_time = 50
        post_time = 30
        # Have to add the sigma in case the respiratory rate is too close to the ISI
        sigma = np.random.uniform(-0.5, 0.5, size=nreps)
        onset_times = sigma + pre_time + np.arange(nreps) * ISI
        intervals = np.c_[onset_times, onset_times + stim_dur]
        tbins = np.arange(0, np.max(intervals) + post_time, binsize)
        v = _get_stim_vector(intervals, tbins, stim_amplitudes=stim_amplitude)
        # plt.plot(tbins, v)

        # If applied is provided, assume it is a sink of the form (V,sink)
        if applied is not None:
            ks, xs, ys = sample_with_stim(self, tbins.shape[0], applied, input=v)
        else:
            ks, xs, ys = rslds.sample(v.shape[0], with_noise=False, input=v)

        predicted_dia = self.predict_dia(xs)

        # Wrap in try catch in case there is not reasonable breaths in either the stim or control
        try:
            breaths = burst_stats_dia(predicted_dia, 1.0 / binsize)
            breaths["times"] = breaths["on_sec"]
            breaths = AlfBunch.from_df(breaths)

            xstim, ystim, xcontrol, ycontrol = plot_reset_curve(
                breaths, intervals[:, 0], plot_tgl=plot_tgl
            )
        except Exception:
            xstim, ystim, xcontrol, ycontrol = (np.nan,) * 4

        return xstim, ystim, xcontrol, ycontrol

    def sim_phasic_stims(
        self,
        stim_amplitude=1,
        dia_thresh=0.5,
        mode="insp",
        pre_time=10,
        direction="both",
        applied=None,
        tmax = 20
    ):
        rslds = self.rslds
        binsize = self.binsize

        pre_bins = int(pre_time / binsize)
        # Initialize
        
        max_bins = int(tmax / binsize)
        t = int(pre_time / binsize)
        ks = np.full(max_bins, dtype="int", fill_value=-1)
        xs = np.full((max_bins, rslds.D), fill_value=np.nan)
        ys = np.full((max_bins, rslds.N), fill_value=np.nan)
        dia_predicted = np.zeros(max_bins)
        triggered = np.zeros(max_bins, dtype="bool")

        k, x, y = rslds.sample(pre_bins, with_noise=False, input=None)
        ks[:pre_bins], xs[:pre_bins, :], ys[:pre_bins, :] = rslds.sample(
            pre_bins, with_noise=False, input=None
        )
        dia_predicted[:pre_bins] = self.predict_dia(x)

        t_since_triggered = 0
        trigger_armed = False
        try:
            while t < max_bins:
                # Get the last 10 time bins
                k = ks[t - 10 : t]
                x = xs[t - 10 : t, :]
                y = ys[t - 10 : t, :]

                # Trigger stim logic
                above_thresh = dia_predicted[t - 1] > dia_thresh
                rising = dia_predicted[t - 1] > dia_predicted[t - 2]
                falling = dia_predicted[t - 1] < dia_predicted[t - 2]

                if above_thresh and mode == "insp":
                    triggered[t] = True
                elif not above_thresh and mode == "exp":
                    triggered[t] = True
                else:
                    triggered[t] = False

                if direction == "rising" and not rising:
                    triggered[t] = False
                elif direction == "falling" and not falling:
                    triggered[t] = False

                # if triggered[t-1] and t_since_triggered<0.010:
                #     triggered[t] = True
                # elif above_thresh:
                #     triggered[t] = True
                #     t_since_triggered = 0
                # else:
                #     triggered[t] = False

                # if not triggered[t] and not above_thresh:
                #     trigger_armed = True

                # Create stim vector
                v = stim_amplitude if triggered[t] else 0
                v = np.array([[v]])

                # Sample the next state and observation
                if applied is not None:
                    _k, _x, _y = sample_with_stim(
                        self, 1, applied, input=v, prefix=(k, x, y)
                    )
                else:
                    _k, _x, _y = rslds.sample(
                        1, with_noise=False, input=v, prefix=(k, x, y)
                    )
                ks[t] = _k[0]
                xs[t] = _x
                ys[t] = _y

                # Map to dia
                _dia = self.predict_dia(xs[t, np.newaxis])
                dia_predicted[t] = _dia[0]
                t += 1
                t_since_triggered += binsize
        except Exception as e:
            print(f"Exception during sim_phasic_stims: {e}")
            ks[t:] =  np.nan
            xs[t:,:] =  np.nan
            ys[t:,:] =  np.nan
            dia_predicted[t:] = 0
            triggered[t:] = False
            
        output = {
            "ks": ks,
            "xs": xs,
            "ys": ys,
            "dia_predicted": dia_predicted,
            "triggered": triggered,
            "stim_amplitude": stim_amplitude,
            "dia_thresh": dia_thresh,
            "binsize": binsize,
            "stim_amplitude": stim_amplitude,
        }
        return output

    def make_streamplot(self, amp=0, ax=None, xlim=None, ylim=None, figscale=1.5,density=0.6):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)
        if xlim is None:
            xlim = ax.get_xlim()[1]
        if ylim is None:
            ylim = ax.get_ylim()[1]
        stability_markers = {"stable": "o", "unstable": "x", "saddle": "^"}
        lw = 0.5 * figscale
        arrowsize = 0.5 * figscale
        markersize = 4 * figscale
        cmap = mcolors.ListedColormap(self.k_colors)
        X, Y, U, V, K = self.compute_vectorfield(
            xlim=xlim, ylim=xlim, res=0.1, input_strength=amp
        )
        ax.streamplot(
            X,
            Y,
            U,
            V,
            color=K,
            linewidth=lw,
            cmap=cmap,
            density=density,
            arrowsize=arrowsize,
        )
        fp, stability = self.compute_fixedpoints(input_strength=amp)
        for ii in range(2):
            ax.plot(
                *fp[ii],
                color=cmap(ii),
                marker=stability_markers[stability[ii]],
                markersize=markersize,
                mew=1,
                mec="k",
            )
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_aspect("equal")
        return ax

    def plot_example_stimmed_dynamics(
        self, amps=None, figscale=1, xlim=None, ylim=None
    ):
        if amps is None:
            amps = self.generate_wellspaced_amps()

        if xlim is None:
            xlim = np.percentile(np.abs(self.q.mean_continuous_states[0]), 99) * 1.1
            xlim = np.round(xlim, 0)
            ylim = xlim

        f, axs = plt.subplots(
            ncols=len(amps),
            figsize=(len(amps) * figscale, figscale * 1.5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if isinstance(axs, plt.Axes):
            axs = [axs]
        for amp, ax in zip(amps, axs):
            ax = self.make_streamplot(amp, ax, xlim=xlim, ylim=ylim, figscale=figscale)

            tmax = 3
            stim_time = np.arange(0, tmax, self.binsize)
            v = _get_stim_vector(np.array([[0, tmax]]), stim_time, stim_amplitudes=amp)
            try:
                zs, xs, ys = self.rslds.sample(
                    stim_time.shape[0], with_noise=False, input=v
                )
            except ValueError:
                xs = np.full((stim_time.shape[0], self.rslds.D), np.nan)
            ax.plot(*xs.T, color="k", alpha=0.75, lw=figscale)

            if amp > 0:
                ax.set_title(f"Stim: {amp} {self.stim_units}")
            else:
                ax.set_title("No stim")
        f.supxlabel("$x_{1}$", fontsize="small", y=-0.02)
        f.supylabel("$x_{2}$", fontsize="small", x=-0.02)
        ax.set_xticks([-xlim, 0, xlim])
        ax.set_yticks([-ylim, 0, ylim])

    def assign_phase_to_k(self):
        """
        Assign respiratory phase to each k state based on diaphragm amplitude
        """
        assert self.rslds.K == 2, "Map only to inspiration and expiration"
        dia2 = remap_time_basis(self.physiology.dia, self.physiology.times, self.tbins)
        k = np.argmax(self.q.mean_discrete_states[0], 1)
        # get mean value of dia2 for each k state
        mean_dia_k = np.zeros(self.rslds.K)
        for kk in range(self.rslds.K):
            mean_dia_k[kk] = np.nanmedian(dia2[k == kk])

        if mean_dia_k[0] > mean_dia_k[1]:
            self.k_phase = ["insp", "exp"]
        else:
            self.k_phase = ["exp", "insp"]
        self.k_colors = [PHASE_MAP[phase] for phase in self.k_phase]
        return self.k_phase

    def compute_vectorfield(self, xlim=30, ylim=30, res=0.1, input_strength=0):
        model = copy.deepcopy(self.rslds)
        x = np.arange(-xlim, xlim, res)
        y = np.arange(-ylim, ylim, res)
        X, Y = np.meshgrid(x, y)
        K = np.zeros_like(X)

        xy = np.c_[X.ravel(), Y.ravel()]

        z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for k, (A, b, v) in enumerate(
            zip(model.dynamics.As, model.dynamics.bs, model.dynamics.Vs)
        ):
            ival = np.ones(model.M) * input_strength
            v = v.dot(ival)
            dxydt_m = xy.dot(A.T) + b + v - xy

            U.ravel()[z == k] = dxydt_m[z == k, 0]
            V.ravel()[z == k] = dxydt_m[z == k, 1]
            K.ravel()[z == k] = k
        return (X, Y, U, V, K)

    def plot_example_phasic(
        self, applied=None, direction="both", amp=None, dia_thresh=0.5
    ):
        f, axs = plt.subplots(
            nrows=2, figsize=(3, 2), constrained_layout=True, sharex=True, sharey=True
        )
        if amp is None:
            amp = self.generate_wellspaced_amps()[-2]
        for ax, phase in zip(axs, ["insp", "exp"]):
            stim_response = self.sim_phasic_stims(
                stim_amplitude=amp,
                dia_thresh=dia_thresh,
                mode=phase,
                direction=direction,
                applied=applied,
            )
            plot_sim_phasic_stims(stim_response, ax=ax, laser_color=self.laser_color)
            ax.set_xlim(8, 12)
            ax.set_ylim(0, None)
            ax.set_ylabel("")
            ax.axhline(dia_thresh, color="k", lw=0.5, ls=":")

        direction_label = "" if direction == "both" else f" {direction}"
        axs[0].set_xlabel("")
        axs[0].set_title(f"Insp. triggered {direction} stim")
        axs[1].set_title(f"Exp. triggered {direction} stim")
        axs[0].text(
            10,
            ax.get_ylim()[1],
            f"{amp}{self.stim_units}",
            weight="bold",
            fontsize="xx-small",
            va="top",
            ha="right",
        )
        f.supylabel(r"$\int{Dia.}$ Predicted", fontsize="small")
        return f

    def compute_reset_curve_sweep(self, amps=None, stim_durs=None, nreps=200):
        if amps is None:
            amps = [0, 2, 10, 20]
        if stim_durs is None:
            stim_durs = [0.01, 0.05, 0.1]
        df = pd.DataFrame()
        for amp, stim_dur in product(amps, stim_durs):
            x_stim, y_stim, _, _ = self.sim_reset_curve(
                stim_dur=stim_dur, nreps=nreps, plot_tgl=False, stim_amplitude=amp
            )
            _df = pd.DataFrame()
            _df["y_stim"] = y_stim
            _df["x_stim"] = x_stim
            _df["stim_amp"] = amp
            _df["stim_dur"] = stim_dur
            df = pd.concat([df, _df])
        df.reset_index(drop=True, inplace=True)
        df["eid"] = self.eid
        df["subject"] = self.subject
        df["sequence"] = self.sequence
        df["genotype"] = self.genotype

        return df

    def plot_phasic_parameter_sweep(self, amps=None, threshs=None):
        if amps is None:
            amps = self.generate_wellspaced_amps()
        if threshs is None:
            threshs = [0.1, 0.2, 0.5, 1, 2]

        pre_time = 10

        df = pd.DataFrame()
        figs = []
        for phase in ["insp", "exp"]:
            for direction in ["both", "rising", "falling"]:
                f, axs = plt.subplots(
                    nrows=len(amps),
                    ncols=len(threshs),
                    figsize=(len(threshs) * 2, len(amps) * 2),
                    constrained_layout=True,
                    sharex=True,
                    sharey=True,
                )
                for ii, amp in enumerate(amps):
                    for jj, thresh in enumerate(threshs):
                        stim_response = self.sim_phasic_stims(
                            stim_amplitude=amp,
                            dia_thresh=thresh,
                            mode=phase,
                            pre_time=pre_time,
                            direction=direction,
                        )
                        t = (
                            np.arange(len(stim_response["dia_predicted"]))
                            * self.binsize
                        )

                        plot_sim_phasic_stims(
                            stim_response, ax=axs[ii, jj], laser_color=self.laser_color
                        )
                        if jj == 0:
                            axs[ii, jj].set_ylabel(
                                f"{amp} {self.stim_units}", weight="bold"
                            )
                        else:
                            axs[ii, jj].set_ylabel("")
                        if ii == 0:
                            axs[ii, jj].set_title(f"Threshold: {thresh}", weight="bold")

                        axs[ii, jj].set_xlim(pre_time - 4, pre_time + 4)
                        axs[ii, jj].set_ylim(0, None)

                        # Drop state variables
                        for key in ["ks", "xs", "ys"]:
                            stim_response.pop(key, None)

                        _df = pd.DataFrame(stim_response)
                        _df["t"] = t
                        _df["stim_phase"] = phase
                        _df["direction"] = direction
                        df = pd.concat([df, _df])
                    suffix = direction
                else:
                    suffix = ""

                f.suptitle(
                    f"{phase.capitalize()}. {suffix} triggered stim", fontsize=16
                )
                f.supylabel(r"$\int{Dia.}$ Predicted", fontsize="small")
                figs.append(f)
        df.reset_index(drop=True, inplace=True)
        return (df, figs)

    def get_slow_exp_direction(self):
        # Get eigen vectors of exp Jacobian
        k_exp = self.k_phase.index("exp")
        A =  np.eye(self.rslds.D)-self.rslds.dynamics.As[k_exp]
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Get eigen vector with smallest eigenvalue
        idx = np.argmin(np.abs(eigenvalues))
        v_e = eigenvectors[:, idx]

        # Get direction of the eigenvector fromt the projeciton as it evolves
        # in the latent space
        sinks = self.compute_putative_sinks_from_latent()
        aa = sinks['pre_I']-sinks['I_off']
        if aa @ v_e < 0:
            v_e = -v_e

        return v_e
    
    def get_fast_exp_direction(self):
        # Get eigen vectors of exp Jacobian
        k_exp = self.k_phase.index("exp")
        A =  np.eye(self.rslds.D)-self.rslds.dynamics.As[k_exp]
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Get eigen vector with largest eigenvalue
        idx = np.argmax(np.abs(eigenvalues))
        v_e = eigenvectors[:, idx]

        # Get direction of the eigenvector fromt the projeciton as it evolves
        # in the latent space
        sinks = self.compute_putative_sinks_from_latent()
        aa = sinks['I_off']-sinks['I_peak']
        if aa @ v_e < 0:
            v_e = -v_e

        return v_e

    def apply_uniform_stim_field(self, angle_wrt_e=180, active_phase="both"):
        """
        Set the stimulus field to be a uniform vector field in a direction relative to the slowest
        eigenvector of the Expiratory dynamics

        Scales the stimulus field by the amplitude of the fitted stimulus response

        Modifies the rslds model in place,stores the original stimulus field in self.original_Vs

        Args:
            angle_wrt_e: angle in degrees to rotate the slowest eigenvector
            active_phase: 'both', 'exp', or 'insp' to set the stimulus field for
        """
        assert active_phase in [
            "both",
            "exp",
            "insp",
        ], "active_phase must be 'both', 'exp', or 'insp'"

        k_exp = self.k_phase.index("exp")
        k_insp = self.k_phase.index("insp")

        # Rotate v_e by angle_wrt_e
        v_e = self.get_slow_exp_direction()
        v_e = rotate_vector(v_e, angle_wrt_e)
        # angle_wrt_e = np.deg2rad(angle_wrt_e)
        # new_angle = np.arctan2(v_e[1], v_e[0]) + angle_wrt_e
        # v_e = np.array([np.cos(new_angle), np.sin(new_angle)])

        # Scale by the fit stim response if it exists
        if not np.all(self.rslds.dynamics.Vs == 0):
            a = np.linalg.norm(self.rslds.dynamics.Vs[k_exp])
            v_e *= a

        if active_phase == "both":
            self.rslds.dynamics.Vs[k_exp, :] = v_e[:, np.newaxis]
            self.rslds.dynamics.Vs[k_insp, :] = v_e[:, np.newaxis]
        elif active_phase == "exp":
            self.rslds.dynamics.Vs[k_exp, :] = v_e[:, np.newaxis]
            self.rslds.dynamics.Vs[k_insp, :] *= 0
        elif active_phase == "insp":
            self.rslds.dynamics.Vs[k_exp, :] *= 0
            self.rslds.dynamics.Vs[k_insp, :] = v_e[:, np.newaxis]
        else:
            raise ValueError("active_phase must be 'both', 'exp', or 'insp'")

    def compare_learned_angle_to_eig(self):
        e = self.get_slow_exp_direction()
        dot_products = {"insp": 0, "exp": 0}
        for k in range(self.rslds.K):
            v = self.rslds.dynamics.Vs[k]
            v = v / np.linalg.norm(v)
            dot_products[self.k_phase[k]] = e @ v
        return dot_products

    def reset_Vs(self):
        if hasattr(self, "original_Vs"):
            self.rslds.dynamics.Vs = self.original_Vs

    def sweep_stim_field(self, angles=None, amps=None, plot_tgl=True):
        if amps is None:
            amps = self.generate_wellspaced_amps()
        if angles is None:
            angles = np.arange(0, 360, 5)

        v_e = self.get_slow_exp_direction()

        df = pd.DataFrame()
        for angle in angles:
            for mode in ["uniform", "exp", "insp"]:
                stim_vector = rotate_vector(np.real(v_e), angle)
                if mode in ["insp", "exp"]:
                    stim = Stimulus(
                        "sigmoid", k=self.k_phase.index(mode), stim_vector=stim_vector
                    )
                else:
                    stim = Stimulus(mode, stim_vector=stim_vector)
                rez = self.compute_amp_sweep(applied=stim, amps=amps)
                rez["angle"] = angle
                rez["phase"] = mode
                df = pd.concat([df, rez])
        df.reset_index(drop=True, inplace=True)
        df.query("stim_amp >= 0", inplace=True)

        if plot_tgl:
            p = (
                so.Plot(df, x="angle", y="resp_rate", color="stim_amp")
                .facet(col="phase")
                .add(so.Line(), group="stim_amp", linewidth="stim_amp")
                .add(so.Dot(), group="stim_amp", pointsize="amp")
                .scale(color="winter", x=so.Continuous().tick(at=[0, 90, 180, 270]))
                .label(x="Angle (degrees)", y="Resp. Rate (Hz)")
            ).plot()
            return df, p
        else:
            return df

    def apply_converging_stim_field(self):
        k_exp = self.k_phase.index("exp")
        k_insp = self.k_phase.index("insp")

        # Rotate v_e by angle_wrt_e
        v_e = self.get_slow_exp_direction()
        field_E = rotate_vector(v_e, 0)
        field_I = rotate_vector(v_e, 180)
        self.rslds.dynamics.Vs[k_exp, :] = field_E[:, np.newaxis]
        self.rslds.dynamics.Vs[k_insp, :] = field_I[:, np.newaxis]

    def compute_putative_sinks_from_latent(self, plot_tgl=False):
        projection = self.q.mean_continuous_states[0]
        phi2 = remap_time_basis(self.phi, self.phi_t, self.tbins)

        I_off_mask = np.logical_or(phi2 > np.pi - 0.1, phi2 < -np.pi + 0.1)

        pre_I_mask = np.logical_and(phi2 > -np.pi / 2 - 0.1, phi2 < -np.pi / 2 + 0.1)

        I_on_mask = np.logical_and(phi2 > -0.1, phi2 < 0.1)
        I_peak_mask = np.logical_and(phi2 > np.pi / 2 - 0.1, phi2 < np.pi / 2 + 0.1)
        I_off_latent = projection[I_off_mask, :]
        pre_I_latent = projection[pre_I_mask, :]
        I_on_latent = projection[I_on_mask, :]
        I_peak_latent = projection[I_peak_mask, :]

        mean_I_off_latent = np.nanmean(I_off_latent, axis=0)
        mean_pre_I_latent = np.nanmean(pre_I_latent, axis=0)
        mean_I_on_latent = np.nanmean(I_on_latent, axis=0)
        mean_I_peak_latent = np.nanmean(I_peak_latent, axis=0)

        if plot_tgl:
            f = plt.figure()
            plt.plot(projection[:, 0], projection[:, 1])
            plt.plot(*I_off_latent.T, ".", color="k", alpha=0.01)
            plt.plot(*pre_I_latent.T, ".", color="b", alpha=0.01)
            plt.plot(*I_on_latent.T, ".", color="r", alpha=0.01)
            plt.plot(*I_peak_latent.T, ".", color="g", alpha=0.01)

            plt.plot(*mean_I_off_latent, "o", color="k", alpha=0.5, ms=10, mec="w")
            plt.plot(*mean_pre_I_latent, "o", color="b", alpha=0.5, ms=10, mec="w")
            plt.plot(*mean_I_on_latent, "o", color="r", alpha=0.5, ms=10, mec="k")
            plt.plot(*mean_I_peak_latent, "o", color="g", alpha=0.5, ms=10, mec="k")

            bins = np.arange(-np.pi, np.pi, 0.1)
            f = plt.figure()
            ax = f.add_subplot(111, projection="polar")
            ax.hist(phi2[I_off_mask], bins)
            ax.hist(phi2[pre_I_mask], bins)
            ax.hist(phi2[I_on_mask], bins)
            ax.hist(phi2[I_peak_mask], bins)

        out = {
            "I_off": mean_I_off_latent,
            "pre_I": mean_pre_I_latent,
            "I_on": mean_I_on_latent,
            "I_peak": mean_I_peak_latent,
        }
        self.latent_landmarks = out
        return out

    def get_dynamics(self):
        A_insp = self.rslds.dynamics.As[self.k_phase.index("insp")]
        A_exp = self.rslds.dynamics.As[self.k_phase.index("exp")]

        eigenvalues_insp, eigenvectors_insp = np.linalg.eig(A_insp-np.eye(self.rslds.D))
        eigenvalues_exp, eigenvectors_exp = np.linalg.eig(A_exp-np.eye(self.rslds.D))

        out = {
            'insp':{
                "eigenvalues": eigenvalues_insp,
                "eigenvectors": eigenvectors_insp,
            },
            'exp':{
                "eigenvalues": eigenvalues_exp,
                "eigenvectors": eigenvectors_exp,
            }
        }
        return(out)


def run_recording(ii, suffix=""):
    # Load
    rec = Rec(one, EIDS_REGEN[ii])
    rec.load_rslds(suffix=suffix)
    rec.fit_sim_dia('SVR')

    # First plot learned stimulus
    rec.plot_stim_field()
    fn = rec.model_path.joinpath("learned_stim_field.pdf")
    plt.savefig(fn)

    # Plot how the learned field alters the dynamics
    rec.plot_example_stimmed_dynamics(amps=[0.0, 0.5, 1, 1.2])
    fn = rec.model_path.joinpath("example_stimmed_dynamics.pdf")
    plt.savefig(fn)

    # Plot example phasic stimulations
    rec.plot_example_phasic(amp=0.1)
    fn = rec.model_path.joinpath("example_phasic.pdf")
    plt.savefig(fn)

    # Plot example holds
    rec.plot_amp_sweep()
    fn = rec.model_path.joinpath("example_holds.pdf")
    plt.savefig(fn)

    # Plot summary of the amplitude sweep
    amplitude_sweep_rez = rec.compute_amp_sweep()
    p = plot_summary_amp_sweep(amplitude_sweep_rez, rec.stim_units)
    fn = rec.model_path.joinpath("amp_sweep_summary.pdf")
    p.save(fn)
    fn = rec.model_path.joinpath("amp_sweep_summary.pqt")
    amplitude_sweep_rez.to_parquet(fn)

    # Plot phasic stimulations for parameter sweep
    print("Plotting phasic parameter sweep")
    df, figs = rec.plot_phasic_parameter_sweep()
    phases = [
        "insp_both",
        "insp_rising",
        "insp_falling",
        "exp_both",
        "exp_rising",
        "exp_falling",
    ]
    for phase, f in zip(phases, figs):
        fn = rec.model_path.joinpath(f"phasic_sweep_{phase}.pdf")
        f.savefig(fn)
    fn = rec.model_path.joinpath("phasic_sweep_learned.pqt")
    df.to_parquet(fn)

    print("Computing reset curve sweep")
    # Compute reset curves for parameter sweep
    reset_curves = rec.compute_reset_curve_sweep()
    fn = rec.model_path.joinpath("reset_curve_sweep.pqt")
    reset_curves.to_parquet(fn)

    # Sweep a uniform stimulus field
    print("Sweeping uniform stimulus field")
    df, p = rec.sweep_stim_field()

    p.save(rec.model_path.joinpath("sweep_uniform_stim_field.pdf"))


@click.group()
def main():
    pass


@main.command()
@click.argument("ii", type=int)
@click.option("--norm", "-n", is_flag=True, default=False)
def single(ii, norm):
    """
    Run the analysis for a single recording
    """
    suffix = "_norm" if norm else ""
    run_recording(ii, suffix=suffix)


@main.command()
@click.option("--norm", "-n", is_flag=True, default=False)
def batch(norm):
    """
    Run the analysis for all recordings
    """
    for ii in range(len(EIDS_REGEN)):
        suffix = "_norm" if norm else ""
        run_recording(ii, suffix=suffix)


if __name__ == "__main__":
    main()
