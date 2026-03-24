import os

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('./VLAD')
from pathlib import Path

from utils import one, EIDS_REGEN,set_style

import pandas as pd
import seaborn.objects as so

# re-use the Rec class and utilities from the original analysis module
from analyze_single_rec import (
    Rec as BaseRec,
    sample_with_stim,
    sample_x_partition_distance,
    _get_stim_vector,
    plot_reset_curve,
    Stimulus
)
from cibrrig.preprocess.physiology import burst_stats_dia
from cibrrig.plot import AlfBunch
set_style()


def estimate_limit_cycle_center(rslds, nsteps=5000):
    """Simulate the model for a while and return the mean of the latent trajectory.

    The first half of the simulation is treated as a transient and discarded.
    """
    ks, xs, ys = rslds.sample(nsteps, with_noise=False, input=None)
    return xs[int(nsteps // 2) :].mean(axis=0)


def orthogonal_vectors(v: np.ndarray):
    """Return two unit vectors orthogonal to *v* (2‑D only).

    The first vector is obtained by rotating *v* by +90 degrees, the second by
    -90 degrees.  If *v* is zero the pair of zero vectors is returned.
    """
    if np.linalg.norm(v) == 0:
        return np.zeros_like(v), np.zeros_like(v)
    perp = np.array([-v[1], v[0]])
    perp = perp / np.linalg.norm(perp)
    return perp, -perp


def choose_orientation(
    x: np.ndarray, velocity: np.ndarray, center: np.ndarray, inward: bool = True
) -> np.ndarray:
    """Pick the orthogonal unit vector pointing toward (or away from) the limit-cycle centre.

    *x* is the current latent state, *velocity* the difference between the last
    two states, and *centre* the centre of the limit cycle.  If *inward* is
    ``True`` the returned vector points toward *centre*; otherwise it points
    away.
    """
    perp, opp = orthogonal_vectors(velocity)
    # determine which of the two candidates points inward
    if np.dot(center - x, perp) > 0:
        inward_vec = perp
        outward_vec = -perp
    else:
        inward_vec = -perp
        outward_vec = perp
    return inward_vec if inward else outward_vec


class Rec(BaseRec):
    """Extension of the original ``Rec`` with inward/outward stimulation logic."""

    def compute_limit_cycle_center(self, nsteps: int = 5000) -> np.ndarray:
        return estimate_limit_cycle_center(self.rslds, nsteps=nsteps)

    def sim_inout_stim(
        self,
        pre_time: float = 10,
        tmax: float = 20,
        period: float = 1,
        jitter: float = 0,
        pulse_dur=0.05,
        stim_amplitude: float = 1,
        inward: bool = True,
        applied=None,
        input=None,
    ):
        """Closed‑loop sampling with periodically‑reoriented uniform field.

        A stimulus pulse is delivered on a regular schedule (or, if ``input`` is
        provided, whenever ``input[t]>0``) and the field is rotated to be
        orthogonal to the current velocity.  ``inward`` controls whether the
        pulse is directed toward or away from the estimated limit‑cycle
        centre.

        Returns a dictionary containing the latent trajectory, predicted
        diaphragm, stimulus directions, and other bookkeeping arrays.
        """
        binsize = self.binsize
        rslds = self.rslds
        pulse_dur_bins = int(pulse_dur / binsize)
        assert pulse_dur < (1 / period), "Pulse duration must be less than the inter-pulse interval"

        pre_bins = int(pre_time / binsize)
        max_bins = int(tmax / binsize)
        # use float dtype for ks so we can fill with NaNs on exception
        ks = np.full(max_bins, dtype="int", fill_value=-1)
        xs = np.full((max_bins, rslds.D), fill_value=np.nan)
        ys = np.full((max_bins, rslds.N), fill_value=np.nan)
        v = np.full((max_bins, rslds.M), fill_value=0)
        dia_predicted = np.zeros(max_bins)
        stim_dirs = np.zeros((max_bins, rslds.D))

        center = self.compute_limit_cycle_center()

        # if no external input provided, use a zero vector for transitions
        if input is None:
            # shape (max_bins, M) where M is dimensionality of RSN input
            input = np.zeros((max_bins, rslds.M))

        # initialise history with an unstimulated sample
        ks[:pre_bins], xs[:pre_bins], ys[:pre_bins] = rslds.sample(
            pre_bins, with_noise=False, input=None
        )
        dia_predicted[:pre_bins] = self.predict_dia(xs[:pre_bins])

        period_bins = int(period / binsize) if period is not None else None
        next_pulse = pre_bins
        t = pre_bins
        stim_times = []
        stim_start_points = []
        stim_dirs = []
        # try:
        while t < max_bins:
            k = ks[t - 10 : t]
            x = xs[t - 10 : t, :]
            y = ys[t - 10 : t, :]
            if t >= next_pulse:
                trigger = True
                next_pulse += period_bins
            else:
                trigger = False

            if trigger:
                vel = xs[t - 1] - xs[t - 2]
                orient = choose_orientation(xs[t - 1], vel, center, inward=inward)
                stim_times.append(t-1)
                stim_start_points.append(xs[t - 1])
                stim_dirs.append(orient)
                applied = Stimulus('uniform',stim_vector=orient) 
                # Sample the next state and observation
                v[t:t + pulse_dur_bins] = stim_amplitude
                _k, _x, _y = sample_with_stim(
                    self, pulse_dur_bins, applied, input=v[t:t + pulse_dur_bins],prefix=(k,x,y)
                )
                ks[t:t + pulse_dur_bins] = _k[0]
                xs[t:t + pulse_dur_bins] = _x
                ys[t:t + pulse_dur_bins] = _y

                t += pulse_dur_bins
                
            else:
                # Sample the next state and observation without stimulation
                _k, _x, _y = rslds.sample(1, with_noise=False, input=v[t:t + 1], prefix=(k, x, y))
                ks[t] = _k[0]
                xs[t] = _x
                ys[t] = _y
                t += 1
        dia_predicted = self.predict_dia(xs)

        return {
            "ks": ks,
            "xs": xs,
            "ys": ys,
            "v": v,
            "dia_predicted": dia_predicted,
            "stim_dirs": np.array(stim_dirs),
            "stim_times": np.array(stim_times) * binsize,  # convert to seconds
            "stim_start_points": np.array(stim_start_points),
            "binsize": binsize,
            "amplitude": stim_amplitude,
        }

    def plot_limit_cycle_with_stim(self, sim_output, ax=None,scale=10,max_vectors=10,color='b',width=0.005,lw=1):
        """Show the latent trajectory with stimulus arrows superimposed."""
        xs = sim_output["xs"][:int(3/self.binsize),:]  # skip the first few seconds to avoid transients
        stim_ons = sim_output["stim_start_points"]
        stim_dirs = sim_output["stim_dirs"]
        amplitude = sim_output["amplitude"]

        skip = max(1, len(stim_dirs) // max_vectors)  # determine how many vectors to plot
        stim_dirs = stim_dirs[::skip]
        stim_ons = stim_ons[::skip,:]

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(xs[:, 0], xs[:, 1], "-k", lw=lw)
        for idx in range(len(stim_dirs)):
            ax.quiver(
                stim_ons[idx, 0],
                stim_ons[idx, 1],
                stim_dirs[idx, 0] * amplitude,
                stim_dirs[idx, 1] * amplitude,
                angles="xy",
                scale_units="xy",
                scale=1/scale,
                color=color,
                width=width,
            )
        ax.set_xlabel("$x_{1}$")
        ax.set_ylabel("$x_{2}$")
        ax.set_aspect("equal")
        return ax

    def compute_reset_curve_inout(
        self,
        sim,
        plot_tgl: bool = False,
        applied=None,
    ):

        predicted_dia = sim["dia_predicted"]

        breaths = burst_stats_dia(predicted_dia, 1.0 / self.binsize)
        breaths['times'] = breaths['on_sec']
        breath = AlfBunch.from_df(breaths)
        stim_times = sim["stim_times"]

        t0 = breath.times[0]
        tf = breath.times[-2]
        valid = (stim_times > t0) & (stim_times < tf)
        stim_times = stim_times[valid]

        xstim, ystim, xcontrol, ycontrol = plot_reset_curve(
            breath, stim_times, plot_tgl=plot_tgl
        )
        return xstim, ystim, xcontrol, ycontrol


def plot_example(one,eid):
    """
    Plot an example of the limit cycle with inward and outward stimulation vectors.
    """
    rec = Rec(one, eid)
    rec.load_rslds(suffix='_norm')  # use default suffix
    rec.fit_sim_dia('SVR')
    
    sim_inward = rec.sim_inout_stim(pre_time=3, tmax=100, period=1, stim_amplitude=4, inward=True)
    sim_outward = rec.sim_inout_stim(pre_time=3, tmax=100, period=1, stim_amplitude=4, inward=False)

    f,ax = plt.subplots(figsize=(2.5,2))
    rec.plot_limit_cycle_with_stim(sim_outward,scale=1,max_vectors=10,color='tab:blue',ax=ax)
    rec.plot_limit_cycle_with_stim(sim_inward,scale=1,max_vectors=10,color='tab:red',ax=ax)
    plt.savefig('limit_cycle_with_stim.pdf')

def compute_reset_curves(rec,amplitude=1,tmax=100):
    sim_inward = rec.sim_inout_stim(pre_time=3, tmax=tmax, period=1, stim_amplitude=amplitude, inward=True)
    sim_outward = rec.sim_inout_stim(pre_time=3, tmax=tmax, period=1, stim_amplitude=amplitude, inward=False)
    DF = pd.DataFrame()
    for sim,label in zip([sim_inward, sim_outward], ['inward', 'outward']):
        breaths = burst_stats_dia(sim['dia_predicted'], 1.0 / rec.binsize)
        duty_cycle = (breaths['duration_sec'] / breaths['IBI']).mean()
        df = pd.DataFrame()
        xstim, ystim, xcontrol, ycontrol = rec.compute_reset_curve_inout(sim,plot_tgl=True)
        df['xstim'] = xstim
        df['ystim'] = ystim
        df['direction'] = label
        df['duty_cycle'] = duty_cycle
        DF = pd.concat((DF, df), ignore_index=True)
    DF['amplitude'] = sim_inward['amplitude']
    DF['eid'] = rec.eid
    DF['genotype'] = rec.genotype
    return DF

def get_reset_curves():
    fn = Path('reset_curves_inout.pqt')
    if not os.path.exists(fn):
        DF = pd.DataFrame()
        for eid in EIDS_REGEN:
            rec = Rec(one, eid)
            rec.load_rslds(suffix='_norm')  # use default suffix
            rec.fit_sim_dia('SVR')
            try:
                df = compute_reset_curves(rec,amplitude=1,tmax=300)
                DF = pd.concat((DF, df), ignore_index=True)
            except Exception as e:
                print(f"Error processing {eid}: {e}")

        DF.to_parquet('reset_curves_inout.pqt')
    DF = pd.read_parquet(fn)
    return DF



if __name__ == "__main__":

    plot_example(one, EIDS_REGEN[1])

    df = get_reset_curves()
    df = df.query('ystim<3')
    idx = df['duty_cycle']<1
    df = df[idx]
    df["x_stim_binned"] = pd.cut(df["xstim"], bins=np.arange(0, 2.01, 0.05), labels=np.arange(0.0, 2.0, 0.05)).astype(float)
    by_res = df.groupby(['eid','direction','x_stim_binned'])['ystim'].mean().reset_index()
    p=(
        so.Plot(by_res,x='x_stim_binned',y='ystim',color='direction')
        .add(so.Line(linewidth=1), so.Agg('median'))
        # .add(so.Line(alpha=0.5,linewidth=0.5),group='eid')
        .add(so.Band(), so.Est(errorbar=('pi', 50)))
        # .add(so.Band(), so.Est())
        .scale(color={'inward':'tab:red','outward':'tab:blue'})
        .limit(x=(0,1.2),y=(0.5,1.5))
        .label(x='Stimulus phase (Normalized)',y='Cycle duration (Normalized)')
        .layout(size=(3,2))
    ).plot()

    ax = p._figure.axes[0]
    ax.axhline(1, ls='--', color='gray', lw=0.5)
    ax.axvline(1, ls='--', color='gray', lw=0.5)
    ax.axvline(np.mean(df['duty_cycle']), ls='--', color='m', lw=0.5)

    p.save('reset_curves_inout.pdf')

