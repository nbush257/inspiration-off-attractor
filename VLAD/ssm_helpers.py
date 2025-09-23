from sklearn.svm import SVR
import ssm
import numpy as np
from ssm.model_selection import cross_val_scores
import numpy.random
import matplotlib.pyplot as plt
import pickle
import logging
from cibrrig.analysis import population
from joblib import Parallel, delayed, effective_n_jobs

try:
    plt.style.use("../VLAD.mplstyle")
except:
    print("Not using VLAD styling")



logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
WITH_NOISE = False


def plot_results(results, save_dir):
    # Plotting for all the results of a given binsize
    rslds = results["rslds"]
    tbins = results["tbins"]
    q = results["q"]
    binsize = results["binsize"]
    elbos = results["elbos"]

    f = plt.figure()
    gs = f.add_gridspec(nrows=3, ncols=2)
    ax_z_obs = f.add_subplot(gs[0, 0])
    ax_y_obs = f.add_subplot(gs[1, 0], sharex=ax_z_obs)
    ax_z_sim = f.add_subplot(gs[0, 1])
    ax_y_sim = f.add_subplot(gs[1, 1], sharex=ax_z_sim, sharey=ax_y_obs)
    ax_elbo = f.add_subplot(gs[2, :])

    y = q.mean_continuous_states[0]
    z = np.argmax(q.mean_discrete_states[0], 1)
    ax_z_obs.plot(tbins, z)
    ax_y_obs.plot(tbins, y)
    ax_z_obs.set_xlim(tbins[100], tbins[100] + 3)
    ax_y_obs.set_xlim(tbins[100], tbins[100] + 3)

    N = int(100 / binsize)
    try:
        zhat, yhat, emhat = rslds.sample(N, with_noise=WITH_NOISE)
        t = np.linspace(0, N * binsize, N)
        ax_z_sim.plot(t, zhat)
        ax_y_sim.plot(t, yhat)
        ax_y_sim.set_xlim(60, 63)
        ax_z_sim.set_xlim(60, 63)
    except ValueError as e:
        _log.warning(e)

    ax_elbo.plot(elbos[1:])
    ax_elbo.set_xlabel("Iterations")
    ax_elbo.set_ylabel("ELBO")

    plt.tight_layout()
    fig_fn = save_dir.joinpath(f"K{rslds.K}_D{rslds.D}_binsize{binsize:0.03f}.png")
    plt.savefig(fig_fn)
    plt.close("all")


def compute_LDS_poisson(raster, K=2, D_latent=2, num_iters=50,inputs=None):
    D_obs = raster.shape[0]
    M=0 if inputs is None else inputs.shape[1]
    rslds = ssm.SLDS(
        D_obs,
        K,
        D_latent,
        M=M,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="poisson_orthog",
        emission_kwargs=dict(link="softplus"),
        single_subspace=True,
    )

    X = raster.T
    rslds.initialize(X,inputs=inputs)

    elbos, q = rslds.fit(
        X,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=num_iters,
        initialize=False,
        alpha=0,
        inputs=inputs
    )
    return (elbos, q, rslds)


def compute_LDS_bernoullli(raster, K=2, D_latent=2, num_iters=50,inputs=None):
    D_obs = raster.shape[0]
    M=0 if inputs is None else inputs.shape[1]
    rslds = ssm.SLDS(
        D_obs,
        K,
        D_latent,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="bernoulli_orthog",
        emission_kwargs=dict(link="logit"),
        single_subspace=True,
    )

    X = raster.T
    X = X.astype("bool").astype("int")
    rslds.initialize(X,inputs=inputs)

    elbos, q = rslds.fit(
        X,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=num_iters,
        initialize=False,
        alpha=0,
        inputs=inputs
    )
    return (elbos, q, rslds)


def _get_stim_vector(stim_intervals,tbins,stim_amplitudes=None):
    stim_vector = np.zeros_like(tbins)
    if stim_amplitudes is None:
        stim_amplitudes = np.ones(stim_intervals.shape[0])   
    elif isinstance(stim_amplitudes,(int,float)):
        stim_amplitudes = np.ones(stim_intervals.shape[0])*stim_amplitudes
    else:
        pass
    for (_t0,_tf),amp in zip(stim_intervals,stim_amplitudes):
        s0,sf = np.searchsorted(tbins,[_t0,_tf])
        stim_vector[s0:sf]=amp
    return(stim_vector[:,np.newaxis])


def compute_LDS(
    spike_times,
    spike_clusters,
    t0,
    tf,
    binsize,
    K,
    D,
    save_dir=None,
    num_iters=50,
    metadata=None,
    overwrite=False,
    stim_intervals=None,
    stim_amplitudes=None
):
    rez_fn = save_dir.joinpath(f"rslds_K{K}_D{D}_binsize{binsize:0.03f}.pkl")
    if rez_fn.exists() and not overwrite:
        _log.warning(f"{rez_fn} exists. Skipping")
        return None
    _log.info(f"Creating raster with binsize {binsize}")
    raster, tbins, cbins = population.rasterize(
        spike_times, spike_clusters, binsize=binsize
    )
    _raster, _tbins = population._subset_raster(raster, tbins, t0, tf)
    _raster = _raster.astype("int")

    if stim_intervals is not None:
        stim_vector = _get_stim_vector(stim_intervals,_tbins,stim_amplitudes) 
    else:
        stim_vector=None

    if binsize == 0.001:
        elbos, q, rslds = compute_LDS_bernoullli(
            _raster, K=K, D_latent=D, num_iters=num_iters,inputs=stim_vector
        )
    else:
        elbos, q, rslds = compute_LDS_poisson(
            _raster, K=K, D_latent=D, num_iters=num_iters,inputs=stim_vector
        )

    rez = {}
    rez["elbos"] = elbos
    rez["q"] = q
    rez["rslds"] = rslds
    rez["tbins"] = _tbins
    rez["cbins"] = cbins
    rez["K"] = K
    rez["D"] = D
    rez["binsize"] = binsize
    rez["metadata"] = metadata
    if save_dir is None:
        return rez
    else:
        _log.info(f"Saving to {rez_fn}")
        write_results(rez, rez_fn)
        plot_results(rez, save_dir)
        return None


def plot_most_likely_dynamics(
    model,
    xlim=(-4, 4),
    ylim=(-3, 3),
    nxpts=20,
    nypts=20,
    alpha=0.8,
    ax=None,
    figsize=(3, 3),
    plot_inputs=False,
    input_strength=1,
    colors=["tab:red", "tab:blue", "silver"],
    scale=1e3
):
    K = model.K
    assert model.D == 2
    if plot_inputs:
        assert model.M == 1
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b, v) in enumerate(zip(model.dynamics.As, model.dynamics.bs, model.dynamics.Vs)):
        if not plot_inputs:
            v = np.zeros_like(b)
        else:
            ival = np.ones(model.M) * input_strength
            v = v.dot(ival)
        dxydt_m = xy.dot(A.T) + b + v - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(
                xy[zk, 0],
                xy[zk, 1],
                dxydt_m[zk, 0],
                dxydt_m[zk, 1],
                color=colors[k % len(colors)],
                alpha=alpha,
                scale=scale
            )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    plt.tight_layout()

    return ax


def write_results(results, fn):
    with open(fn, "wb") as fid:
        pickle.dump(results, fid)
