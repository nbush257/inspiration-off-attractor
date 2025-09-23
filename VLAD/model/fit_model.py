import sys
sys.path.append("../")
from utils import (
    set_style,
    Rec,
    one,
    EIDS_NEURAL,
)
import ssm
from ssm_helpers import _get_stim_vector,write_results,plot_results
from cibrrig.analysis import population
import numpy as np
from pathlib import Path
import click
from joblib import Parallel, delayed
np.random.seed(42)

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

    if save_dir is not None:
        rez_fn = save_dir.joinpath(f"rslds_K{K}_D{D}_binsize{binsize:0.03f}.pkl")
        if rez_fn.exists() and not overwrite:
            return None
        print(f"Saving to {rez_fn}")

    raster, tbins, cbins = population.rasterize(
        spike_times, spike_clusters, binsize=binsize
    )
    _raster, _tbins = population._subset_raster(raster, tbins, t0, tf)
    _raster = _raster.astype("int")

    if stim_intervals is not None:
        stim_vector = _get_stim_vector(stim_intervals,_tbins,stim_amplitudes) 
    else:
        stim_vector=None

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
        print(f"Saving to {rez_fn}")
        write_results(rez, rez_fn)
        plot_results(rez, save_dir)
        return None

def compute_LDS_poisson(raster, K=2, D_latent=2, num_iters=50,inputs=None):
    D_obs = raster.shape[0]
    M=0 if inputs is None else inputs.shape[1]
    emissions = 'poisson_orthog'
    if emissions == 'poisson_nn':
        emission_kwargs = dict(
            link='softplus',
            hidden_layer_sizes=(50,50)
        )
        method='bbvi'
        variational_posterior='mf'
        num_iters *=100
    else:
        emission_kwargs={}
        method='laplace_em'
        variational_posterior='structured_meanfield'
    rslds = ssm.SLDS(
        D_obs,
        K,
        D_latent,
        M=M,
        dynamics='diagonal_gaussian',
        transitions="recurrent_only",
        emissions=emissions,
        emission_kwargs=emission_kwargs,
        single_subspace=True,
    )

    X = raster.T
    # T,N = X.shape
    # mask = npr.rand(T,N) < 0.95
    # X_masked = X * mask
    # rslds.initialize(X_masked,masks=mask,inputs=inputs)
    rslds.initialize(X,inputs=inputs)

    elbos, q = rslds.fit(
        X,
        method=method,
        variational_posterior=variational_posterior,
        num_iters=num_iters,
        initialize=False,
        alpha=0,
        inputs=inputs
    )
    return (elbos, q, rslds)

class Rec(Rec):

    def fit_rslds(self,K=2,D=2,binsize=0.01,save_dir=None,overwrite=False,respMod_thresh=0,t0=300,tf=900,num_iters=500,normalize_stim_amp=False):
        spike_times, spike_clusters = self.get_phasic_spikes(respMod_thresh=respMod_thresh)
        meta = dict(
            subject=self.subject,
            sequence=self.sequence,
            genotype=self.genotype,
            eid = self.eid,
        )
        stim_amps = None if normalize_stim_amp else self.laser.amplitudesMilliwatts 
        dat = compute_LDS(
            spike_times,
            spike_clusters,
            t0,
            tf,
            binsize=binsize,
            K=K,
            D=D,
            save_dir=save_dir,
            num_iters=num_iters,
            metadata=meta,
            overwrite=overwrite,
            stim_intervals=self.laser.intervals,
            stim_amplitudes=stim_amps,
        )
        return dat
    
    def get_phasic_spikes(self,respMod_thresh=0):
        good_clus = np.where(self.clusters.respMod>=respMod_thresh)[0]
        idx = np.isin(self.spikes.clusters,good_clus)
        spike_clusters = self.spikes.clusters[idx]
        spike_times = self.spikes.times[idx]
        return spike_times, spike_clusters
    

@click.command()
@click.argument("ii", type=int)
@click.option("--normalize_stim_amp","-n", is_flag=True, default=False)
@click.option("--t0", type=float, default=300)
@click.option("--tf", type=float, default=900)
@click.option("--num_iters",'-i', type=int, default=500)
def main(ii,normalize_stim_amp,t0,tf,num_iters):
    print(f"{normalize_stim_amp=}")
    rec = Rec(one, EIDS_NEURAL[ii])
    if normalize_stim_amp:
        suffix = "_norm" 
    elif tf<600:
        suffix = '_nostim'
    else:
        suffix = ''
    save_dir = Path(f"./rslds_{rec.subject}_g{rec.sequence}{suffix}")
    save_dir.mkdir(exist_ok=True)
    rec.fit_rslds(save_dir=save_dir, overwrite=True,t0=t0,tf=tf,normalize_stim_amp=normalize_stim_amp,num_iters=num_iters)



if __name__ == '__main__':
    main()