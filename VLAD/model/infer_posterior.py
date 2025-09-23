from cibrrig.analysis.population import rasterize
import sys
sys.path.append("../")
from ssm_helpers import _get_stim_vector
from utils import (
    set_style,
    one,
    EIDS_NEURAL,
)
import numpy as np
import click

from analyze_single_rec import Rec


@click.command()
@click.argument("ii", type=int)
@click.option('--norm','-n',is_flag=True,default=False)
@click.option('--no_stim','-ns',is_flag=True,default=False)
def main(ii,norm,no_stim):
    if norm:
        suffix = "_norm" 
    elif no_stim:
        suffix = '_nostim'
    else:
        suffix = ''
    rec = Rec(one,EIDS_NEURAL[ii])
    rec.load_rslds(suffix=suffix)
    X,tbins = rec.project_full_recording()
    fn = rec.model_path.joinpath('projection.values.npy')
    np.save(fn,X)
    fn = rec.model_path.joinpath('projection.times.npy')
    np.save(fn,tbins)

if __name__ == "__main__":
    main()







