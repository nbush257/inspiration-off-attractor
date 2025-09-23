'''
Label clusters as out of brain
This has to be done on the RSS active because I did not move the raw data to Sasquatch
Then, need to rsync the new data over to Sasquatch

Doing here because this was not implemented in the spikesorting pipeline when data were acquired

'''
from cibrrig.sorting.spikeinterface_ks4 import remove_and_interpolate
import pandas as pd
import sys
sys.path.append('../')
from one.api import One
import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np
import logging
import click
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

if sys.platform == 'linux':
    cache_dir = '/data/rss/helens/ramirez_j/ramirezlab/alf_data_repo'
elif sys.platform == 'win32':
    cache_dir = r"X:/alf_data_repo"

one = One(cache_dir=cache_dir)
sessions = pd.read_csv('../sessions_to_use.csv').query('has_neural')

subjects = sessions['subject'].unique()
eids = one.search(subject = subjects)
eid = eids[0]

def get_inBrain_labels(eid,overwrite=False):
    '''
    Get the inBrain labels for a given session
    Saves the labels to the alf folder
    Also saves to the raw_ephys_data folder along with a QC image

    Args-
        eid: str, session id
    Returns-
        labels: np.array, inBrain labels for each cluster
    '''
    _log.info(f'Getting inBrain labels for {one.eid2ref(eid)}')
    session_path = one.eid2path(eid)
    alf_probe_path = session_path.joinpath('alf/probe00')
    save_fn = alf_probe_path.joinpath('_spikeinterface_clusters.inBrain.npy')
    clusters = one.load_object(eid,'clusters',revision='')

    if not overwrite:
        if 'inBrain' in clusters.keys():
            _log.info('Label already incorporated in clusters')
            return None

        if save_fn.exists():
            _log.info('Labels already saved')
            if 'inBrain' not in clusters.keys():
                _log.warning('Labels not incorporated in clusters')
            return None


    ap_file = one.list_datasets(eid,filename='*ephysData*ap.bin*')[0]
    raw_probe_path = session_path.joinpath(ap_file).parent
    SR = si.read_spikeglx(raw_probe_path,stream_id='imec0.ap')
    _,chan_labels = remove_and_interpolate(SR)

    labels = chan_labels[clusters['channels']]
    labels = labels!='out'
    np.save(save_fn,labels)
    _log.info(f'Saved inBrain labels to {save_fn}')


@click.command()
@click.option('--overwrite',is_flag=True,help='Overwrite existing labels')
def main(overwrite):
    for eid in eids:
        get_inBrain_labels(eid,overwrite=overwrite)

if __name__ == '__main__':
    main()





