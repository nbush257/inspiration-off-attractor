'''
Script to extract respiratory coherence for all the sessions in this dataset
'''

import sys
sys.path.append('../')
from utils import one,EIDS_NEURAL,CACHE_DIR,One
import logging
from cibrrig.postprocess import extract_resp_modulation
import matplotlib
matplotlib.use('Agg')
logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
OVERWRITE=False


for eid in EIDS_NEURAL:
    sess_path = one.eid2path(eid)
    clusters = one.load_object(eid,'clusters')
    run_probe = True

    if 'respMod' in clusters.keys():
        _log.info('Respiratory modulation already computed')
        if OVERWRITE:
            _log.warning('OVERWRITE is true, recomputing')
        else:
            run_probe=False
    if run_probe:
        _log.info('Computing respiratory modulation')
        extract_resp_modulation.run_session(sess_path,t0=300,tf=600)

# Refresh the cache
One.setup(cache_dir=CACHE_DIR,silent=True)