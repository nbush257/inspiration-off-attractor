""" 
ALlows the user to determine the caudal end of VII by looking at the depth vs response modulation plot
"""
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from utils import Rec,EIDS_NEURAL,one
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


class Rec(Rec):
    def get_VII(self):
        x = self.clusters.depths
        y = self.clusters.respMod
        y[np.isnan(y)] = 0

        s = pd.Series(index=x,data=y)
        s = s.sort_index()
        s.rolling(50,center=True,min_periods=1).mean().ffill().bfill().plot()
        plt.plot(x,y,'.',ms=1,color='k')
        plt.show()
        VII = plt.ginput(1)[0][0]
        
        self.VII_AP = VII
        plt.close('all')

    def adjust_depths(self,save=True):
        assert hasattr(self,'VII_AP'), 'Run get_VII first'
        self.clusters.depthsVII = self.clusters.depths - self.VII_AP
        if save:
            _log.info(f'Saving VII adjusted depths')
            np.save(self.probe_path.joinpath('_cibrrig_clusters.depthsVII.npy'),self.clusters.depthsVII)

for eid in EIDS_NEURAL:
    r = Rec(one,eid)
    if 'depthsVII' in r.clusters.keys():
        _log.info(f'{eid} already has depthsVII')
    else:
        r.get_VII()
        r.adjust_depths(save=True)
    
_log.info('Done. Dont forget to update the repo')

        