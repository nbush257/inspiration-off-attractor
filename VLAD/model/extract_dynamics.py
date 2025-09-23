import sys
sys.path.append("../")
from analyze_single_rec import Rec
from utils import one, EIDS_REGEN
import pickle


class Rec(Rec):
    def extract(self,suffix=''):
        self.load_rslds(suffix=suffix)

        As = self.rslds.dynamics.As
        bs = self.rslds.dynamics.bs
        Vs = self.rslds.dynamics.Vs
        Rs = self.rslds.transitions.Rs
        r = self.rslds.transitions.r
        phase_k = self.assign_phase_to_k()
        fast_exp_dynamics = self.get_fast_exp_direction()
        slow_exp_dynamics = self.get_slow_exp_direction()

        d = {
            "As": As,
            "bs": bs,
            "Vs": Vs,
            "Rs": Rs,
            "r": r,
            'exp_fast': fast_exp_dynamics,
            'exp_slow': slow_exp_dynamics,
            "phase_k": phase_k,
            "subject": self.subject,
            "number": self.sequence,
            "eid": self.eid,
            "genotype": self.genotype
        }
        return d

D = {}
for eid in EIDS_REGEN:
    rec = Rec(one, eid)
    d = rec.extract(suffix='_norm')
    D[eid] = d

with open("dynamics.pkl", "wb") as f:
    pickle.dump(D, f)
