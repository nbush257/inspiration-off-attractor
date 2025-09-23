"""
Utility file to store global variables and helper functions
Sets the cache directory for the ONE API and defines the subject IDs for different genotypes
Also defines the color maps for the different phases and genotypes
Contains the Rec class to load and organize data from a single recording
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from cibrrig.preprocess.physiology import compute_dia_phase,compute_avg_hr
from cibrrig.plot import laser_colors
from one.api import One
from itertools import chain
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib
import seaborn.objects as so


logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

if sys.platform == "win32":
    CACHE_DIR = Path(r'C:\Users\nbush\local_working\VLAD\alf_data_repo')
else:
    CACHE_DIR = Path("/data/hps/assoc/private/medullary/data/alf_data_repo")
assert CACHE_DIR.exists(), f"{CACHE_DIR} Does not exist"
assert CACHE_DIR.joinpath('datasets.pqt').exists(), f"{CACHE_DIR} is not an initilaized ONE repo"
one = One(CACHE_DIR)

QC_QUERY = "bitwise_fail==0 & presence_ratio>0.8"
VLAD_ROOT = Path(__file__).parent
sessions_to_include = pd.read_csv(VLAD_ROOT.joinpath("sessions_to_use.csv"))
def set_style():
    plt.style.use(VLAD_ROOT.joinpath("VLAD.mplstyle"))
    so.Plot.config.theme.update(matplotlib.rcParams)


# ======= DEFINE COLORS AND LABELS ======= # 
PHASE_MAP = {
    "insp": "tab:red",
    "exp": "tab:blue",
    "tonic": "#5A5A5A",
    "qc_fail": "yellow",
}

HB_MAP = {
    2: "plum",
    5: "mediumvioletred",
}

GENOTYPE_LABELS = dict(
    vgatai32="VRC$^{GABA}$",
    vglut2ai32="VRC$^{VGLUT2}$",
    vgatcre_ntschrmine="NTS$^{GABA}$",
)

GENOTYPE_COLORS = dict(
    vglut2ai32="C0",
    vgatai32="C1",
    vgatcre_ntschrmine="C2",
)

GENOTYPES = sessions_to_include.groupby("subject").first()["genotype"].to_dict()
SUBJECTS = {}
for gg in sessions_to_include["genotype"].unique():
    SUBJECTS[gg] = list(sessions_to_include.query("genotype==@gg")["subject"].unique())


# ======================================== #
# Map eid to each row in sessions_to_include
for ii, rr in sessions_to_include.iterrows():
    eid = one.search(subject=rr["subject"], number=rr["sequence"])
    assert(len(eid)==1), f"Found {len(eid)} eids for {rr['subject']}_{rr['sequence']}"
    sessions_to_include.loc[ii, "eid"] = str(eid[0])


# ======== HELPER FUNCTIONS ======== #
def sig2star(p):
    """
    Convert a p-value to a star representation
    Args:
        p (float): p-value
    Returns:
        str: star representation of the p-value
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def get_prefix(one, eid):
    """
    Get a user readable ID for a given experiment ID

    Args:
        one (one.api.One): instance of the ONE API
        eid (str): experiment ID
    Returns:
        prefix (str): user readable
    """
    
    ref = one.eid2ref(eid)
    subject = ref["subject"]
    sequence = ref["sequence"]
    genotype = GENOTYPES[subject]
    return f"{subject}_g{sequence}_{genotype}"

def get_eids_from_filter(df, query):
    """
    Get the experiment IDs from a dataframe using a query
    Args:
        df (pd.DataFrame): dataframe with columns 'subject' and 'sequence'
        query (str): query to filter the dataframe
    Returns:
        eid_list (list): list of experiment IDs
    """

    eid_list = []
    for _, rr in sessions_to_include.query(query).iterrows():
        eid = one.search(subject=rr["subject"], number=rr["sequence"])
        eid_list.append(str(eid[0]))
    return eid_list

# ========== GET EIDS FOR DIFFERENT CONDITIONS ========== #
EIDS_NEURAL = get_eids_from_filter(sessions_to_include, "has_neural")
EIDS_HOLDS = get_eids_from_filter(sessions_to_include, "use_holds")
EIDS_RESET = get_eids_from_filter(sessions_to_include, "use_reset")
EIDS_PHASIC = get_eids_from_filter(sessions_to_include, "use_phasic")
EIDS_REGEN = get_eids_from_filter(sessions_to_include, "use_regenerative")
EIDS_REGEN_NOSTIM = get_eids_from_filter(sessions_to_include, "use_regenerative_nostim")
# Get all unique eids from holds, resets, and phasic
EIDS_PHYSIOL = list(set(chain(EIDS_HOLDS, EIDS_RESET, EIDS_PHASIC)))
sessions_to_include.groupby(["genotype"]).sum()[
    ["has_neural", "use_holds", "use_phasic", "use_reset"]
]

# ======== FUNCTIONS TO FILTER AND ANNOTATE NEURAL DATA ======== #
def get_good_spikes(spikes, clusters):
    """
    Convinience function to return only good spikes
    """
    cluster_ids = clusters.metrics.query(QC_QUERY)[
        "cluster_id"
    ].values
    in_brain_clusters = np.where(clusters.inBrain)[0]
    cluster_ids = np.intersect1d(cluster_ids, in_brain_clusters)
    idx = np.isin(spikes.clusters, cluster_ids)
    for k in spikes.keys():
        spikes[k] = spikes[k][idx]
    _log.info(
        f"Total clusters: {clusters.metrics.shape[0]}\nQC pass {cluster_ids.size}\nKeeping only good clusters."
    )
    return (spikes, cluster_ids)


def assign_resp_category(respMod, preferredPhase, respModThresh=0.2):
    """
    Classify the clusters as either tonic, inspiratory or expiratory based on the preferred phase and modulation index
    Args:
        respMod (np.array): modulation index of the clusters
        preferredPhase (np.array): preferred phase of the clusters
        respModThresh (float): threshold to classify clusters as tonic

    Returns:
        category (np.array): array of strings with the category of each cluster (tonic,insp,exp)

    """
    category = np.where(
        respMod < respModThresh, "tonic", np.where(preferredPhase > 0, "insp", "exp")
    )
    return category

# ======== CLASS TO LOAD AND ORGANIZE DATA FROM A SINGLE RECORDING ======== #
class Rec:
    """
    Class to load and organize data from a single recording

    Attributes:
        one (one.api.One): instance of the ONE API
        eid (str): experiment ID
        breaths (alf.io.Bunch): breaths data
        physiology (alf.io.Bunch): physiology data
        diaphragm (alf.io.Bunch): diaphragm data
        laser (alf.io.Bunch): laser data
        log (alf.io.Bunch): log data
        subject (str): subject ID
        sequence (int): sequence number
        genotype (str): genotype of the subject
        has_spikes (bool): whether the recording has spike data
        spikes (alf.io.Bunch): spike data
        clusters (alf.io.Bunch): cluster data
        channels (alf.io.Bunch): channel data
        curate (bool): whether to curate the spike data
        wavelength (int): wavelength of the laser
        prefix (str): prefix for the recording

    Args:
        one (one.api.One): instance of the ONE API
        eid (str): experiment ID
        curate (bool): whether to curate the spike data
        load_spikes (bool): whether to load the spike data
        load_raw_dia (bool): whether to load the raw diaphragm data
    """

    def __init__(self, one, eid, curate=True, load_spikes=True, load_raw_dia=False):
        """
        Initialize the Rec object

        Args:
            one (one.api.One): instance of the ONE API
            eid (str): experiment ID
            curate (bool): whether to curate the spike data
            load_spikes (bool): whether to load the spike data
            load_raw_dia (bool): whether to load the raw diaphragm data
        """
        self.one = one
        self.eid = str(eid)
        print(one.eid2ref(eid))
        self.alf_path = one.eid2path(eid).joinpath('alf')
        self.breaths = self.one.load_object(eid, "breaths",revision='')
        self.physiology = self.one.load_object(eid, "physiology",revision='')
        if load_raw_dia:
            self.diaphragm = self.one.load_object(eid, "diaphragm",revision='')
        self.laser = self.one.load_object(eid, "laser",revision='')
        self.log = self.one.load_object(eid, "log",revision='')
        self.subject = one.eid2ref(eid)["subject"]
        self.sequence = one.eid2ref(eid)["sequence"]
        self.genotype = GENOTYPES[self.subject]
        self.has_spikes = False
        self.curate = curate
        self.wavelength = 473
        self.prefix = get_prefix(one,eid)
        self.get_phi()
        self.recompute_heartrate()

        # Set the laser color based on the wavelength
        if self.genotype == "vgatcre_ntschrmine":
            self.wavelength = 635
        self.laser_color = laser_colors[self.wavelength]

        # Load spike data if available
        if "alf/probe00" in one.list_collections(eid):
            if load_spikes:
                _log.info("Has spikes, loading...")
                self.has_spikes = True
                self.probe_path = one.eid2path(eid).joinpath("alf/probe00")
                clusters = self.one.load_object(self.eid, "clusters",revision='')
                spikes = self.one.load_object(self.eid, "spikes",revision='')
                channels = self.one.load_object(self.eid, "channels",revision='')
                try:
                    clusters.category = assign_resp_category(
                        clusters.respMod, clusters.preferredPhase
                    )
                except Exception as e:
                    print('='*20)
                    print("Could not assign categories")
                    print(e)
                    pass

                if self.curate:
                    spikes, cluster_ids = get_good_spikes(spikes, clusters)
                    self.cluster_ids = cluster_ids
                else:
                    self.cluster_ids = np.arange(clusters.amps.shape[0])
                self.spikes = spikes
                self.clusters = clusters
                self.channels = channels
                self.curate = curate


    def get_phi(self):
        """
        Shortcut to compute respiratory phase from the breaths data and add as attributes of the recording
        """
        phi_t, phi = compute_dia_phase(self.breaths.on_sec, self.breaths.off_sec)
        self.phi_t = phi_t
        self.phi = phi

    def get_phasic_stims(self, phase, mode="hold", frequency=None):
        """
        Extract the phasic stimulus times
        Args:
            phase (str): 'insp' or 'exp'
            mode (str): ['hold','train','pulse']
            frequency (int,optional): frequency of the train stimulations. Required if mode is 'train'. Default is None
        Returns:
            intervals: an n x 2 array of the phasic stim replicates starts and stops times (in seconds)
            stim_periods: a pandas dataframe of the individual pulse times across all replicates
        """
        assert phase in ["insp", "exp"], 'Mode must be ["insp","exp"]'
        assert mode in [
            "hold",
            "train",
            "pulse",
        ], 'Mode must be ["hold","train","pulse"]'

        log = self.log.to_df()
        if frequency is not None:
            stim_periods = log.query(
                'label=="opto_phasic" and phase==@phase and mode==@mode and frequency==@frequency'
            )
        else:
            stim_periods = log.query(
                'label=="opto_phasic" and phase==@phase and mode==@mode'
            )

        if mode == "train" and frequency is None:
            _log.warning(
                "Looking for train stimulations without passing a frequency. Returning all frequencies"
            )

        _log.info(
            f"Found {stim_periods.shape[0]} replicates with {phase=}, {mode=}, {frequency=}"
        )
        if stim_periods.shape[0] == 0:
            return np.array([]), stim_periods

        pulse_df = pd.DataFrame()
        for ii, (_, trial) in enumerate(stim_periods.iterrows()):
            t0 = trial["start_time"] - 0.5
            tf = trial["end_time"] + 0.5
            pulses = (
                self.laser.to_df()
                .query("intervals_0>@t0 & intervals_1<@tf")[
                    ["intervals_0", "intervals_1"]
                ]
                .values
            )
            _df = pd.DataFrame(pulses, columns=["start_time", "end_time"])
            _df["replicate"] = ii
            pulse_df = pd.concat([pulse_df, _df])
        pulse_df.reset_index(drop=True, inplace=True)
        intervals = pulse_df[["start_time", "end_time"]].values.astype("float")

        return (intervals, stim_periods)

    def get_pulse_stims(self, duration, pre_vagotomy=True):
        """
        Get the on and off times of single pulses of a given duration (e.g., 10,50ms pulses)
        Args:
            duration (int): duration of the pulse in seconds
            pre_vagotomy (bool): whether to include only the stims before the vagotomy. Default is True. If False, only the stims after the vagotomy are included

        Returns:
            intervals: an n x 2 array of the phasic stim replicates starts and stops times (in seconds)
            stim_periods: a pandas dataframe of the individual pulse times across all replicates
        """
        log = self.log.to_df()
        vagotomy = log.query('label=="vagotomy"')
        if vagotomy.size > 0:
            if pre_vagotomy:
                log = log.iloc[: vagotomy.index[0] + 1]
            else:
                log = log.iloc[vagotomy.index[0] :]

        stim_periods = log.query('label=="opto_pulse" & duration==@duration')
        if stim_periods.shape[0] == 0:
            _log.warning(f"No stims of duration {1000*duration:0.0f}ms found")
            return (np.array([]), stim_periods)

        t0 = stim_periods["start_time"].iloc[0]
        tf = stim_periods["end_time"].iloc[-1]
        t0 -= 0.5
        tf += 0.5

        intervals = (
            self.laser.to_df()
            .query("intervals_0>@t0 & intervals_1<@tf")[["intervals_0", "intervals_1"]]
            .values.astype("float")
        )

        if intervals.shape[0] != 0:
            _log.info(f"{intervals.shape[0]} stims of {duration*1000:0.0f}ms found")

        return intervals, stim_periods

    def get_HB_stims(self, duration=5):
        """
        Return the intervals of the hering breuer stimulations

        Args:
            duration (int): duration of the HB stimulations in seconds

        Returns:
            intervals: an n x 2 array of the phasic stim replicates starts and stops times (in seconds)
            stim_periods: a pandas dataframe of the individual pulse times across all replicates
        """

        stim_periods = self.log.to_df().query(
            'label=="hering_breuer" and duration==@duration'
        )
        intervals = stim_periods[["start_time", "end_time"]].values.astype("float")
        if intervals.shape[0] == 0:
            _log.warning("No HB stims found")
        else:
            _log.info(f"{intervals.shape[0]} HB stims of {duration*1000:0.0f}ms found")
        return (intervals, stim_periods)

    def get_stims(self, condition, t0=300, tf=500):
        """
        Get the intervals of the stimulations for a given condition

        Wraps the get_phasic_stims, get_pulse_stims, and get_HB_stims methods
        If condition is 'exp' or 'insp', returns the phasic stimulations
        If condition is 'hold', returns the 2s hold stimulations
        If condition is 'hb', returns the 5s HB stimulations
        If condition is 'control', returns the intervals from t0 to tf

        Args:
            condition (str): 'insp','exp','hold','hb','control'
            t0 (int): start time in seconds
            tf (int): end time in seconds

        Returns:
            intervals: an n x 2 array of the phasic stim replicates starts and stops times (in seconds)
            stim_periods: a pandas dataframe of the individual pulse times across all replicates
        """
        if condition == "exp":
            return self.get_phasic_stims("exp")
        if condition == "insp":
            return self.get_phasic_stims("insp")
        if condition == "hold":
            return self.get_pulse_stims(2)
        if condition == "10ms":
            return self.get_pulse_stims(0.01)
        if condition == "50ms":
            return self.get_pulse_stims(0.05)
        if condition == "hb":
            return self.get_HB_stims(2)
        if condition == "control":
            intervals = np.array([[t0, tf]])
            _df = pd.DataFrame(intervals, columns=["start_time", "end_time"])
            return (intervals, _df)

    def recompute_heartrate(self):
        """
        Recompute the heartrate with a more appropriate smoothing
        """
        try:
            heartbeats = one.load_object(self.eid, "heartbeat",revision='')
            t_target = self.physiology.times
            if np.max(heartbeats.times)<20:
                heartbeats.times = heartbeats.times*1e4
                _log.warning(f"Heartbeats times look to be incorrectly converted from samples.\n New heartbeat max:{np.max(heartbeats.times)}. \nPhysiology max:{np.max(self.physiology.times)}")
            resmoothed_hr = compute_avg_hr(heartbeats.times,'1s',t_target=t_target)[1]
            self.physiology['hr_bpm'] = resmoothed_hr
        except Exception as e:
            _log.error(f"Error recomputing heartrate: {e}")
            return

    def get_phasic_unit_ids(self, phase):
        """Get the cluster_ids of all units that are a given phase in descending order of respMod

        Args:
            rec (Recording): Recording object

        Returns:
            list: list of cluster_ids
        """
        idx_phase = np.where(self.clusters.category == phase)[0]
        idx_phase = idx_phase[np.isin(idx_phase, self.cluster_ids)]
        good_respMod = self.clusters.respMod[idx_phase]
        if phase == "tonic":
            order = np.argsort(self.clusters.metrics.loc[idx_phase, "firing_rate"])[
                ::-1
            ].values
        else:
            order = np.argsort(good_respMod)[::-1]
        cluster_ids = idx_phase[order]
        return cluster_ids

