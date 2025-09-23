import sys

sys.path.append("../")
sys.path.append("VLAD/")
from utils import (
    SUBJECTS,
    GENOTYPES,
    Rec,
    one,
    EIDS_HOLDS,
    EIDS_PHASIC,
    EIDS_RESET,
    EIDS_PHYSIOL,
)
import logging
from cibrrig.plot import plot_reset_curve
from cibrrig.utils.utils import make_pre_post_trial
import pandas as pd
import numpy as np
import scipy.stats

np.random.seed(0)
genotypes = set(GENOTYPES.values())

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

eids = EIDS_PHYSIOL
eids = eids[::-1]
print(f"# Sessions: {len(eids)}")



class Rec(Rec):
    def get_reset_curves(self, duration=0.05, df=None):
        log_df = self.log.to_df()
        laser_df = self.laser.to_df()

        # ----------------------- #
        # short pulse reset curves
        # ----------------------- #
        t0, tf = (
            log_df.query('label=="opto_pulse" & duration==@duration')["start_time"]
            .iloc[[0, -1]]
            .values
        )
        t0 -= 0.5
        tf += 0.5
        pulses = laser_df.query("intervals_0>@t0 & intervals_1<@tf")[
            "intervals_0"
        ].values
        x_stim, y_stim, x_control, y_control = plot_reset_curve(
            self.breaths,
            pulses,
            norm=True,
            annotate=False,
            plot_tgl=False,
            wavelength=rec.wavelength,
        )

        stim_reset = pd.DataFrame()
        control_reset = pd.DataFrame()
        stim_reset["Stim Phase"] = x_stim
        stim_reset["Cycle Duration"] = y_stim
        stim_reset["comparison"] = "stim"
        control_reset["Stim Phase"] = x_control
        control_reset["Cycle Duration"] = y_control
        control_reset["comparison"] = "control"
        _df = pd.concat([stim_reset, control_reset])

        _df["eid"] = self.eid
        _df["subject"] = self.subject
        _df["sequence"] = self.sequence
        _df["genotype"] = self.genotype
        _df = _df.reset_index().rename({"index": "Stim #"}, axis=0)

        # Get duty cycle:
        duty_cycle = 1 - self.breaths["postBI"] / self.breaths["IBI"]
        idx = np.logical_and(self.breaths.times > t0 - 100, self.breaths.times < t0)
        duty_cycle = duty_cycle[idx].mean()

        # get peak dia duty cycle
        pk_duty_cycle = (self.breaths.pk_time - self.breaths.on_sec) / self.breaths.IBI
        idx = np.logical_and(self.breaths.times > t0 - 100, self.breaths.times < t0)
        pk_duty_cycle = pk_duty_cycle[idx].mean()

        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])
        return (df, duty_cycle, pk_duty_cycle)

    def get_rebound_latency(rec,df=None):
        intervals,stims = rec.get_stims('hold')
        latencies = []
        amps = []
        durations = []
        if df is None:
            df = pd.DataFrame()
        for ii,(_,offset) in enumerate(intervals):
            idx = np.where(rec.breaths.times>offset)[0][0]
            # Latency
            latency = rec.breaths.times[idx] - offset
            latencies.append(latency)
            #  Amplitude
            amps.append(rec.breaths.amp[idx])
            # Duration
            durations.append(rec.breaths.duration_sec[idx])

        

        _df = pd.DataFrame()
        _df["latency"] = latencies
        _df["eid"] = rec.eid
        _df["subject"] = rec.subject
        _df["sequence"] = rec.sequence
        _df["genotype"] = rec.genotype
        _df['amp'] = amps
        _df['duration'] = durations

        df = pd.concat([df, _df]).reset_index(drop=True)
        return(df)

    def get_stim_resp_rate(self, query, conditions=None, df=None):
        """
        Get the respiratory rate for a given query of the log dataframe

        Args:
            query (_type_): _description_
            conditions (_type_, optional): _description_. Defaults to None.
            df (_type_, optional): _description_. Defaults to None.
        """
        log_df = self.log.to_df()
        stims = log_df.query(query)
        if stims.shape[0] == 0:
            return df
        intervals = stims[["start_time", "end_time"]].values
        conditions = conditions or []
        conditions.append("duration")  # Add duration to be able to compute rates
        _df = make_pre_post_trial(
            self.breaths,
            intervals,
            conditions=stims[conditions],
            vars=["inst_freq"],
            agg_func="count",
        )

        # Rename test to stim
        _df["comparison"] = _df["comparison"].map(
            {"test": "stim", "control": "control"}
        )

        # Convert to respiratory rate
        _df["resp_rate"] = _df["inst_freq_count"] / _df["duration"]

        _df["subject"] = self.subject
        _df["sequence"] = self.sequence
        _df["eid"] = self.eid
        _df["genotype"] = self.genotype

        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])
            df.reset_index(drop=True, inplace=True)

        return df

    def get_phasic_amps(self,df=None):
        for phase in ['insp','exp']:
            stims = self.get_stims(phase)[1]
            intervals = stims[["start_time", "end_time"]].values
            _df = make_pre_post_trial(
                self.breaths,
                intervals,
                conditions=None,
                vars=['amp'],
            )
            _df.fillna(0, inplace=True)
            _df["subject"] = self.subject
            _df["sequence"] = self.sequence
            _df["eid"] = self.eid
            _df["genotype"] = self.genotype
            _df["phase"] = phase


            if df is None:
                df = _df
            else:
                df = pd.concat([df, _df])
                df.reset_index(drop=True, inplace=True)

        return df

    def get_stim_heartrate(self, query, conditions=None, df=None):
        """
        Get the respiratory rate for a given query of the log dataframe

        Args:
            query (_type_): _description_
            conditions (_type_, optional): _description_. Defaults to None.
            df (_type_, optional): _description_. Defaults to None.
        """
        log_df = self.log.to_df()
        stims = log_df.query(query)
        if stims.shape[0] == 0:
            return df
        intervals = stims[["start_time", "end_time"]].values
        conditions = stims[conditions] if conditions is not None else None
        _df = make_pre_post_trial(
            self.physiology,
            intervals,
            conditions=conditions,
            vars=["hr_bpm", "temperature"],
            agg_func="mean",
        )

        # Rename test to stim
        _df["comparison"] = _df["comparison"].map(
            {"test": "stim", "control": "control"}
        )
        _df["subject"] = self.subject
        _df["sequence"] = self.sequence
        _df["eid"] = self.eid
        _df["genotype"] = self.genotype

        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])
            df.reset_index(drop=True, inplace=True)

        return df


# --------------------------------------------------------------- #
# ------------------------ RUN COMPUTATION ---------------------- #
# --------------------------------------------------------------- #

reset_curves = pd.DataFrame()
long_opto = pd.DataFrame()
duty_cycle = {x: [] for x in SUBJECTS.keys()}
pk_duty_cycle = {x: [] for x in SUBJECTS.keys()}
HB_stims = pd.DataFrame()
phasic_opto = pd.DataFrame()
duration_reset = 0.05
aux_physiol = pd.DataFrame()
intercept = pd.DataFrame()
phasic_amps = pd.DataFrame()
rebound_latencies = pd.DataFrame()

# Loop all mice
for eid in eids:

    rec = Rec(one, eid, load_spikes=False)
    subject = rec.subject
    sequence = rec.sequence

    # Get reset curves
    if eid in EIDS_RESET:
        reset_curves, _duty_cycle, _pk_duty_cycle = rec.get_reset_curves(
            duration_reset, reset_curves
        )
        x, y = (
            reset_curves.query('comparison=="stim" & eid==@eid')
            .sort_values("Stim Phase")[["Stim Phase", "Cycle Duration"]]
            .values.T
        )
        
        
        # Get the intercept of the resetcurve with 1
        y2 = scipy.signal.savgol_filter(y, 21, 2)
        # Find where y2 crosses 1
        idx = np.where(np.diff(np.sign(y2 - 1)))[0]
        if len(idx) > 0:
            _intercept = x[idx[0]]
        else:
            _intercept = np.nan

        intercept = pd.concat(
            [
                intercept,
                pd.DataFrame(
                    {
                        "intercept": [_intercept],
                        "eid": [eid],
                        "subject": [subject],
                        "sequence": [sequence],
                        "genotype": [rec.genotype],
                        "duty_cycle": [_duty_cycle],
                        "peak_duty_cycle": [_pk_duty_cycle],
                        "duration_pulse": [duration_reset],
                    }
                ),
            ]
        )

        duty_cycle[rec.genotype].append(_duty_cycle)
        pk_duty_cycle[rec.genotype].append(_pk_duty_cycle)

    if eid in EIDS_HOLDS:
        # Get long pulses
        long_opto = rec.get_stim_resp_rate(
            'duration==2.0 & label=="opto_pulse"', df=long_opto
        )
        # Get Hering Breuer
        HB_stims = rec.get_stim_resp_rate('label=="hering_breuer"', df=HB_stims)

        # Get rebound latency
        rebound_latencies = rec.get_rebound_latency(df=rebound_latencies)

        # Get heart rate and temp
        if ("hr_bpm" in rec.physiology.keys()) and (
            "temperature" in rec.physiology.keys()
        ):
            aux_physiol = rec.get_stim_heartrate(
                'duration ==2.0 & label=="opto_pulse"', df=aux_physiol
            )
        else:
            _log.warning(f"No heart rate found")

    if eid in EIDS_PHASIC:
        # Get phasic stims
        phasic_opto = rec.get_stim_resp_rate(
            'label=="opto_phasic" & mode=="hold"', df=phasic_opto, conditions=["phase"]
        )
        phasic_amps = rec.get_phasic_amps(df=phasic_amps)

# Clean up and bin reset curves
reset_curves.drop(columns=["index"], inplace=True)
reset_curves.reset_index(drop=True, inplace=True)
bb = np.arange(0, 2, 0.05)
reset_curves["Stim Phase Bins"] = bb[np.digitize(reset_curves["Stim Phase"], bb)] - 0.05
reset_curves["duration_pulse"] = duration_reset

# Clean up reset_curve_intercept
intercept.reset_index(drop=True, inplace=True)
intercept.to_csv("reset_intercept.csv", index=False)

# Cast to float
long_opto["resp_rate"] = long_opto["resp_rate"].astype("f")
phasic_opto["resp_rate"] = phasic_opto["resp_rate"].astype("f")
HB_stims["resp_rate"] = HB_stims["resp_rate"].astype("f")
aux_physiol["hr_bpm_mean"] = aux_physiol["hr_bpm_mean"].astype("f")
aux_physiol["temperature_mean"] = aux_physiol["temperature_mean"].astype("f")

# Group by genotype and subject
long_opto_pivot = (
    long_opto.groupby(["genotype", "subject", "sequence", "eid", "comparison"])[
        "resp_rate"
    ]
    .mean()
    .reset_index()
)
phasic_opto_pivot = (
    phasic_opto.groupby(
        ["genotype", "subject", "sequence", "eid", "phase", "comparison"]
    )["resp_rate"]
    .mean()
    .reset_index()
)
HB_pivot = (
    HB_stims.groupby(
        ["genotype", "subject", "sequence", "duration", "eid", "comparison"]
    )["resp_rate"]
    .mean()
    .reset_index()
)
aux_physiol_pivot = (
    aux_physiol.groupby(["genotype", "subject", "sequence", "eid", "comparison"])[
        ["hr_bpm_mean", "temperature_mean"]
    ]
    .mean()
    .reset_index()
)
phasic_amps['comparison'] = phasic_amps['comparison'].map({'test':'stim','control':'control'})
phasic_amps_pivot = (
    phasic_amps.groupby(["genotype", "subject", "sequence", "eid","phase","comparison"])[
        "amp_mean"]
    .mean()
    .reset_index()
)

# Save results to csv
long_opto.to_csv('2s_opto_stims_raw.csv',index=False)
long_opto_pivot.to_csv("2s_opto_stims.csv", index=False)
phasic_opto_pivot.to_csv("phasic_opto_stims.csv", index=False)
phasic_opto.to_csv('phasic_opto_stims_raw.csv')
HB_pivot.to_csv("HB_stims.csv", index=False)
reset_curves.to_csv("reset_curves.csv", index=False)
phasic_amps_pivot.to_csv("phasic_amps.csv", index=False)

# Delta heartrate per trial
hr_normed = aux_physiol.pivot(
    index=["genotype", "eid", "trial"], values=["hr_bpm_mean"], columns="comparison"
).reset_index()
hr_normed["delta_hr_pct"] = (
    hr_normed["hr_bpm_mean"]["stim"] / hr_normed["hr_bpm_mean"]["control"] * 100
)
hr_normed["delta_hr_bpm"] = (
    hr_normed["hr_bpm_mean"]["stim"] - hr_normed["hr_bpm_mean"]["control"]
)
hr_normed.drop(columns=["hr_bpm_mean"], inplace=True, level=0)
hr_normed.to_csv("heartrate_delta.csv", index=False)

# divide resp_rate for all conditions by resp rate for control
phasic_opto_normed = phasic_opto_pivot.pivot(
    index=["genotype", "eid", "phase"], values="resp_rate", columns="comparison"
).reset_index()
phasic_opto_normed["resp_rate_norm"] = (
    phasic_opto_normed["stim"] / phasic_opto_normed["control"] * 100
)
phasic_opto_normed.to_csv("phasic_opto_normed.csv")

# Save duty cycle
duty_cycle = pd.DataFrame(
    {k: [np.mean(v)] for k, v in duty_cycle.items()}, index=["duty_cycle"]
)
pk_duty_cycle = pd.DataFrame(
    {k: [np.mean(v)] for k, v in pk_duty_cycle.items()}, index=["peak_duty_cycle"]
)
df_duty_cycle = pd.concat([duty_cycle, pk_duty_cycle]).T.reset_index(names="genotype")
df_duty_cycle.to_csv("duty_cycle.csv")

rebound_latencies.to_csv("rebound_latencies.csv", index=False)

# --------------------------------------------------------------- #
# ------------------------ STATS -------------------------------- #
# --------------------------------------------------------------- #

stats = []
genotype = []
label = []

for gg in genotypes:
    print(gg)

    # Get respiratory rate changes for 2s pulses
    ctrl = long_opto_pivot.query('genotype==@gg and comparison=="control"')["resp_rate"]
    test = long_opto_pivot.query('genotype==@gg and comparison=="stim"')["resp_rate"]
    t, p = scipy.stats.ttest_rel(ctrl, test)
    n = ctrl.size
    stats.append([t, p, n])
    label.append("2s")
    genotype.append(gg)

    # Get heart rate changes for 2s pulses
    ctrl = aux_physiol_pivot.query('genotype==@gg and comparison=="control"')[
        "hr_bpm_mean"
    ]
    test = aux_physiol_pivot.query('genotype==@gg and comparison=="stim"')[
        "hr_bpm_mean"
    ]
    t, p = scipy.stats.ttest_rel(ctrl, test)
    n = ctrl.size
    stats.append([t, p, n])
    label.append("hr_bpm")
    genotype.append(gg)

    # Get temperature changes for 2s pulses
    ctrl = aux_physiol_pivot.query('genotype==@gg and comparison=="control"')[
        "temperature_mean"
    ]
    test = aux_physiol_pivot.query('genotype==@gg and comparison=="stim"')[
        "temperature_mean"
    ]
    t, p = scipy.stats.ttest_rel(ctrl, test)
    n = ctrl.size
    stats.append([t, p, n])
    label.append("temperature")
    genotype.append(gg)

    for pp in ["insp", "exp"]:
        ctrl = phasic_opto_pivot.query(
            'genotype==@gg and comparison=="control" and phase==@pp'
        )["resp_rate"]
        test = phasic_opto_pivot.query(
            'genotype==@gg and comparison=="stim" and phase==@pp'
        )["resp_rate"]
        t, p = scipy.stats.ttest_rel(ctrl, test)
        n = ctrl.size
        stats.append([t, p, n])
        genotype.append(gg)
        label.append(f"phasic_{pp}")

        # Phasic normed
        samp = phasic_opto_normed.query("genotype==@gg and phase==@pp")[
            "resp_rate_norm"
        ]
        t, p = scipy.stats.ttest_1samp(samp, 100)
        n = samp.size
        label.append(f"phasic_{pp}_normed")
        genotype.append(gg)
        stats.append([t, p, n])


        # Phasic amplitudes
        ctrl = phasic_amps_pivot.query(
            'genotype==@gg and comparison=="control" and phase==@pp'
        )["amp_mean"]
        test = phasic_amps_pivot.query(
            'genotype==@gg and comparison=="stim" and phase==@pp'
        )["amp_mean"]
        t, p = scipy.stats.ttest_rel(ctrl, test)
        n = ctrl.size
        stats.append([t, p, n])
        genotype.append(gg)
        label.append(f"phasic_amps_{pp}")


    for dd in [2, 5]:
        ctrl = HB_pivot.query(
            'genotype==@gg and comparison=="control" and duration==@dd'
        )["resp_rate"]
        test = HB_pivot.query('genotype==@gg and comparison=="stim" and duration==@dd')[
            "resp_rate"
        ]
        t, p = scipy.stats.ttest_rel(ctrl, test)
        n = ctrl.size
        stats.append([t, p, n])
        genotype.append(gg)
        label.append(f"hering_breuer_{dd:0.0f}s")


idx = pd.MultiIndex.from_tuples(list(zip(genotype, label)))
stats_df = pd.DataFrame(stats, index=idx, columns=["t", "p_value", "n"])
stats_df["reject_null"] = stats_df["p_value"] < 0.05
stats_df.to_csv("physiology_pvals.csv")
print(stats_df)

# Hering breuer ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Compute a three-way ANOVA where the factors are genotype, duration and comparison
# and the dependent variable is resp_rate
formula = "resp_rate ~ C(genotype) * C(duration) * C(comparison)"
model = ols(formula, data=HB_pivot).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
pd.DataFrame(anova_table).to_csv("HB_ANOVA.csv")


