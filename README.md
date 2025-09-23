# Inspiration off attractor manuscript 

Running name: VLAD (Ventilatory Latent Attractor Dynamics)

---
### Organization
Code to analyze and make figures for Inspiratory off attractor manuscript in preparation.
> [!NOTE]
> Subfolders of `VLAD` contain code to generate intermediate results, figures, and statistics. See those for details


 - `utils.py` contains code and global variables used across analysis and figure making
 - `VLAD.mplstyle` defines matplotlib rcParams

 - All subfolders have their own README files to describe the packages contained within
 - Relies heavily on the [CIBRRIG](https://github.com/nbush257/cibrrig) repository. CIBRRIG is in active development, however, best efforts have been made to not break functionality. This code works with imports from CIBRRIG version 0.9.0. 
    - Data were preprocessed with older versions of the CIBRRIG pipeline (<=0.6.0), and data processed with CIBRRIG versions newer than that may have incompatabilities. 

### Data inclusion
The `sessions_to_include.csv` file indicates which subject and session number to use for what analyses

We record neural data multiple times from the same mouse, but in different brain locations. We do not double count the physiology of these animals as different n's, so we exclude those from consideration by setting "use_holds","use_reset", and "use_phasic" to True/False. In some cases, some, but not all stimulations were not executed properly during experiment. We exclude stimulations that were not enabled correctly. Specifically, some of the phasic stimulations (e.g., m2024-61) did not have a proper trigger set on the diaphargm signals.

 ### Processing (steps for augmenting the dataset with new recordings):
 > Note: These steps have been largely or entirely subsumed into newer versions of CIBRRIG (>=0.9.0).
 - Move data over from rss
 - Modify `sessions_to_use.csv` to incorporate new data and assign to analyses
 - Run `scripts/convertks2alf.py` to convert neural data to alf structure
 - Run `scripts/extract_resp_coherence.py` to extract respiratory coherence data (on sasquatch)


