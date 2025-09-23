# model
Intermediate results and plotting for rSLDS models. This is the folder used in publication.

Many `.py` files are associated with a `.slurm` file for running on the HPC.


## Data 
There are 3 models fit for each recording:
1. rslds_\<mouse>_\<gX>
    - Model fit where stimulus is continuous value scaled to mW as calibrated from acquisition hardware
1. rslds_\<mouse>_\<gX>_norm
    - Model fit where stimulus is binary (stim_on=1, stim_off =0)
1. rslds_\<mouse>_\<gX>_nostim
    - Model fit during baseline period only

`norm` and `no_stim` are used in publication. Since there was only ever one amplitude given, the scaling of the mW calibrated experiments made interpretation difficult when the rSLDS model tried to extrapolate.

## Code
### SLURM Jobs

`rslds_pipeline_normed.slurm` - Runs fitting, posterior inference, and single recording analysis in a job array for every recording


### Python scripts

`fit_model.py` - Fits the rSLDS models. Can fit all three types of data described above

`analyze_single_rec.py` - Main code for analysis. Creates classes that read in the rslds model and can apply arbitrary stimulus. Generates plots for each recording. Generates intermediate data used in summary plotting.

`analyze_rec_no_stim_fit.py` - Generateplots for each recording and intermediate data that is used in summary plotting.

`extract_dynamics.py` - Generates the 
`dynamics.pkl` file which contains the `A,b,V` matrices for analysis of alignemtn of stimulus vector fields. This was created to avoid overhead of loading in all the recording data to do analyses of the dynamics and to create a self-contained dataset of the dynamics.

`infer_posterior.py` - Takes the dynamics model (fit to a subset of each recording) and applies them to the unfitted parts of the recording. This is primarily useful so as to have the entire recording mapped into the latent space for fitting of the mapping between the latent and the diaphragm.

`plot_stim_field_alignments.py` - Plots analysis of stimulus fields with the underlying latent dynamics fields. Relies on the `dynamics.pkl` file.

`plot_summary_learned_stims.py`:warning: Unused in favor of `nostim` analyses

`plot_summary_nostim_figures.py` - Compute intermediate data and create plots that analyse the dynamics with arbitrary simulated stimulations (e.g., uniform vector fields) across all recordings.

`run_high_res_amplitude_sweeps.py` - :warning: Unused - Runs amplitude sweeps at many amplitude values. 

