# scripts

Code to incorporate, modify, and standardize raw datasets. Used when new data are included in the analysis dataset. Much of this has been subsumed into newer versions of the CIBRRIG module that supports all NPX processing at SCRI.

`adjust_depths_relative_to_VII.py` - Graphical user input to label the border of VII by looking for a rise in respiratory modulation in the population. Automated methods were not robust enough. Creates cluster feature `depthsVII`

`convertks2alf.py` - convinience script to check if data is in ALF format and convert it if needed. This is a necesarry step in the CIBRRIG pipeline. N.B. As of 2025-09-12 the CIBRRIG pipline has been updated since collectino of this dataset for publication. This step has been incorporated into the CIBRRIG pipeline and does not need to be run on new data. 

`extract_resp_coherence.py` - Create a cluster feature `respMod`,`preferredPhase`, and `maxFiringRatePhase`. N.B. as of 2025-09-12 this has been subsumed into the CIBRRIG pipeline.

`label_out_of_brain_clusters.py` - Use the raw ephys to detect the out of brain channels and remove all units that are not in the brain so they are not considered. N.B. as of 2025-09-12 this has been subsumed into the CIBRRIG pipeline.