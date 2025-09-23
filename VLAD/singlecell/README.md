# singlecell
Analyses and figures for single cell activity (e.g., firing rates and PETHs)

## Data
Each recording has a folder containing breath aligned PETHs for all units with respiratory modulation and preferred phase computed.

## Code
`plot_single_cell.py` performs computations of intermediate data files and figure creation. User can pass a CLI flag to change the figure file type (e.g., `-e pdf` or `-e png`)

`plot_single_cell.py -c` computes intermediate data and plots

`plot_single_cell.py -p` plots all the single cell peths. **This takes a long time!**
