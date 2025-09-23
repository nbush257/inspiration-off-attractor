from iblatlas.plots import plot_scalar_on_slice
from iblatlas.atlas import FranklinPaxinosAtlas,AllenAtlas
from iblatlas.regions import BrainRegions
import matplotlib.pyplot as plt
br = BrainRegions()

import numpy as np

acronyms = ['NTS','PGRNl']
acronyms = np.unique(br.acronym2acronym(acronyms, mapping='Beryl'))

coords = [-6500,-6900]
values = np.array([100,50])
ba = AllenAtlas()

for coord in coords:
    fig, ax = plot_scalar_on_slice(
        acronyms,
        values,
        coord=coord,
        slice="coronal",
        background="boundary",
        hemisphere='both',
        brain_atlas=ba,
        vector=True,
        mapping='Beryl',
        clevels=[0,100],
        linewidth=0.5,
        empty_color='w'
    )
    ax.axis('off')
    ax.axis('equal')
    plt.savefig(f'coronal_{coord}.pdf')

fig, ax = plot_scalar_on_slice(
    acronyms,
    values,
    coord=-1250,
    slice="sagittal",
    background="boundary",
    hemisphere='both',
    brain_atlas=ba,
    vector=True,
    mapping='Beryl',
    clevels=[0,100],
    empty_color='w',
    linewidth=0.5,

)
ax.axis('off')
ax.axis('equal')
plt.savefig(f'sagittal_{coord}.pdf')
