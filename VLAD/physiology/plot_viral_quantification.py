import pandas as pd
from pathlib import Path
import seaborn.objects as so
import sys
sys.path.append('../')
from utils import set_style
set_style()

df = pd.read_csv('../../data/chrmine_quantification.csv').dropna()
df.set_index('Bregma', inplace=True)
df = df.melt(var_name = 'subject',value_name='neuron_count',ignore_index=False).reset_index()
p = (
    so.Plot(
        df,x='Bregma',y='neuron_count'
    )
    .add(so.Line(linewidth=2,color='k'),so.Agg())
    .add(so.Line(linewidth=0.5,alpha=0.5),group='subject',color='subject',legend=False)
    .add(so.Band(color='k'),so.Est('mean','se'))
    .label(x='Bregma (mm)',y='Neuron count')
    .layout(size=(2.5,1.5),engine='constrained')
    .limit(y=(0,None),x=(7,8))

).plot()
p.save('viral_quantification.pdf')