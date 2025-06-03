import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.analysis_causality import _transfer_entropy

df = pd.read_parquet('data/processed/consolidated_long.parquet')
wide = df[(df['experimental_phase']=='2 - Attack') & (df['metric_name']=='cpu_usage')].pivot_table(index='timestamp', columns='tenant_id', values='metric_value').interpolate(method='time').ffill().bfill()
print(wide.head())
tenants = wide.columns
for x in tenants:
    print(f'--- {x} ---')
    for y in tenants:
        if x != y:
            xv = wide[x].values
            yv = wide[y].values
            print(f'{x} <- {y}: TE =', _transfer_entropy(xv, yv, bins=8))
