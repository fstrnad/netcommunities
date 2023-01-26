#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
from importlib import reload
import climnet.datasets.evs_dataset as cds

import os

reload(cds)

name = 'mswep'
grid_type = 'fekete'
grid_step = 1

scale = 'global'

vname = 'pr'

start_month = 'Jun'
end_month = 'Sep'

output_folder = 'summer_monsoon'

output_dir = '/home/strnad/data/climnet/outputs/'
plot_dir = '/home/strnad/data/climnet/plots/'
data_dir = "/home/strnad/data/climnet/outputs/climate_data/"


fname = data_dir + 'mswep_pr_1_1979_2021_ds.nc'
th_eev = 15

q_ee = 0.9
name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"
min_evs = 20
if start_month != 'Jan' or end_month != 'Dec':
    name_prefix += f"_{start_month}_{end_month}"
    min_evs = 3

can = False

dataset_file = output_dir + \
        f"/{output_folder}/{name_prefix}_1979_2021_ds.nc"


sp_grid = f'{grid_type}_{grid_step}.npy'
# %%
reload(cds)
ds = cds.EvsDataset(fname,
                    # time_range=time_range,
                    month_range=[start_month, end_month],
                    grid_step=grid_step,
                    grid_type=grid_type,
                    sp_grid=sp_grid,
                    q=q_ee,
                    th_eev=th_eev,
                    min_evs=min_evs,
                    )
# %%
ds.save(dataset_file)

