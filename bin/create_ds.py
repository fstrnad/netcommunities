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

name = 'mswep'
grid_type = 'fibonacci'
grid_step = 10

scale = 'south_asia'
vname = 'pr'

start_month = 'Jun'
end_month = 'Sep'

output_folder = 'global_monsoon'
output_folder = "global_monsoon"
output_dir = "../outputs/"
plot_dir = "../outputs/images/"
data_dir = '/home/strnad/data/climnet/outputs/climate_data/'

# %%
reload(cds)
th_eev = 15
q_ee = 0.9
name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"
min_evs = 20
if start_month != 'Jan' or end_month != 'Dec':
    name_prefix += f"_{start_month}_{end_month}"
    min_evs = 3

dataset_file = output_dir + \
    f"/{output_folder}/{name_prefix}_ds.nc"

# Already pre-proccessed large grid ds
fname = f"{data_dir}/{name}_{vname}_{grid_step}_ds.nc"
sp_grid = f'{grid_type}_{grid_step}.npy'

lat_range = [-15, 45]
lon_range = [55, 150]
time_range = ['1980-01-01', '2019-12-31']


ds = cds.EvsDataset(fname,
                    var_name=vname,
                    lon_range=lon_range,
                    lat_range=lat_range,
                    time_range=time_range,
                    month_range=[start_month, end_month],
                    grid_step=grid_step,
                    grid_type=grid_type,
                    # large_ds=True,
                    sp_grid=sp_grid,
                    q=q_ee,
                    can=False,
                    th_eev=th_eev,
                    min_evs=min_evs,
                    )
ds.save(dataset_file)
# %%
