#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
File for network of rainfall events
@author: Felix Strnad
"""
import os
import climnet.network.clim_networkx as nx
import climnet.datasets.evs_dataset as cds
from importlib import reload
import numpy as np
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
import xarray as xr

# Run the null model
name = "mswep"
grid_type = "fekete"
grid_step = 1

output_folder = "summer_monsoon"

if os.getenv("HOME") == "/home/goswami/fstrnad80":
    output_dir = "/mnt/qb/work/goswami/fstrnad80/data/climnet/outputs/"
    plot_dir = "/mnt/qb/work/goswami/fstrnad80/data/climnet/plots/"
else:
    output_dir = "/home/strnad/data/climnet/outputs/"
    plot_dir = "/home/strnad/data/climnet/plots/"


# %%
reload(cds)
q_ee = 0.9
name_prefix = f"{name}_{grid_type}_{grid_step}_{q_ee}"

start_month = "Jun"
end_month = "Sep"
min_evs = 20
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"
    min_evs = 3
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_1979_2021_ds.nc"

# %%
reload(cds)
ds = cds.EvsDataset(
            load_nc=dataset_file,
            rrevs=False,
        )
time_steps = len(
    tu.get_month_range_data(
        ds.ds.time, start_month=start_month, end_month=end_month)
)

# %%
taumax = 10
n_pmts = 1000
weighted = True
networkfile = output_dir + f"{output_folder}/{name_prefix}_ES_net.npz"

E_matrix_folder = (
    f"{output_folder}/{name_prefix}_{q_ee}_{min_evs}/"
)
null_model_file = f'null_model_ts_{time_steps}_taumax_{taumax}_npmts_{n_pmts}_q_{q_ee}_directed.npy'

# %%
q_sign_arr = np.around(
    np.array([
        # 0.9, 0.91, 0.92, 0.93, 0.94,
        0.95, 0.98, 0.99, 0.995, 0.999
    ]),
    decimals=3,
)
spars_arr = {q: None for q in q_sign_arr}

# %%
reload(nx)
reload(cds)
for q_sig in q_sign_arr:
    if gut.exist_file(dataset_file):
        ds = cds.EvsDataset(
            load_nc=dataset_file,
            rrevs=False,
        )
        time_steps = len(
            tu.get_month_range_data(
                ds.ds.time, start_month=start_month, end_month=end_month)
        )
        null_model_file = f'null_model_ts_{time_steps}_taumax_{taumax}_npmts_{n_pmts}_q_{q_ee}_directed.npy'
    else:
        raise ValueError(f'{dataset_file} does not exist!')
    nx_path_file = output_dir + \
        f"{output_folder}/{name_prefix}_{q_sig}_ES_nx.gml.gz"
    if not gut.exist_file(nx_path_file):

        Net = nx.Clim_NetworkX(dataset=ds,
                               taumax=taumax,
                               weighted=weighted,)
        gut.myprint(f"Use q = {q_sig}")
        Net.create(
            method='es',
            null_model_file=null_model_file,
            E_matrix_folder=E_matrix_folder,
            q_sig=q_sig
        )
        Net.save(nx_path_file)

# %%
# reload(nx)
Net = nx.Clim_NetworkX(dataset=ds,
                       taumax=taumax,
                       weighted=weighted,)
Net.create(
    method='es',
    null_model_file=null_model_file,
    E_matrix_folder=E_matrix_folder,
    q_sig=q_sig,
    # num_cpus=1
)

Net.save(nx_path_file)
# %%
