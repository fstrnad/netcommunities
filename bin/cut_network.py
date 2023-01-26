#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Strnad
"""
# %%
import geoutils.utils.time_utils as tu
import numpy as np
from importlib import reload
import geoutils.plotting.plots as cplt
import climnet.datasets.evs_dataset as eds
import climnet.network.clim_networkx as cnx

name = "mswep"
grid_type = "fekete"
grid_step = 1

vname = "pr"

output_dir = "/home/strnad/data/climnet/outputs/"
plot_dir = "/home/strnad/data/climnet/plots/"

# %%
# Load Network file EE
reload(eds)
q_ee = .9
scale = "global"
output_folder = "global_monsoon"
name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"
output_folder = "summer_monsoon"
name_prefix = f"{name}_{grid_type}_{grid_step}_{q_ee}"

start_month = "Jun"
end_month = "Sep"
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_1979_2021_ds.nc"
# dataset_file = output_dir + f"/{output_folder}/{name_prefix}_ds.nc"

ds = eds.EvsDataset(
    load_nc=dataset_file,
    rrevs=False
)

# %%
reload(cplt)
time = ds.ds.time
mmap = tu.get_month_range_data(dataset=ds.ds['pr'],
                               start_month='Jun',
                               end_month='Sep').mean(dim='time')
im = cplt.plot_map(
    mmap,
    label=f"Precipitation",
    projection="Robinson",
    plot_type="contourf",
    ds=ds,
    significance_mask=ds.mask,
    cmap="Blues",
    vmin=0,
    vmax=15,
    levels=5
)

# %%
reload(cnx)
q_sig = 0.95
networkfile = (
    output_dir + f"{output_folder}/{name_prefix}_{q_sig}_lb_ES_nx.gml.gz"
)
networkfile = (
    output_dir + f"{output_folder}/{name_prefix}_{q_sig}_ES_nx.gml.gz"
)
networkfile = (
    output_dir + f"{output_folder}/{name_prefix}_{q_sig}_lb_ES_nx.gml.gz"
)

net = cnx.Clim_NetworkX(dataset=ds,
                        nx_path_file=networkfile)


# %%
# cut network
# Use South Asian Monsoon Domain
lon_range = [-50, 140]
lat_range = [-20, 50]
net.cut_network(lon_range=lon_range,
                lat_range=lat_range)
# %%
nx_path_file = output_dir + \
    f"{output_folder}/{name_prefix}_{q_sig}_lat_{lat_range}_ES_nx.gml.gz"
nx_path_file = output_dir + \
    f"{output_folder}/{name_prefix}_{q_sig}_lat_{lat_range}_lb_ES_nx.gml.gz"


new_ds_file = output_dir + \
    f"/{output_folder}/{name_prefix}_1979_2021_lat_{lat_range}_lb_ds.nc"
net.save(nx_path_file, ds_savepath=new_ds_file)
# %%
# Plot new cutted network
s_lat_range = [25, 35]
s_lon_range = [70, 80]
link_dict = net.get_edges_nodes_for_region(
    lon_range=s_lon_range, lat_range=s_lat_range, binary=False
)
central_longitude = np.mean(s_lon_range)
# Plot nodes where edges go to
im = cplt.plot_map(
    link_dict['target_map'],
    label=f"Local degree",
    projection="PlateCarree",
    central_longitude=central_longitude,
    plt_grid=True,
    plot_type="colormesh",
    ds=net.ds,
    significance_mask=net.ds.mask,
    cmap="Greens",
    vmin=0,
    vmax=10,
    levels=5,
    bar=True,
    alpha=0.7,
    size=10,
    # tick_step=2,
    fillstyle="none",
)

# im = cplt.plot_edges(
#     cnx.ds,
#     link_dict['el'][::20],
#     ax=im["ax"],
#     significant_mask=True,
#     orientation="vertical",
#     projection="EqualEarth",
#     plt_grid=True,
#     lw=0.2,
#     alpha=0.6,
#     color="grey",
# )

cplt.plot_rectangle(
    ax=im["ax"],
    lon_range=s_lon_range,
    lat_range=s_lat_range,
    color="magenta",
    lw=3,
    zorder=11
)
savepath = f"{plot_dir}/summer_monsoon/ee_plots/lb_cut_{name_prefix}_{s_lon_range}_links.png"
cplt.save_fig(savepath)
# %%
