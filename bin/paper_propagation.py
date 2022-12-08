#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
import matplotlib.pyplot as plt
import geoutils.geodata.moist_static_energy as mse
import geoutils.geodata.helmholtz_decomposition as hd
import geoutils.geodata.base_dataset as bds
import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import xarray as xr
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
from importlib import reload
import numpy as np
import geoutils.tsa.time_series_analysis as tsa
import geoutils.tsa.time_clustering as tcl
import geoutils.plotting.plots as cplt
import geoutils.geodata.wind_dataset as wds

# Run the null model
name = "mswep"
grid_type = "fibonacci"
grid_step = 0.5

grid_type = "fekete"
grid_step = 1

scale = "south_asia"

output_folder = "global_monsoon"


output_dir = "/home/strnad/data/climnet/outputs/"
plot_dir = "/home/strnad/data/climnet/plots/"

q_ee = .9
name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"
start_month = "Jun"
end_month = "Sep"
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"
q_sig = 0.95


# %%
# SST data
reload(bds)
dataset_file = output_dir + \
    f"/climate_data/era5_sst_{2.5}_ds.nc"

ds_sst = bds.BaseDataset(load_nc=dataset_file,
                         can=True,
                         detrend=True,
                         an_types=['JJAS', 'month', 'dayofyear'],
                         )


# %%
# Load Wind-Field data
reload(wds)
nc_files_u = []
nc_files_v = []
nc_files_w = []

plevels = [200, 300,
           500,
           800, 850,
           900, 1000]
plevels = [100, 200, 300,
           400, 500, 600,
           700, 800, 850,
           900, 1000]
# plevels = [
#     50,
#     100, 150, 200,
#     250, 300, 350,
#     400, 450, 500,
#     550, 600, 650,
#     700, 750, 800,
#     850, 900, 950,
#     1000]

# plevels = [200,
#            400,
#            500,
#            600,
#            800,
#            1000]

# plevels = [
#     100,
#     900
# ]
for plevel in plevels:
    dataset_file_u = output_dir + \
        f"/climate_data/era5_u_{2.5}_{plevel}_ds.nc"
    nc_files_u.append(dataset_file_u)
    dataset_file_v = output_dir + \
        f"/climate_data/era5_v_{2.5}_{plevel}_ds.nc"
    nc_files_v.append(dataset_file_v)
    dataset_file_w = output_dir + \
        f"/climate_data/era5_w_{2.5}_{plevel}_ds.nc"
    nc_files_w.append(dataset_file_w)

ds_wind = wds.Wind_Dataset(load_nc_arr_u=nc_files_u,
                           load_nc_arr_v=nc_files_v,
                           load_nc_arr_w=nc_files_w,
                           plevels=plevels,
                           can=True,
                           an_types=['JJAS'],
                           month_range=['Jun', 'Sep']
                           )
# %%
# Helmholtz Decomposition and Rossby Wave Source
reload(hd)


ds_hd = hd.HelmholtzDecomposition(ds_wind=ds_wind,
                                  an_types=['JJAS'],
                                  )
# %%
# Vertical Cuts Walker Circulation
reload(cplt)
an_type = 'JJAS'

u_var_type = f'u_an_{an_type}'
v_var_type = f'v_an_{an_type}'
w_var_type = f'w_an_{an_type}'

c_lat_range = [-0, 10]
c_lon_range = [50, -80]

wind_data_lon = sput.cut_map(ds_wind.ds[[u_var_type,
                                        v_var_type,
                                        w_var_type]],
                             lon_range=c_lon_range,
                             lat_range=c_lat_range, dateline=True)
# %%
s_lat_range = [-10, 30]
s_lon_range = [70, 80]
wind_data_lat = sput.cut_map(ds_wind.ds[[u_var_type,
                                         v_var_type,
                                         w_var_type]],
                             lon_range=s_lon_range,
                             lat_range=s_lat_range, dateline=False)

var_type_wind_u = 'mf_u_an_JJAS'
var_type_wind_v = 'mf_v_an_JJAS'
range_data_lon = sput.cut_map(ds_hd.ds[[var_type_wind_u,
                                        var_type_wind_v]],
                              lon_range=c_lon_range,
                              lat_range=c_lat_range, dateline=True)
range_data_lat = sput.cut_map(ds_hd.ds, lon_range=s_lon_range,
                              lat_range=s_lat_range, dateline=False)

# %%
# Moist static energy analysis
reload(mse)

nc_files_q = []
nc_files_t = []
nc_files_z = []


plevels = [
    200,
    600,
    850,
    1000,
]
for plevel in plevels:
    dataset_file_q = output_dir + \
        f"/climate_data/era5_q_{2.5}_{plevel}_ds.nc"
    nc_files_q.append(dataset_file_q)
    dataset_file_t = output_dir + \
        f"/climate_data/era5_t_{2.5}_{plevel}_ds.nc"
    nc_files_t.append(dataset_file_t)
    dataset_file_z = output_dir + \
        f"/climate_data/era5_z_{2.5}_{plevel}_ds.nc"
    nc_files_z.append(dataset_file_z)

ds_mse = mse.MoistStaticEnergy(load_nc_arr_q=nc_files_q,
                               load_nc_arr_t=nc_files_t,
                               load_nc_arr_z=nc_files_z,
                               plevels=plevels,
                               can=True,
                               an_types=['JJAS'],
                               month_range=['Jun', 'Sep']
                               )


# %%
# Define time points
reload(tu)
B_max = 6
cd_folder = 'graph_tool'
savepath_loc = (
    plot_dir +
    f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_prob_maps.npy"
)

loc_dict = gut.load_np_dict(savepath_loc)
region = 'EIO'
# EE TS
this_dict = loc_dict[region]
evs = this_dict['data'].evs
t_all = tsa.get_ee_ts(evs=evs)

comp_th = 0.9
t_jjas = tu.get_month_range_data(t_all, start_month='Jun',
                                 end_month='Sep')
tps_dict = gut.get_locmax_composite_tps(t_jjas, q=comp_th)

# Tps are the time points of which the composites are made off
tps = tps_dict['peaks']

# %%
# OLR data
reload(bds)
dataset_file = output_dir + \
    f"/climate_data/era5_ttr_{1}_ds.nc"

ds_olr = bds.BaseDataset(load_nc=dataset_file,
                         can=True,
                         an_types=['dayofyear', 'JJAS'],
                         detrend=True
                         )

# %%
# Plot Hovmöller lon lat
an_type = 'dayofyear'
var_type = f'an_{an_type}'
dist_days = 20
rm_tps = tu.remove_consecutive_tps(tps=tps, steps=dist_days)

hov_data_lon = tu.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                            tps=rm_tps,
                                            lat_range=[-5, 5],
                                            lon_range=[50, 160],
                                            num_days=30,
                                            start=5,
                                            gf=(2, 2),
                                            var=var_type,
                                            zonal=True)

hov_data_lat = tu.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                            tps=rm_tps,
                                            lat_range=[-10, 22],
                                            lon_range=[70, 80],
                                            num_days=30,
                                            start=5,
                                            gf=(2, 2),
                                            var=var_type,
                                            zonal=False)

# %%
# Create lon and lat hovmöller diagrams
reload(tu)
an_type = 'dayofyear'
var_type = f'an_{an_type}'
dist_days = 20
rm_tps = tu.remove_consecutive_tps(tps=tps, steps=dist_days)
# Hovmöller along longitudes
hov_data_lon_kmeans = tu.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                                   tps=rm_tps,
                                                   lat_range=[-5, 5],
                                                   lon_range=[50, 160],
                                                   num_days=30,
                                                   start=5,
                                                   gf=(5, 5),
                                                   var=var_type,
                                                   zonal=True)
hov_data_lat_kmeans = tu.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                                   tps=rm_tps,
                                                   lat_range=[-10, 22],
                                                   lon_range=[70, 80],
                                                   num_days=30,
                                                   start=5,
                                                   gf=(5, 5),
                                                   var=var_type,
                                                   zonal=False)
# %%
# Cluster longs and lats together
reload(tcl)
n = 3
an_type = 'dayofyear'
var_type = f'an_{an_type}'

type_names = [
    'Stationary',
    'Eastward Blocked',
    'Canonical',
]

k_means_tps_lon_lat = tcl.tps_cluster_2d_data([hov_data_lon_kmeans,
                                               hov_data_lat_kmeans],
                                              tps=rm_tps,
                                              method='kmeans',
                                              n_clusters=n,
                                              random_state=0,
                                              n_init=100,
                                              max_iter=3000,
                                              key_names=type_names,
                                              rm_ol=True
                                              #   metric='correlation'
                                              #   metric='euclidean',
                                              #   plot_statistics=True,
                                              )
n = 3
savepath = f"{plot_dir}/{output_folder}/propagation/olr_hovmoeller_{n}_cluster.npy"
# gut.save_np_dict(k_means_tps_lon_lat,
#                  savepath)
# %%
# Plot the Hovemölller diagrams
n = 3
type_names = [
    'Canonical',
    'Eastward Blocked',
    'Stationary',
]
savepath = f"{plot_dir}/{output_folder}/propagation/olr_hovmoeller_{n}_cluster.npy"

k_means_tps_lon_lat = gut.load_np_dict(sp=savepath)
# %%
# Plot lon lat Hovmoeller diagrams
reload(cplt)
vtimes = hov_data_lon.day
lons = hov_data_lon.lon.values
lats = hov_data_lat.lat.values

vmax = 5e4
vmin = -vmax
an_type = 'dayofyear'
var_type = f'an_{an_type}'
label = f'OLR Anomalies (wrt {var_type})'
n = len(k_means_tps_lon_lat)
im = cplt.create_multi_plot(nrows=n, ncols=2,
                            hspace=0.4, wspace=0.2,
                            )

plot_order = type_names

for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    k_data = tu.get_sel_tps_ds(
        ds=hov_data_lon, tps=sel_tps[:])
    mean, pvalues_ttest = sut.ttest_field(
        k_data, hov_data_lon, zdim=('day', 'lon'))
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    h_im = cplt.plot_2D(x=lons, y=vtimes,
                        ax=im['ax'][idx*2],
                        z=mean,
                        levels=9,
                        title=f'Lon: {group} ({len(sel_tps)} cases)',
                        y_title=1.,
                        orientation='vertical',
                        cmap='RdBu_r',
                        tick_step=2,
                        round_dec=2,
                        plot_type='contourf',
                        vmin=vmin, vmax=vmax,
                        xlabel='Longitude [degree]',
                        ylabel='day',
                        bar=False,
                        # significance_mask=mask,
                        hatch_type='..'
                        )

    k_data = tu.get_sel_tps_ds(
        ds=hov_data_lat, tps=sel_tps[:])
    mean, pvalues_ttest = sut.ttest_field(
        k_data, hov_data_lat, zdim=('day', 'lat'))
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    h_im = cplt.plot_2D(x=lats, y=vtimes,
                        z=mean,
                        ax=im['ax'][idx*2 + 1],
                        levels=12,
                        title=f'Lat: {group} ({len(sel_tps)} cases)',
                        y_title=1,
                        orientation='vertical',
                        cmap='RdBu_r',
                        tick_step=2,
                        round_dec=2,
                        plot_type='contourf',
                        vmin=vmin, vmax=vmax,
                        xlabel='Latitude [degree]',
                        ylabel='day',
                        bar=False,
                        extend='both',
                        # significance_mask=mask,
                        hatch_type='..'
                        )

cplt.add_colorbar(im=h_im, fig=im['fig'], sci=3,
                  label=label,
                  x_pos=0.1,
                  height=0.02,
                  #   y_pos=-0.02
                  )
savepath = plot_dir +\
    f"{output_folder}/propagation/olr_hovmoeller_{region}_{var_type}_k_means_lon_lat.png"
cplt.save_fig(savepath, fig=im['fig'])

# %%
# SST data
reload(bds)
dataset_file = output_dir + \
    f"/climate_data/era5_sst_{2.5}_ds.nc"

ds_sst = bds.BaseDataset(load_nc=dataset_file,
                         can=True,
                         detrend=True,
                         an_types=['JJAS', 'month', 'dayofyear'],
                         )

# %%
# Temperatures
reload(sut)
reload(cplt)
an_type = 'dayofyear'
var_type = f'an_{an_type}'
label = rf'Sea Surface Temperature Anomalies (wrt {an_type}) [K]'
n = len(k_means_tps_lon_lat)

vmax = 1
vmin = -vmax

lon_range_c = [20, -60]
lat_range_c = [-70, 70]

# ds_sst.lon_2_360()

im = cplt.create_multi_plot(nrows=1, ncols=3,
                            # hspace=0.4,
                            wspace=0.2,
                            projection='PlateCarree',
                            central_longitude=180,
                            end_idx=n)

for idx, (group, sel_tps) in enumerate(k_means_tps_lon_lat.items()):
    data_cut = sput.cut_map(ds=ds_sst.ds, lon_range=lon_range_c,
                            lat_range=lat_range_c,
                            dateline=True)
    data_cut = tu.get_month_range_data(
        data_cut, start_month='Jun', end_month='Sep')
    this_comp_ts = tu.get_sel_tps_ds(
        data_cut, tps=sel_tps)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts[var_type], data_cut[var_type])
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_comp = cplt.plot_map(mean,
                            ax=im['ax'][idx],
                            title=f'{group}',
                            cmap='RdBu_r',
                            plot_type='contourf',
                            levels=9,
                            vmin=vmin, vmax=vmax,
                            bar=False,
                            plt_grid=True,
                            extend='both',
                            orientation='horizontal',
                            significance_mask=mask,
                            hatch_type='..'
                            )

    loc_lons = this_dict['data'].lon
    loc_lats = this_dict['data'].lat

    locs = gut.zip_2_lists(loc_lons, loc_lats)

    loc_map = ds_sst.get_map_for_locs(locations=locs)
    loc_map = sput.cut_map(loc_map, lon_range=lon_range_c,
                           lat_range=lat_range_c,
                           dateline=True)
    cplt.plot_map(loc_map,
                  ax=im['ax'][idx],
                  plot_type='contour',
                  color='magenta',
                  levels=1,
                  #   vmin=-5, vmax=5,
                  bar=False,
                  plt_grid=True,
                  lw=2,)

cplt.add_colorbar(im=im_comp, fig=im['fig'],
                  sci=None,
                  tick_step=3,
                  round_dec=2,
                  label=label,
                  x_pos=0.2,
                  width=0.6,
                  height=0.04,
                  y_pos=0.05
                  )

savepath = plot_dir +\
    f"{output_folder}/propagation/sst_{region}_groups.png"
cplt.save_fig(savepath, fig=im['fig'])
# %%
reload(cplt)
im = cplt.create_multi_plot(nrows=1, ncols=3,
                            # hspace=0.4,
                            wspace=0.2,
                            projection='PlateCarree',
                            central_longitude=180,
                            end_idx=n)
im_comp = cplt.plot_map(
    # mean,
    ds_sst.mask,
    ax=im['ax'][0],
    title=f'{group}',
    # projection='Robinson',
    # set_global=True,
    cmap='RdBu_r',
    plot_type='contourf',
    levels=9,
    vmin=vmin, vmax=vmax,
    bar=True,
    plt_grid=True,
    # extend='both',
    orientation='horizontal',
    significance_mask=ds_sst.mask,
    hatch_type='..',
    central_longitude=180
)


# %%
# Plot for paper OLR Hovmöller + SST patterns
reload(cplt)
reload(sut)
lon_range_c = [-70, 35]
lat_range_c = [-75, 75]

lon_ticks = [60, 80, 100, 120, 140]
lon_ticklabels = ['60°E', '80°E', '100°E', '120°E', '140°E']
lat_ticks = [-10, 0, 10, 20, 30]
lat_ticklabels = ['10°S', '0°', '10°N', '20°N', '30°N']

# For Hovmöller Diagrams
vtimes = hov_data_lon.day
lons = hov_data_lon.lon.values
lats = hov_data_lat.lat.values
vmax_vert = 6e4
vmin_vert = -vmax_vert

# For SST
data_cut = sput.cut_map(ds=ds_sst.ds, lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=True)
data_cut = tu.get_month_range_data(
    data_cut, start_month='Jun', end_month='Sep')
vmax_sst = 1
vmin_sst = -vmax_sst

an_type = 'dayofyear'
var_type = f'an_{an_type}'
label_hov = r'OLR (wrt JJAS)-Anomalies [$\frac{W}{m^2}$]'


ncols = 2
nrows = 3
im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            # figsize=(14, 6),
                            orientation='horizontal',
                            hspace=0.25,
                            wspace=0.2,
                            )

axs = im['ax']
fig = im['fig']
plot_order = ['Canonical',
              'Eastward Blocked',
              'Stationary']

for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    k_data = tu.get_sel_tps_ds(
        ds=hov_data_lon, tps=sel_tps).mean(dim='time')
    h_im = cplt.plot_2D(x=lons, y=vtimes,
                        ax=axs[idx*ncols],
                        z=k_data,
                        levels=9,
                        vertical_title=f'{group} ({len(sel_tps)} cases)',
                        title='Zonal Hovmöller diagrams' if idx == 0 else None,
                        x_title_offset=-0.3,
                        orientation='vertical',
                        cmap='bwr',
                        tick_step=2,
                        round_dec=2,
                        plot_type='contourf',
                        vmin=vmin_vert, vmax=vmax_vert,
                        xlabel='Longitude [degree]' if idx % nrows == 2 else None,
                        ylabel='day',
                        bar=False,
                        xticks=lon_ticks,
                        xticklabels=lon_ticklabels,
                        )
    cplt.plot_vline(
        ax=axs[idx*ncols],
        x=110,
        lw=3)
    cplt.plot_vline(
        ax=axs[idx*ncols],
        x=130,
        lw=3,
        label='Maritime Continent barrier')

    k_data = tu.get_sel_tps_ds(
        ds=hov_data_lat, tps=sel_tps).mean(dim='time')
    h_im = cplt.plot_2D(x=lats, y=vtimes,
                        z=k_data,
                        ax=axs[idx*ncols + 1],
                        levels=9,
                        title='Meridional Hovmöller diagrams' if idx == 0 else None,
                        orientation='vertical',
                        cmap='bwr',
                        tick_step=2,
                        round_dec=2,
                        plot_type='contourf',
                        vmin=vmin_vert, vmax=vmax_vert,
                        xlabel='Latitude [degree]' if idx % nrows == 2 else None,
                        ylabel='day',
                        bar=False,
                        extend='both',
                        xticks=lat_ticks,
                        xticklabels=lat_ticklabels,
                        )


cplt.add_colorbar(im=h_im, fig=fig,
                  sci=3,
                  label=label_hov,
                  x_pos=0.2,
                  width=0.6,
                  height=0.02,
                  y_pos=-0.01,
                  tick_step=3)
cplt.set_legend(ax=im['ax'][-2],
                fig=im['fig'],
                # order=order,
                loc='outside',
                ncol_legend=3,
                box_loc=(0.2, 0.06)
                )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/propagation_olr_hovmoeller_k_means_lon_lat_{region}.png"
cplt.save_fig(savepath, fig=fig)

# %%
# Plot SSTs only
reload(tu)
reload(cplt)
lon_range_c = [-70, 35]
lat_range_c = [-75, 75]

an_type = 'JJAS'
var_type = f'an_{an_type}'
label = f'day'

# For SST
data_cut = sput.cut_map(ds=ds_sst.ds, lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=True)
data_cut = tu.get_month_range_data(
    data_cut, start_month=start_month,
    end_month=end_month)

# %%
vmax_sst = 1
vmin_sst = -vmax_sst

an_type = 'dayofyear'
var_type = f'an_{an_type}'
label_sst = r'SST (wrt JJAS)-Anomalies [K]'

ncols = 3
im = cplt.create_multi_plot(nrows=1,
                            ncols=ncols,
                            # figsize=(14, 6),
                            projection='PlateCarree',
                            orientation='horizontal',
                            # hspace=0.25,
                            central_longitude=180,
                            wspace=0.2,
                            end_idx=3
                            )
q_th = 0.05
plot_order = ['Canonical',
              'Eastward Blocked',
              'Stationary']
for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut, tps=sel_tps)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts[var_type], data_cut[var_type])
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_sst = cplt.plot_map(mean,
                           ax=im['ax'][idx],
                           title=f'{group}',
                           cmap='RdBu_r',
                           plot_type='contourf',
                           levels=9,
                           vmin=vmin_sst, vmax=vmax_sst,
                           projection='PlateCarree',
                           bar=False,
                           plt_grid=True,
                           extend='both',
                           orientation='horizontal',
                           significance_mask=mask,
                           hatch_type='..'
                           )

cplt.add_colorbar(im=im_sst, fig=im['fig'],
                  sci=None,
                  label=label_sst,
                  x_pos=0.2,
                  width=0.6,
                  height=0.05,
                  y_pos=0.05,
                  round_dec=2,
                  tick_step=3
                  )


savepath = plot_dir +\
    f"{output_folder}/paper_plots/sst_background_all.png"
cplt.save_fig(savepath, fig=im['fig'])

# %%
# Plot for paper multiple pressure levels wind fields
reload(cplt)
reload(sut)
lon_range_c = [-70, 35]
lat_range_c = [-75, 75]

var_type_wind_u = 'mf_u_an_JJAS'
var_type_wind_v = 'mf_v_an_JJAS'
var_type_map_u = 'u_chi_an_JJAS'
var_type_map_v = 'v_chi_an_JJAS'
var_type_map_u = 'mf_u_an_JJAS'
var_type_map_v = 'mf_v_an_JJAS'

# For MSF Field
plevel_range = [300, 400, 500]
data_cut_hd = sput.cut_map(ds=ds_hd.ds.sel(plevel=plevel_range)[
    [var_type_map_u, var_type_map_v]],
    lon_range=lon_range_c,
    lat_range=lat_range_c,
    dateline=True)
data_cut_hd = data_cut_hd.mean(dim='plevel')

# %%
# Stream function background plot
reload(cplt)

an_type = 'JJAS'
var_type = f'an_{an_type}'
label_msf_u = r'$\Psi_u$ JJAS-anomalies [$\frac{kg}{m^2s}$]'
label_msf_v = r'$\Psi_v$ JJAS-anomalies [$\frac{kg}{m^2s}$]'
label_vert_u = r'$\bar{\Psi}_u$  JJAS-anomalies [$\frac{kg}{m^2s}$]'
label_vert_v = r'$\bar{\Psi}_u$  JJAS-anomalies [$\frac{kg}{m^2s}$]'

vmax_vert = 4e4
vmin_vert = -vmax_vert
vmax_map = 3e4
vmin_map = -vmax_map


lon_ticks = [60, 120, 180, 240]
lon_ticklabels = ['60°E', '120°E', '180°', '120°W']
lat_ticks = [-10, 0, 10, 20, 30]
lat_ticklabels = ['10°S', '0°', '10°N', '20°N', '30°N']

sci = 3
sci_map = 3
n = len(k_means_tps_lon_lat)
ncols = 4
proj = cplt.get_projection(projection='PlateCarree',
                           central_longitude=180)
fig = plt.figure(
    figsize=(17, 14)
)
ax1 = fig.add_axes([0., 0.7, .2, .2],)
ax2 = fig.add_axes([0.25, 0.7, .24, .2], projection=proj)
ax3 = fig.add_axes([0.55, 0.7, .2, .2])
ax4 = fig.add_axes([0.81, 0.7, 0.24, 0.2], projection=proj)
ax5 = fig.add_axes([0.0, 0.45, 0.2, 0.2])
ax6 = fig.add_axes([0.25, 0.45, .24, .2], projection=proj)
ax7 = fig.add_axes([0.55, 0.45, 0.2, 0.2])
ax8 = fig.add_axes([0.81, 0.45, 0.24, 0.2], projection=proj)
ax9 = fig.add_axes([0.0, 0.2, 0.2, 0.2])
ax10 = fig.add_axes([0.25, 0.2, .24, .2], projection=proj)
ax11 = fig.add_axes([0.55, 0.2, 0.2, 0.2])
ax12 = fig.add_axes([0.81, 0.2, 0.24, 0.2], projection=proj)


axs = [ax1, ax2, ax3, ax4,
       ax5, ax6, ax7, ax8,
       ax9, ax10, ax11, ax12,
       ]
cplt.enumerate_subplots(axs=axs)


plot_order = ['Canonical',
              'Eastward Blocked',
              'Stationary']

for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    k_data_lon = tu.get_sel_tps_ds(
        ds=range_data_lon[var_type_wind_u], tps=sel_tps).mean(dim=['lat', 'time'])
    k_data_lat = tu.get_sel_tps_ds(
        ds=range_data_lat[var_type_wind_v], tps=sel_tps).mean(dim=['lon', 'time'])
    lons = k_data_lon.lon
    lats = k_data_lat.lat
    plevels = k_data_lon.plevel

    # Wind fields
    k_data_u = tu.get_sel_tps_ds(
        ds=wind_data_lon[u_var_type], tps=sel_tps).mean(dim=['lat', 'time'])
    k_data_uw = tu.get_sel_tps_ds(
        ds=wind_data_lon[w_var_type], tps=sel_tps).mean(dim=['lat', 'time'])
    k_data_v = tu.get_sel_tps_ds(
        ds=wind_data_lat[v_var_type], tps=sel_tps).mean(dim=['lon', 'time'])
    k_data_vw = tu.get_sel_tps_ds(
        ds=wind_data_lat[w_var_type], tps=sel_tps).mean(dim=['lon', 'time'])

    h_im_u = cplt.plot_2D(x=lons, y=plevels,
                          ax=axs[idx*ncols + 0],
                          z=k_data_lon,
                          levels=9,
                          vertical_title=f'{group}',
                          title='Zonal Circulation' if idx == 0 else None,
                          x_title_offset=-0.4,
                          cmap='coolwarm',
                          tick_step=2,
                          round_dec=2,
                          plot_type='colormesh',
                          extend='both',
                          vmin=vmin_vert, vmax=vmax_vert,
                          xlabel='Longitude [degree]' if idx % ncols == 2 else None,
                          ylabel='Pressure Level [hPa]',
                          bar=False,
                          flip_y=True,
                          xticks=lon_ticks,
                          xticklabels=lon_ticklabels
                          )

    dict_w = cplt.plot_wind_field(
        ax=axs[idx*ncols],
        u=k_data_u,
        v=k_data_uw*-100,
        x_vals=k_data_uw.lon,
        y_vals=k_data_uw.plevel,
        steps=1,
        x_steps=4,
        transform=False,
        scale=50,
        key_length=2,
        wind_unit='m/s | 0.02 hPa/s',
        key_loc=(0.95, 1.05) if idx == 0 else None
    )

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_hd, tps=sel_tps)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts[var_type_map_u], data_cut_hd[var_type_map_u])
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_wind_u = cplt.plot_map(mean,
                              ax=axs[idx*ncols + 1],
                              title=r'400hPa-600hPa $\bar{\Psi}_u$' if idx == 0 else None,
                              y_title=1.32,
                              cmap='coolwarm',
                              plot_type='contourf',
                              levels=9,
                              vmin=vmin_map, vmax=vmax_map,
                              projection='PlateCarree',
                              bar=False,
                              plt_grid=True,
                              extend='both',
                              orientation='horizontal',
                              significance_mask=mask,
                              hatch_type='..'
                              )

    h_im_v = cplt.plot_2D(x=lats, y=plevels,
                          z=k_data_lat,
                          title='Meridional Circulation' if idx == 0 else None,
                          ax=axs[idx*ncols + 2],
                          levels=9,
                          orientation='vertical',
                          cmap='coolwarm',
                          tick_step=2,
                          round_dec=2,
                          plot_type='colormesh',
                          extend='both',
                          vmin=vmin_vert, vmax=vmax_vert,
                          xlabel='Latitude [degree]' if idx % ncols == 2 else None,
                          xticks=lat_ticks,
                          xticklabels=lat_ticklabels,
                          # ylabel='Pressure Level [hPa]',
                          bar=False,
                          flip_y=True,
                          )

    dict_w = cplt.plot_wind_field(ax=axs[idx*ncols + 2],
                                  u=k_data_v,
                                  v=k_data_vw*-50,
                                  x_vals=k_data_vw.lat,
                                  y_vals=k_data_vw.plevel,
                                  steps=1,
                                  key_length=2,
                                  transform=False,
                                  pivot='tip',
                                  wind_unit='m/s | 0.02 hPa/s',
                                  scale=30,
                                  key_loc=(0.95, 1.05) if idx == 0 else None
                                  )

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_hd, tps=sel_tps)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts[var_type_map_v], data_cut_hd[var_type_map_v])
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_wind_v = cplt.plot_map(mean,
                              ax=axs[idx*ncols + 3],
                              title=r'400hPa-600hPa $\bar{\Psi}_v$' if idx == 0 else None,
                              y_title=1.32,
                              cmap='coolwarm',
                              plot_type='contourf',
                              levels=9,
                              vmin=vmin_map, vmax=vmax_map,
                              projection='PlateCarree',
                              bar=False,
                              plt_grid=True,
                              extend='both',
                              orientation='horizontal',
                              significance_mask=mask,
                              hatch_type='..'
                              )


cplt.add_colorbar(im=h_im_u, fig=fig,
                  sci=sci,
                  label=label_msf_u,
                  x_pos=0.0,
                  width=0.2,
                  height=0.02,
                  y_pos=0.12,
                  tick_step=3)


cplt.add_colorbar(im=im_wind_u, fig=fig,
                  sci=sci_map,
                  label=label_vert_u,
                  x_pos=0.25,
                  width=0.25,
                  height=0.02,
                  y_pos=0.12,
                  round_dec=2,
                  tick_step=3
                  )

cplt.add_colorbar(im=h_im_v, fig=fig,
                  sci=sci,
                  label=label_msf_v,
                  x_pos=0.55,
                  width=0.2,
                  height=0.02,
                  y_pos=0.12,
                  tick_step=3)

cplt.add_colorbar(im=im_wind_v, fig=fig,
                  sci=sci_map,
                  label=label_vert_v,
                  x_pos=0.8,
                  width=0.25,
                  height=0.02,
                  y_pos=0.12,
                  round_dec=2,
                  tick_step=3
                  )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/propagation_msf_lon_lat_all.png"
cplt.save_fig(savepath, fig=fig)

# %%
# Plot together Eastward propagation in terms of Kelvin and Rossby Waves
reload(cplt)
reload(sut)

lon_range_c = [40, 180]
lat_range_c = [-30, 50]
plevel_gph = 850
plevel_u = 850
plevel_v = 850

an_type = 'JJAS'
label_gph = rf'{plevel_gph}-hPa GPH (JJAS)-Anomalies [m]'
label_gph = rf'{plevel_gph}-hPa Q JJAS-Anomalies []'
label_uwind = rf'{plevel_u}-hPa u-winds JJAS anomalies [m/s]'
label_vwind = rf'{plevel_u}-hPa v-winds JJAS anomalies [m/s]'

gph_var_type = f'z_an_{an_type}'
sh_var_type = f'q_an_{an_type}'
u_var_type = f'u_an_{an_type}'
v_var_type = f'v_an_{an_type}'
w_var_type = f'w_an_{an_type}'

var_type_kelvin = sh_var_type

vmax_gph = 1.5e-3
vmin_gph = -vmax_gph
vmax_uwind = 4.5
vmin_uwind = -vmax_uwind
vmax_vwind = 1.5
vmin_vwind = -vmax_vwind

step = 4


data_cut_gph = sput.cut_map(ds=ds_mse.ds[
    [gph_var_type, sh_var_type]].sel(plevel=[plevel_gph]),
    lon_range=lon_range_c,
    lat_range=lat_range_c,
    dateline=False)


data_cut_wind = sput.cut_map(ds=ds_wind.ds[
    [u_var_type, v_var_type, w_var_type]].sel(plevel=[plevel_u]),
    # [u_var_type, v_var_type]].sel(plevel=[plevel_u, plevel_gph]),
    lon_range=lon_range_c,
    lat_range=lat_range_c,
    dateline=False)

# data_cut_gph = data_cut_wind

ncols = 3
im = cplt.create_multi_plot(nrows=3, ncols=3,
                            # hspace=0.4,
                            wspace=0.2,
                            projection='PlateCarree',
                            central_longitude=180,
                            # figsize=(14,14)
                            # end_idx=9
                            )

plot_order = ['Canonical',
              'Eastward Blocked',
              'Stationary']

for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    # sel_tps_step = tu.add_time_step_tps(tps=sel_tps,
    #                                     time_step=step)
    sel_tps_step = tu.get_periods_tps(tps=sel_tps,
                                      step=step)
    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_gph[var_type_kelvin].sel(plevel=plevel_gph), tps=sel_tps)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_gph[var_type_kelvin].sel(plevel=plevel_gph))
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_gph = cplt.plot_map(mean,
                           ax=im['ax'][idx*ncols],
                           title='Kelvin Wave Response' if idx == 0 else None,
                           vertical_title=f'{group}',
                           cmap='PRGn',
                           plot_type='contourf',
                           levels=9,
                           vmin=vmin_gph, vmax=vmax_gph,
                           projection='PlateCarree',
                           plt_grid=True,
                           orientation='horizontal',
                           significance_mask=mask,
                           extend='both',
                           bar=True if idx == 2 else None,
                           sci=-4,
                           label=label_gph,
                           round_dec=2,
                           tick_step=3
                           )
    this_comp_ts_u = tu.get_sel_tps_ds(
        data_cut_wind[u_var_type].sel(plevel=plevel_gph), tps=sel_tps)
    this_comp_ts_v = tu.get_sel_tps_ds(
        data_cut_wind[v_var_type].sel(plevel=plevel_gph), tps=sel_tps)

    mean_u, pvalues_ttest_u = sut.ttest_field(
        this_comp_ts_u, data_cut_wind[u_var_type].sel(plevel=plevel_gph))
    mask_u = sut.field_significance_mask(
        pvalues_ttest_u, alpha=0.05, corr_type=None)
    mean_v, pvalues_ttest_v = sut.ttest_field(
        this_comp_ts_v, data_cut_wind[v_var_type].sel(plevel=plevel_gph))
    mask_v = sut.field_significance_mask(
        pvalues_ttest_v, alpha=0.05, corr_type=None)

    dict_w = cplt.plot_wind_field(ax=im['ax'][idx*ncols],
                                  u=xr.where(mask_u, mean_u, np.nan),
                                  v=xr.where(mask_u, mean_v, np.nan),
                                  #   u=mean_u,
                                  #   v=mean_v,
                                  scale=70,
                                  steps=2,
                                  key_length=2,
                                  )

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_wind[u_var_type].sel(plevel=plevel_u), tps=sel_tps_step)
    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_wind[u_var_type].sel(plevel=plevel_u))
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_u = cplt.plot_map(mean,
                         ax=im['ax'][idx*ncols + 1],
                         cmap='PuOr',
                         title='Rossby Wave Response u' if idx == 0 else None,
                         plot_type='contourf',
                         levels=9,
                         vmin=vmin_uwind, vmax=vmax_uwind,
                         projection='PlateCarree',
                         plt_grid=True,
                         orientation='horizontal',
                         significance_mask=mask,
                         extend='both',
                         bar=True if idx == 2 else None,
                         sci=None,
                         label=label_uwind,
                         round_dec=2,
                         tick_step=3
                         )

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_wind[v_var_type].sel(plevel=plevel_v), tps=sel_tps_step)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_wind[v_var_type].sel(plevel=plevel_v))
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_v = cplt.plot_map(mean,
                         ax=im['ax'][idx*ncols + 2],
                         title='Rossby Wave Response v' if idx == 0 else None,
                         cmap='PuOr',
                         plot_type='contourf',
                         levels=9,
                         vmin=vmin_vwind, vmax=vmax_vwind,
                         projection='PlateCarree',
                         plt_grid=True,
                         orientation='horizontal',
                         significance_mask=mask,
                         extend='both',
                         bar=True if idx == 2 else None,
                         sci=None,
                         label=label_vwind,
                         round_dec=2,
                         tick_step=3
                         )


# cplt.add_colorbar(im=im_gph, fig=im['fig'],
#                   sci=None,
#                   label=label_gph,
#                   x_pos=0.12,
#                   width=0.2,
#                   height=0.02,
#                   y_pos=0.08,
#                   tick_step=2)

savepath = plot_dir +\
    f"{output_folder}/paper_plots/Kelvin_Rossby_Waves.png"
cplt.save_fig(savepath, fig=im['fig'])

# %%
# Compute RV
rv_an = ds_wind.compute_vorticity()
rv_an = ds_wind.ds['vorticity_an_JJAS'].rename('rv_an_JJAS')
# %%
# Compute gradient vertical velocity
grad_w, grad_w_an = ds_wind.compute_vertical_velocity_gradient(dp='lat')
# Compute vertical shear
vertical_shear = ds_wind.compute_vertical_shear()

# %%
# Plot together Northward propagation in terms of Vertical Shear and Vorticity
reload(cplt)
reload(sut)
reload(tu)

lon_range_c = [40, 160]
lat_range_c = [-30, 40]
plevel_vorticity = 500
plevel_wgrad = 500

an_type = 'JJAS'
label_shear = rf'Vertical Shear (u200-u850) anomalies [m/s]'
label_wgrad = rf'{plevel_vorticity }-hPa dw/dy an. [Pa/ms]'
label_vorticity = rf'{plevel_vorticity }-hPa RV anomalies [1/s]'

var_type_shear = f'vertical_shear_an_{an_type}'
var_type_shear = f'vertical_shear'
var_type_vorticity = f'rv_an_{an_type}'
var_type_gradw = f'w_grad_lat_an_{an_type}'

vmax_shear = 50
vmin_shear = -vmax_shear
vmax_wgrad = 10e-3
vmin_wgrad = -vmax_wgrad
vmax_vorticity = 10e-6
vmin_vorticity = -vmax_vorticity


data_cut_shear = sput.cut_map(ds=vertical_shear[var_type_shear],
                              lon_range=lon_range_c,
                              lat_range=lat_range_c,
                              dateline=False)

data_cut_wgrad = sput.cut_map(ds=grad_w_an.sel(plevel=plevel_wgrad),
                              lon_range=lon_range_c,
                              lat_range=lat_range_c,
                              dateline=False)

data_cut_vorticity = sput.cut_map(ds=rv_an.sel(plevel=plevel_vorticity),
                                  lon_range=lon_range_c,
                                  lat_range=lat_range_c,
                                  dateline=False)

ncols = 3
im = cplt.create_multi_plot(nrows=3, ncols=ncols,
                            # hspace=0.4,
                            wspace=0.2,
                            projection='PlateCarree',
                            central_longitude=180,
                            # figsize=(14,14)
                            # end_idx=9
                            )

plot_order = ['Canonical',
              'Eastward Blocked',
              'Stationary']
step = 10
for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    sel_tps_step = tu.get_periods_tps(tps=sel_tps,
                                      start=4,
                                      step=step)

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_shear, tps=sel_tps_step)
    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_shear)
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_shear = cplt.plot_map(mean,
                             ax=im['ax'][idx*ncols + 0],
                             vertical_title=f'{group}',
                             title=r'Vertical Shear' if idx == 0 else None,
                             cmap='RdGy_r',
                             plot_type='contourf',
                             levels=12,
                             vmin=vmin_shear, vmax=vmax_shear,
                             plt_grid=True,
                             orientation='horizontal',
                             significance_mask=mask,
                             extend='both',
                             sci=None,
                             label=label_shear if idx == 2 else None,
                             round_dec=2,
                             tick_step=3
                             )

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_wgrad, tps=sel_tps_step)
    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_wgrad)
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_dw = cplt.plot_map(mean,
                          ax=im['ax'][idx*ncols + 1],
                          cmap='PiYG_r',
                          title='dw/dy' if idx == 0 else None,
                          plot_type='contourf',
                          levels=9,
                          vmin=vmin_wgrad, vmax=vmax_wgrad,
                          plt_grid=True,
                          orientation='horizontal',
                          significance_mask=mask,
                          extend='both',
                          sci=-3,
                          label=label_wgrad if idx == 2 else None,
                          round_dec=2,
                          tick_step=3
                          )

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_vorticity, tps=sel_tps_step)
    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_vorticity)
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_rv = cplt.plot_map(mean,
                          ax=im['ax'][idx*ncols + 2],
                          cmap='BrBG',
                          title=r'Relative Vorticity' if idx == 0 else None,
                          plot_type='contourf',
                          levels=9,
                          vmin=vmin_vorticity, vmax=vmax_vorticity,
                          plt_grid=True,
                          orientation='horizontal',
                          significance_mask=mask,
                          extend='both',
                          sci=-6,
                          label=label_vorticity if idx == 2 else None,
                          round_dec=2,
                          tick_step=3
                          )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/Northward_propagation_ISO.png"
cplt.save_fig(savepath, fig=im['fig'])


# %%
# Kelvin wave response
reload(cplt)
an_type = 'JJAS'
var_type = f'z_an_{an_type}'
plevel = 850
label = rf'Anomalies GPH (wrt {an_type}) ({plevel} hPa) [m]'
n = len(k_means_tps_lon_lat)

vmax = 2e1
vmin = -vmax
lon_range_c = [40, 180]
lat_range_c = [-30, 50]
sci = None

u_var_type = f'u_an_{an_type}'
v_var_type = f'v_an_{an_type}'
wind_cut = sput.cut_map(ds=ds_wind.ds[[u_var_type, v_var_type]].sel(plevel=plevel),
                        lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=False)
wind_cut = tu.get_month_range_data(
    wind_cut, start_month='Jun', end_month='Sep')

data_cut = sput.cut_map(ds=ds_mse.ds[var_type].sel(plevel=plevel),
                        lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=False)

im = cplt.create_multi_plot(nrows=1, ncols=3,
                            # hspace=0.4,
                            wspace=0.2,
                            projection='PlateCarree',
                            central_longitude=180,
                            end_idx=n)

for idx, (group, sel_tps) in enumerate(k_means_tps_lon_lat.items()):
    sel_tps = k_means_tps_lon_lat[group]

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut, tps=sel_tps)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut)
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_comp = cplt.plot_map(mean,
                            ax=im['ax'][idx],
                            title=f'{group}',
                            cmap='PuOr',
                            plot_type='contourf',
                            levels=9,
                            vmin=vmin, vmax=vmax,
                            projection='PlateCarree',
                            bar=False,
                            plt_grid=True,
                            orientation='horizontal',
                            significance_mask=mask,
                            extend='both'
                            )

    this_comp_ts_u = tu.get_sel_tps_ds(
        wind_cut[u_var_type], tps=sel_tps)
    this_comp_ts_v = tu.get_sel_tps_ds(
        wind_cut[v_var_type], tps=sel_tps)

    mean_u, pvalues_ttest_u = sut.ttest_field(
        this_comp_ts_u, wind_cut[u_var_type])
    mask_u = sut.field_significance_mask(
        pvalues_ttest_u, alpha=0.05, corr_type=None)
    mean_v, pvalues_ttest_v = sut.ttest_field(
        this_comp_ts_v, wind_cut[v_var_type])
    mask_v = sut.field_significance_mask(
        pvalues_ttest_v, alpha=0.05, corr_type=None)

    dict_w = cplt.plot_wind_field(ax=im['ax'][idx],
                                  u=xr.where(mask_u, mean_u, np.nan),
                                  v=xr.where(mask_u, mean_v, np.nan),
                                  #   u=mean_u,
                                  #   v=mean_v,
                                  lw=60,
                                  scale=70,
                                  steps=2,
                                  key_length=5,
                                  )

    loc_lons = this_dict['data'].lon
    loc_lats = this_dict['data'].lat

    locs = gut.zip_2_lists(loc_lons, loc_lats)

    loc_map = ds_wind.get_map_for_locs(locations=locs)
    loc_map = sput.cut_map(loc_map, lon_range=lon_range_c,
                           lat_range=lat_range_c,
                           dateline=True)
    cplt.plot_map(loc_map,
                  ax=im['ax'][idx],
                  plot_type='contour',
                  color='magenta',
                  levels=1,
                  #   vmin=-5, vmax=5,
                  bar=False,
                  plt_grid=True,
                  lw=2,)


cplt.add_colorbar(im=im_comp, fig=im['fig'],
                  sci=None,
                  label=label,
                  x_pos=0.2,
                  y_pos=0.02,
                  height=0.07,
                  width=0.6,
                  round_dec=2,
                  tick_step=3
                  )
savepath = plot_dir +\
    f"{output_folder}/propagation/gph_{an_type}_{plevel}_groups.png"
cplt.save_fig(savepath, fig=im['fig'])

# %%
# Rossby Wave response
reload(cplt)
an_type = 'JJAS'
lon_range_c = [40, 180]
lat_range_c = [-30, 50]
# u_var_type = f'u'
# v_var_type = f'v'
# u_var_type = f'u_chi_an_JJAS'
# v_var_type = f'v_chi_an_JJAS'
u_var_type = f'u_an_{an_type}'
v_var_type = f'v_an_{an_type}'
var_type = v_var_type

wind_levels = [900, 500, 200]
# wind_levels = [900]

step = 0

for idx, plevel in enumerate(wind_levels):
    label = rf'{plevel}-hPa u-winds anomalies (wrt {an_type})[m/s]'
    label = rf'{plevel}-hPa v winds anomalies (wrt {an_type})[m/s]'
    vmax = 2.5 + 1*idx
    vmin = -vmax

    data_cut = sput.cut_map(ds=ds_wind.ds[[u_var_type, v_var_type]].sel(plevel=plevel),
                            lon_range=lon_range_c,
                            lat_range=lat_range_c,
                            dateline=False)
    data_cut = tu.get_month_range_data(
        data_cut, start_month=start_month,
        end_month=end_month)
    im = cplt.create_multi_plot(nrows=1, ncols=3,
                                # hspace=0.4,
                                wspace=0.2,
                                projection='PlateCarree',
                                central_longitude=180,
                                end_idx=n)

    for idx, (group, sel_tps) in enumerate(k_means_tps_lon_lat.items()):
        sel_tps = k_means_tps_lon_lat[group]
        sel_tps_step = tu.add_time_step_tps(tps=sel_tps,
                                            time_step=step)
        this_comp_ts = tu.get_sel_tps_ds(
            data_cut[var_type], tps=sel_tps_step)

        mean, pvalues_ttest = sut.ttest_field(
            this_comp_ts, data_cut[var_type])
        mask = sut.field_significance_mask(
            pvalues_ttest, alpha=0.05, corr_type=None)

        im_comp = cplt.plot_map(mean,
                                ax=im['ax'][idx],
                                title=f'{group} (Day {step})',
                                cmap='PuOr',
                                plot_type='contourf',
                                levels=9,
                                vmin=vmin, vmax=vmax,
                                projection='PlateCarree',
                                bar=False,
                                plt_grid=True,
                                orientation='horizontal',
                                significance_mask=mask,
                                extend='both'
                                )

        loc_lons = this_dict['data'].lon
        loc_lats = this_dict['data'].lat

        locs = gut.zip_2_lists(loc_lons, loc_lats)

        loc_map = ds_wind.get_map_for_locs(locations=locs)
        loc_map = sput.cut_map(loc_map, lon_range=lon_range_c,
                               lat_range=lat_range_c,
                               dateline=True)
        cplt.plot_map(loc_map,
                      ax=im['ax'][idx],
                      plot_type='contour',
                      color='magenta',
                      levels=1,
                      #   vmin=-5, vmax=5,
                      bar=False,
                      plt_grid=True,
                      lw=2,)

    cplt.add_colorbar(im=im_comp, fig=im['fig'],
                      sci=None,
                      label=label,
                      x_pos=0.2,
                      y_pos=0.02,
                      height=0.07,
                      width=0.6,
                      round_dec=2,
                      tick_step=3
                      )
    savepath = plot_dir +\
        f"{output_folder}/propagation/{var_type}_{plevel}_day_{step}_groups.png"
    cplt.save_fig(savepath, fig=im['fig'])

# %%
# Define percentages of Early Warning Signals
region = 'SA'
prop_type = 'Canonical'
tps_early_warning = k_means_tps_lon_lat[prop_type]
# EE TS
evs = loc_dict[region]['data'].evs
t_all2 = tsa.get_ee_ts(evs=evs)


t_jjas = tu.get_month_range_data(t_all2, start_month='Jun',
                                 end_month='Sep')
# tps = tps_dict['peaks']
tps_eres = gut.get_quantile_of_ts(t_jjas, q=0.9)

common_tps = tu.get_common_tps(tps_early_warning.time,
                               tps_eres.time,
                               offset=12, delay=20)

print(len(common_tps)/len(tps_early_warning))
print(common_tps)

# %%
# %%
# Plot OLR background progression
reload(tu)
reload(cplt)
lon_range_c = [30, 180]
lat_range_c = [-30, 55]

an_type = 'JJAS'
var_type = f'an_{an_type}'
label = f'OLR Anomalies (wrt {an_type})'

data_cut = sput.cut_map(ds=ds_olr.ds[var_type],
                        lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=False)
data_cut = tu.get_month_range_data(
    data_cut, start_month=start_month,
    end_month=end_month)


vmax = 6e4
vmin = -vmax
step = 5

plot_order = ['Canonical',
              'Eastward Blocked',
              'Stationary']
for i, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]

    sci = 3
    composite_arrs = tu.get_day_progression_arr(ds=data_cut,
                                                tps=sel_tps,
                                                average_ts=False,
                                                start=0,
                                                end=25,
                                                step=step,
                                                var=None)
    ncols = 3
    nrows = int(np.ceil(len(composite_arrs.day)/ncols))

    im = cplt.create_multi_plot(nrows=nrows,
                                ncols=ncols,
                                # figsize=(14, 6),
                                title=f'{group}',
                                y_title=1.1,
                                projection='PlateCarree',
                                orientation='horizontal',
                                # hspace=0.25,
                                central_longitude=180,
                                wspace=0.25,
                                end_idx=len(composite_arrs.day)
                                )
    for idx, (day) in enumerate(composite_arrs.day):
        mean_ts = composite_arrs.sel(day=day)
        im_comp = cplt.plot_map(mean_ts[var_type],
                                ax=im['ax'][idx],
                                title=f'Day {int(day)}',
                                plot_type='contourf',
                                cmap='coolwarm',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                bar=False,
                                extend='both',
                                plt_grid=True,
                                intpol=False,
                                tick_step=2,
                                round_dec=2)

    cplt.add_colorbar(im=im_comp,
                      fig=im['fig'],
                      x_pos=0.2,
                      y_pos=0.08, width=0.6, height=0.02,
                      orientation='horizontal',
                      label=label,
                      tick_step=2,
                      sci=sci
                      )

    savepath = plot_dir +\
        f"{output_folder}/propagation/{group}_olr_propagation_{step}.png"
    cplt.save_fig(savepath, fig=im['fig'])

# %%
# OLR propagation in 1 plot
reload(tu)
reload(cplt)
step = 1
max_steps = 35
lon_range_c = [50, 170]
lat_range_c = [-20, 45]

an_type = 'JJAS'
var_type = f'an_{an_type}'
label = f'day'

data_cut = sput.cut_map(ds=ds_olr.ds[var_type],
                        lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=False)
data_cut = tu.get_month_range_data(
    data_cut, start_month=start_month,
    end_month=end_month)


vmax = max_steps
vmin = 0
sci = None

ncols = 3
im = cplt.create_multi_plot(nrows=1,
                            ncols=ncols,
                            # figsize=(14, 6),
                            projection='PlateCarree',
                            orientation='horizontal',
                            # hspace=0.25,
                            central_longitude=180,
                            wspace=0.15,
                            end_idx=3
                            )
q_th = 0.05
plot_order = ['Canonical',
              'Eastward Blocked',
              'Stationary']
for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]

    day_arr = tu.get_quantile_progression_arr(ds=data_cut,
                                              tps=sel_tps,
                                              average_ts=False,
                                              start=0,
                                              end=max_steps,
                                              step=step,
                                              var=None,
                                              q_th=q_th,
                                              th=-6.9e4)
    im_comp = cplt.plot_map(day_arr[var_type],
                            ax=im['ax'][idx],
                            plot_type='colormesh',
                            cmap='rainbow',
                            vmin=vmin, vmax=vmax,
                            bar=True,
                            title=f'{group}',
                            # levels=int(max_steps/step + 1),
                            extend='max',
                            plt_grid=True,
                            label='Day',
                            tick_step=2,
                            round_dec=0)

savepath = plot_dir +\
    f"{output_folder}/paper_plots/olr_propagation_{step}_all.png"
cplt.save_fig(savepath, fig=im['fig'])
