#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for producing propagation plots of extreme rainfall events
@author: Felix Strnad
"""
# %%
import geoutils.utils.file_utils as fut
import geoutils.tsa.propagation as prop
import matplotlib.pyplot as plt
import geoutils.geodata.moist_static_energy as mse
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
grid_type = "fekete"
grid_step = 1

scale = "south_asia"

output_folder = "bsiso"


output_dir = "/home/strnad/data/climnet/outputs/"
plot_dir = "/home/strnad/data/plots/"
data_dir = "/home/strnad/data/"

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
dataset_file = data_dir + \
    f"/climate_data/2.5/era5_sst_{2.5}_ds.nc"

ds_sst = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         detrend=True,
                         an_types=['JJAS', 'month', 'dayofyear'],
                         month_range=['Jun', 'Sep']
                         )

# %%
# Load Wind-Field data
reload(wds)
nc_files_u = []
nc_files_v = []
nc_files_w = []

plevels = [100, 200, 300,
           400, 500, 600,
           700, 800,
           850, 900, 1000]

for plevel in plevels:
    dataset_file_u = data_dir + \
        f"/climate_data/2.5/era5_u_{2.5}_{plevel}_ds.nc"
    nc_files_u.append(dataset_file_u)
    dataset_file_v = data_dir + \
        f"/climate_data/2.5/era5_v_{2.5}_{plevel}_ds.nc"
    nc_files_v.append(dataset_file_v)
    dataset_file_w = data_dir + \
        f"/climate_data/2.5/era5_w_{2.5}_{plevel}_ds.nc"
    nc_files_w.append(dataset_file_w)

ds_wind = wds.Wind_Dataset(data_nc_u=nc_files_u,
                           data_nc_v=nc_files_v,
                           data_nc_w=nc_files_w,
                           plevels=plevels,
                           can=True,
                           an_types=['JJAS'],
                           month_range=['Jun', 'Sep'],
                           init_mask=False,
                           )
# %%
ds_wind.compute_massstreamfunction()

# %%
# Moist static energy analysis
reload(mse)

nc_files_q = []
nc_files_t = []
nc_files_z = []

plevels = [
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    850,
    900,
    1000,
]


for plevel in plevels:
    dataset_file_q = data_dir + \
        f"/climate_data/2.5/era5_q_{2.5}_{plevel}_ds.nc"
    nc_files_q.append(dataset_file_q)
    dataset_file_t = data_dir + \
        f"/climate_data/2.5/era5_t_{2.5}_{plevel}_ds.nc"
    nc_files_t.append(dataset_file_t)
    dataset_file_z = data_dir + \
        f"/climate_data/2.5/era5_z_{2.5}_{plevel}_ds.nc"
    nc_files_z.append(dataset_file_z)

ds_mse = mse.MoistStaticEnergy(data_nc_q=nc_files_q,
                               data_nc_t=nc_files_t,
                               data_nc_z=nc_files_z,
                               plevels=plevels,
                               can=True,
                               an_types=['JJAS'],
                               month_range=['Jun', 'Sep']
                               )


# %%
# OLR 2.5 regridded
dataset_file = data_dir + \
    f"/climate_data/2.5/era5_ttr_{2.5}_ds.nc"  # 1 resolution also required for clustering

ds_olr_25 = bds.BaseDataset(data_nc=dataset_file,
                            can=True,
                            an_types=['dayofyear', 'JJAS'],
                            detrend=True,
                            month_range=['Jun', 'Sep']
                            )
# %%
# OLR data
reload(bds)
dataset_file = data_dir + \
    f"/climate_data/1/era5_ttr_{1}_ds.nc"  # 1 resolution also required for clustering

ds_olr = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         an_types=['dayofyear', 'JJAS'],
                         detrend=True
                         )
# %%
# Define time points
reload(tu)
reload(fut)
B_max = 6
cd_folder = 'graph_tool'
savepath_loc = (
    plot_dir +
    f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_prob_maps.npy"
)

loc_dict = fut.load_np_dict(savepath_loc)
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
# Create Hovmöller lon lat
reload(prop)
an_type = 'dayofyear'
var_type = f'an_{an_type}'
dist_days = 15
rm_tps = tu.remove_consecutive_tps(tps=tps, steps=dist_days)

# lat_range_med = [-5, 5]
# lon_range_med = [50, 160]

lat_range_med = [-5, 5]
lon_range_med = [50, 180]

lat_range_zon = [-10, 22]
lon_range_zon = [70, 80]

# for plotting
hov_data_lon = prop.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                              tps=rm_tps,
                                              lat_range=lat_range_med,
                                              lon_range=lon_range_med,
                                              num_days=30,
                                              start=5,
                                              gf=(2, 2),
                                              var=var_type,
                                              zonal=True, )

hov_data_lat = prop.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                              tps=rm_tps,
                                              lat_range=lat_range_zon,
                                              lon_range=lon_range_zon,
                                              num_days=30,
                                              start=5,
                                              gf=(2, 2),
                                              var=var_type,
                                              zonal=False)

# %%
# Create lon and lat hovmöller diagrams that are used for clustering
reload(tu)
reload(prop)
an_type = 'dayofyear'
var_type = f'an_{an_type}'

lat_range_med = [-5, 5]
lon_range_med = [60, 155]

lat_range_zon = [-10, 22]
lon_range_zon = [70, 80]

# Hovmöller along longitudes
hov_data_lon_kmeans = prop.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                                     tps=rm_tps,
                                                     lat_range=lat_range_med,
                                                     lon_range=lon_range_med,
                                                     num_days=30,
                                                     start=5,
                                                     gf=(5, 5),
                                                     var=var_type,
                                                     zonal=True)

hov_data_lat_kmeans = prop.get_hovmoeller_single_tps(ds=ds_olr.ds,
                                                     tps=rm_tps,
                                                     lat_range=lat_range_zon,
                                                     lon_range=lon_range_zon,
                                                     num_days=30,
                                                     start=5,
                                                     gf=(5, 5),
                                                     var=var_type,
                                                     zonal=False)
# %%
# Apply the clustering algorithm
reload(tcl)
n = 3
an_type = 'dayofyear'
var_type = f'an_{an_type}'

type_names = [
    'Canonical',
    'Eastward Blocked',
    'Quasi-stationary',
]

k_means_tps_lon_lat = tcl.tps_cluster_2d_data([hov_data_lon_kmeans,
                                               hov_data_lat_kmeans],
                                              tps=rm_tps,
                                              minibatch=False,
                                              method='kmeans',
                                              n_clusters=n,
                                              random_state=0,
                                              n_init=100,
                                              max_iter=3000,
                                              cluster_names=type_names,
                                              rm_ol=True,
                                              sc_th=0.05
                                              #   metric='correlation'
                                              #   metric='euclidean',
                                              #   plot_statistics=True,
                                              )
n = 3
savepath = f"{plot_dir}/{output_folder}/propagation/olr_{1}_hovmoeller_{n}_cluster.npy"

# %%
# Plot for paper OLR Hovmöller + Maritime Continent
reload(cplt)
reload(sut)
lon_range_c = [-70, 35]
lat_range_c = [-75, 75]

lon_ticks = [60, 80, 100, 120, 140, 160, 179]
lon_ticklabels = ['60°E', '80°E', '100°E',
                  '120°E', '140°E', '160°E', '180°E']
lat_ticks = [-10, 0, 10, 20, 30]
lat_ticklabels = ['10°S', '0°', '10°N', '20°N', '30°N']

# For Hovmöller Diagrams
vtimes = hov_data_lon.day
lons = hov_data_lon.lon.values
lats = hov_data_lat.lat.values
vmax_olr = 20
vmin_olr = -vmax_olr

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
label_hov = r'OLR (wrt JJAS)-Anomalies [$Wm^{-2}$]'

ncols = 2
nrows = 3
im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            figsize=(13, 15),
                            orientation='horizontal',
                            hspace=0.25,
                            wspace=0.2,
                            )

axs = im['ax']
fig = im['fig']
plot_order = ['Canonical',
              'Eastward Blocked',
              'Quasi-stationary']

for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    k_data = tu.get_sel_tps_ds(
        ds=hov_data_lon, tps=sel_tps).mean(dim='time')
    h_im = cplt.plot_2D(x=lons, y=vtimes,
                        ax=axs[idx*ncols],
                        z=k_data,
                        levels=9,
                        vertical_title=f'{group} ({len(sel_tps)} samples)',
                        title='Zonal Hovmöller diagrams' if idx == 0 else None,
                        x_title_offset=-0.3,
                        orientation='vertical',
                        cmap='bwr',
                        centercolor='white',
                        plot_type='contourf',
                        vmin=vmin_olr, vmax=vmax_olr,
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

    if idx == 0:
        cplt.plot_arrow(ax=axs[idx*ncols],
                        x1=60, y1=-3,
                        x2=145, y2=17,
                        lw=3,
                        label=rf'5.4 m$s^{{-1}}$')

    if idx == 1:
        cplt.plot_arrow(ax=axs[idx*ncols],
                        x1=60, y1=-3,
                        x2=125, y2=18,
                        lw=3,
                        label=rf'4 m$s^{{-1}}$')

    k_data = tu.get_sel_tps_ds(
        ds=hov_data_lat, tps=sel_tps).mean(dim='time')
    h_im = cplt.plot_2D(x=lats, y=vtimes,
                        z=k_data,
                        ax=axs[idx*ncols + 1],
                        levels=10,
                        title='Meridional Hovmöller diagrams' if idx == 0 else None,
                        orientation='vertical',
                        cmap='bwr',
                        centercolor='white',
                        plot_type='contourf',
                        vmin=vmin_olr,
                        vmax=vmax_olr,
                        xlabel='Latitude [degree]' if idx % nrows == 2 else None,
                        ylabel='day',
                        bar=False,
                        extend='both',
                        xticks=lat_ticks,
                        xticklabels=lat_ticklabels,
                        )

    if idx == 0:
        cplt.plot_arrow(ax=axs[idx*ncols+1],
                        x1=0, y1=0,
                        x2=20, y2=15,
                        lw=3,
                        label=rf'1.7 m$s^{{-1}}$',)

    if idx == 1:
        cplt.plot_arrow(ax=axs[idx*ncols+1],
                        x1=0, y1=0,
                        x2=20, y2=18,
                        lw=3,
                        label=rf'1.4 m$s^{{-1}}$')


cplt.add_colorbar(im=h_im,
                  fig=fig,
                  label=label_hov,
                  x_pos=0.2,
                  width=0.6,
                  height=0.02,
                  y_pos=-0.01,
                  tick_step=2)
cplt.set_legend(ax=im['ax'][-2],
                fig=im['fig'],
                # order=order,
                loc='outside',
                ncol_legend=3,
                box_loc=(0.2, 0.06)
                )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/propagation_olr_hovmoeller_k_means_lon_lat_{region}.pdf"
cplt.save_fig(savepath, fig=fig)

# %%
# SSTs with MSE background and conditional probabilities
reload(cplt)
an_type = 'dayofyear'
var_type = f'an_{an_type}'

lon_range_c = [40, 180]
lat_range_c = [-30, 50]

vmax_tcw = 5
vmin_tcw = -vmax_tcw
# vmax_tcw=70
# vmin_tcw=30
an_type = 'JJAS'
var_type_kelvin = f'vi_mse_an_{an_type}'
# var_type_kelvin = f'vi_mse'
vmax_mse = 1e5
vmin_mse = -vmax_mse
label_mse = rf'MSE (wrt JJAS)-anomalies [J/m$^2$]'
plevel_mse = 1000
data_cut_mse = sput.cut_map(ds=ds_mse.ds[var_type_kelvin].sel(lev=plevel_mse),
                            lon_range=lon_range_c,
                            lat_range=lat_range_c,
                            dateline=False)

vmax_sst = 1
vmin_sst = -vmax_sst
label_sst = r'SST (wrt JJAS)-anomalies [K]'

proj = cplt.get_projection(projection='PlateCarree',
                           central_longitude=0,
                           )
proj_180 = cplt.get_projection(projection='PlateCarree',
                               central_longitude=180,
                               )
im = cplt.create_plot(figsize=(14, 12))
fig = im['fig']
ax1 = fig.add_axes([0., .75, .3, .2], projection=proj_180)
ax2 = fig.add_axes([0.0, .5, .3, .2], projection=proj_180)
ax3 = fig.add_axes([0.0, 0.25, .3, .2], projection=proj_180)
ax4 = fig.add_axes([0.5, .75, .3, .2], projection=proj)
ax5 = fig.add_axes([0.5, .5, .3, .2], projection=proj)
ax6 = fig.add_axes([0.5, 0.25, .3, .2], projection=proj)
# ax7 = fig.add_axes([1, 0.55, .3, 0.2])

axs = [ax1, ax4,
       ax2, ax5,
       ax3, ax6,
       #    ax7
       ]
cplt.enumerate_subplots(axs=axs)


plot_order = ['Canonical',
              'Eastward Blocked',
              'Quasi-stationary']
for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    sel_tps_step = tu.get_periods_tps(tps=sel_tps,
                                      start=-7,
                                      end=-5)
    this_comp_ts = tu.get_sel_tps_ds(
        ds_sst.ds, tps=sel_tps_step)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts[var_type], ds_sst.ds[var_type])
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.02, corr_type=None)

    # SSTs
    im_sst = cplt.plot_map(mean,
                           ax=axs[idx*2 + 0],
                           cmap='RdBu_r',
                           centercolor='white',
                           plot_type='contourf',
                           levels=14,
                           title='Background SST' if idx == 0 else None,
                           vertical_title=f'{group}',
                           vmin=vmin_sst, vmax=vmax_sst,
                           plt_grid=True,
                           extend='both',
                           orientation='horizontal',
                           significance_mask=mask,
                           hatch_type='..',
                           lon_range=[-70, 35],
                           lat_range=[-75, 75],
                           gs_lon=60,
                           gs_lat=30,
                           )

    # MSE
    sel_tps_step = tu.get_periods_tps(tps=sel_tps,
                                      start=-30,
                                      end=-20)
    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_mse, tps=sel_tps_step)
    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_mse)
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.1, corr_type=None)

    im_u = cplt.plot_map(mean,
                         ax=axs[idx*2 + 1],
                         cmap='coolwarm_r',
                         title='Background MSE' if idx == 0 else None,
                         plot_type='contourf',
                         centercolor='white',
                         levels=14,
                         vmin=vmin_mse, vmax=vmax_mse,
                         orientation='horizontal',
                         significance_mask=mask,
                         extend='both',
                         plt_grid=True,
                         sci=None,
                         tick_step=3,
                         gs_lon=30,
                         gs_lat=20,
                         )

    # plots an arrow in the first subplot in the center of the map
    if idx == 0:
        cplt.plot_arrow(ax=axs[idx],
                        x1=100, y1=0,
                        x2=120, y2=0,
                        color='k',
                        fill_color=False,
                        lw=2,
                        width=2.5,
                        )
        cplt.plot_arrow(ax=axs[idx],
                        x1=80, y1=0,
                        x2=80, y2=15,
                        color='k',
                        fill_color=False,
                        lw=2,
                        width=2.5,
                        )

    elif idx == 1:
        cplt.plot_arrow(ax=axs[idx*2],
                        x1=80, y1=0,
                        x2=100, y2=0,
                        color='k',
                        fill_color=False,
                        lw=2,
                        width=2.5,
                        )
        cplt.plot_arrow(ax=axs[idx*2],
                        x1=80, y1=0,
                        x2=80, y2=15,
                        color='k',
                        fill_color=False,
                        lw=2,
                        width=2.5,
                        )

    cplt.plot_vline(ax=axs[idx*2],
                    x=110,
                    lw=2,
                    trafo_axis=True,)
    cplt.plot_vline(ax=axs[idx*2],
                    x=130,
                    lw=2,
                    trafo_axis=True,
                    label='Maritime Continent Barrier',)
    cplt.plot_vline(ax=axs[idx*2+1],
                    x=110,
                    lw=2,
                    trafo_axis=True,)
    cplt.plot_vline(ax=axs[idx*2+1],
                    x=130,
                    lw=2,
                    trafo_axis=True,
                    label='Maritime Continent Barrier',)

# SST colorbar
cplt.add_colorbar(im=im_sst, fig=im['fig'],
                  sci=None,
                  label=label_sst,
                  x_pos=0.35,
                  width=0.015,
                  height=0.6,
                  y_pos=0.3,
                  tick_step=2,
                  orientation='vertical'
                  )
# MSE colorbar
cplt.add_colorbar(im=im_u, fig=im['fig'],
                  label=label_mse,
                  x_pos=0.85,
                  width=0.015,
                  height=0.6,
                  y_pos=0.3,
                  tick_step=2,
                  sci=5,
                  orientation='vertical'
                  )

cplt.set_legend(ax=im_u['ax'],
                fig=fig,
                # order=order,
                loc='outside',
                ncol_legend=1,
                box_loc=(0.03, 0.23)
                )
cplt.set_legend(ax=im_sst['ax'],
                fig=fig,
                # order=order,
                loc='outside',
                ncol_legend=1,
                box_loc=(0.53, 0.23)
                )
savepath = plot_dir +\
    f"{output_folder}/paper_plots/sst_mse_background_all.pdf"
cplt.save_fig(savepath, fig=im['fig'])


# %%
# Plot for paper multiple pressure levels wind fields
# Vertical Cuts Walker Circulation
reload(cplt)
reload(cplt)
reload(sut)

an_type = 'JJAS'
lon_range_c = [-70, 35]
lat_range_c = [-75, 75]

var_type_wind_u = 'msf_u_an_JJAS'
var_type_wind_v = 'msf_v_an_JJAS'
var_type_map_u = 'u_chi_an_JJAS'
var_type_map_v = 'v_chi_an_JJAS'
var_type_map_u = 'msf_u_an_JJAS'
var_type_map_v = 'msf_v_an_JJAS'

# For MSF Field
plevel_range = [300, 400, 500]
data_cut_hd = sput.cut_map(ds=ds_wind.ds.sel(lev=plevel_range)[
    [var_type_map_u, var_type_map_v]],
    lon_range=lon_range_c,
    lat_range=lat_range_c,
    dateline=True)
data_cut_hd = data_cut_hd.mean(dim='lev')

# Range data for wind fields
s_lat_range = [-10, 30]
s_lon_range = [70, 80]
c_lat_range = [-0, 10]
c_lon_range = [50, -80]

u_var_type = f'U_an_{an_type}'
v_var_type = f'V_an_{an_type}'
w_var_type = f'OMEGA_an_{an_type}'
wind_data_lat = sput.cut_map(ds_wind.ds[[u_var_type,
                                         v_var_type,
                                         w_var_type]],
                             lon_range=s_lon_range,
                             lat_range=s_lat_range, dateline=False)
wind_data_lon = sput.cut_map(ds_wind.ds[[u_var_type,
                                        v_var_type,
                                        w_var_type]],
                             lon_range=c_lon_range,
                             lat_range=c_lat_range, dateline=True)

range_data_lon = sput.cut_map(ds_wind.ds[[var_type_wind_u,
                                          var_type_wind_v]],
                              lon_range=c_lon_range,
                              lat_range=c_lat_range, dateline=True)
range_data_lat = sput.cut_map(ds_wind.ds, lon_range=s_lon_range,
                              lat_range=s_lat_range, dateline=False)

# %%
# Stream function background plot
reload(cplt)

an_type = 'JJAS'
var_type = f'an_{an_type}'
label_msf_u = r'$\Psi_u$ JJAS-anomalies [kg$m^{-2}s^{-1}$]'
label_msf_v = r'$\Psi_v$ JJAS-anomalies [kg$m^{-2}s^{-1}$]'
label_vert_u = r'$\bar{\Psi}_u$  JJAS-anomalies [kg$m^{-2}s^{-1}$]'
label_vert_v = r'$\bar{\Psi}_v$  JJAS-anomalies [kg$m^{-2}s^{-1}$]'

vmax_vert = 1e11
vmin_vert = -vmax_vert
vmax_map = 1e11
vmin_map = -vmax_map

lon_ticks = [60, 120, 180, 240]
lon_ticklabels = ['60°E', '120°E', '180°', '120°W']
lat_ticks = [-10, 0, 10, 20, 30]
lat_ticklabels = ['10°S', '0°', '10°N', '20°N', '30°N']

sci = 11
sci_map = 11
n = len(k_means_tps_lon_lat)
ncols = 4
proj = cplt.get_projection(projection='PlateCarree',
                           central_longitude=180)
fig = plt.figure(
    figsize=(19, 15)
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
              'Quasi-stationary']

for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    k_data_lon = tu.get_sel_tps_ds(
        ds=range_data_lon[var_type_wind_u], tps=sel_tps).mean(dim=['lat', 'time'])
    k_data_lat = tu.get_sel_tps_ds(
        ds=range_data_lat[var_type_wind_v], tps=sel_tps).mean(dim=['lon', 'time'])
    lons = k_data_lon.lon
    lats = k_data_lat.lat
    plevels = k_data_lon.lev

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
                          z=k_data_lon.T,
                          levels=10,
                          vertical_title=f'{group}',
                          title='Zonal Circulation' if idx == 0 else None,
                          x_title_offset=-0.4,
                          cmap='coolwarm',
                          centercolor='white',
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
        u=k_data_u.T,
        v=k_data_uw.T*100,
        x_vals=k_data_uw.lon,
        y_vals=k_data_uw.lev,
        steps=1,
        x_steps=4,
        transform=False,
        scale=50,
        width=0.006,
        key_length=2,
        wind_unit=rf'm$s^{{-1}}$ | 0.02 hPa$s^{{-1}}$',
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
                              centercolor='white',
                              plot_type='contourf',
                              levels=10,
                              vmin=vmin_map, vmax=vmax_map,
                              extend='both',
                              orientation='horizontal',
                              significance_mask=mask,
                              hatch_type='..',
                              plt_grid=True,
                              gs_lon=60,
                              gs_lat=30,
                              lon_range=lon_range_c,
                              lat_range=lat_range_c,
                              )

    h_im_v = cplt.plot_2D(x=lats, y=plevels,
                          z=k_data_lat.T,
                          title='Meridional Circulation' if idx == 0 else None,
                          ax=axs[idx*ncols + 2],
                          levels=9,
                          orientation='vertical',
                          cmap='coolwarm',
                          centercolor='white',
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
                                  u=k_data_v.T,
                                  v=k_data_vw.T*50,
                                  x_vals=k_data_vw.lat,
                                  y_vals=k_data_vw.lev,
                                  steps=1,
                                  key_length=2,
                                  width=0.006,
                                  transform=False,
                                  pivot='tip',
                                  wind_unit=rf'm$s^{{-1}}$ | 0.02 hPa$s^{{-1}}$',
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
                              centercolor='white',
                              plot_type='contourf',
                              levels=9,
                              vmin=vmin_map, vmax=vmax_map,
                              extend='both',
                              orientation='horizontal',
                              significance_mask=mask,
                              hatch_type='..',
                              plt_grid=True,
                              gs_lon=60,
                              gs_lat=30,
                              lon_range=lon_range_c,
                              lat_range=lat_range_c,
                              )


cplt.add_colorbar(im=h_im_u, fig=fig,
                  sci=sci,
                  label=label_msf_u,
                  x_pos=-0.02,
                  width=0.24,
                  height=0.02,
                  y_pos=0.12,
                  tick_step=2)


cplt.add_colorbar(im=im_wind_u, fig=fig,
                  sci=sci_map,
                  label=label_vert_u,
                  x_pos=0.25,
                  width=0.25,
                  height=0.02,
                  y_pos=0.12,
                  tick_step=2
                  )

cplt.add_colorbar(im=h_im_v, fig=fig,
                  sci=sci,
                  label=label_msf_v,
                  x_pos=0.53,
                  width=0.24,
                  height=0.02,
                  y_pos=0.12,
                  tick_step=2)

cplt.add_colorbar(im=im_wind_v, fig=fig,
                  sci=sci_map,
                  label=label_vert_v,
                  x_pos=0.8,
                  width=0.25,
                  height=0.02,
                  y_pos=0.12,
                  tick_step=2
                  )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/propagation_msf_lon_lat_all.pdf"
cplt.save_fig(savepath, fig=fig)

# %%
# Eastward propagation in terms of Kelvin and Rossby Waves
reload(cplt)
reload(sut)

lon_range_c = [30, 180]
lat_range_c = [-30, 50]
plevel_mse = 850
plevel_u = 850
plevel_v = 850

an_type = 'JJAS'

label_vwind = rf'{plevel_u}-hPa v-winds JJAS anomalies [m$s^{{-1}}$]'
label_uwind = rf'{plevel_u}-hPa u-winds JJAS anomalies [m$s^{{-1}}$]'

label_olr = r'OLR JJAS anomalies [Wm$^{-2}$]'

u_var_type = f'U_an_{an_type}'
v_var_type = f'V_an_{an_type}'


vmax_olr = 2.5e1
vmin_olr = -vmax_olr
vmax_uwind = 4.5
vmin_uwind = -vmax_uwind
vmax_vwind = 1.5
vmin_vwind = -vmax_vwind
vmax_uwind = 4.5
vmin_uwind = -vmax_uwind

step = 4

data_cut_olr = sput.cut_map(ds=ds_olr_25.ds[f'an_{an_type}'],
                            lon_range=lon_range_c,
                            lat_range=lat_range_c,
                            dateline=False)

var_type_kelvin = f'vi_mse_an_{an_type}'
vmax_mse = 2.e5
vmin_mse = -vmax_mse
label_mse = r'MSE JJAS anomalies [Jm$^{-2}$]'
data_cut_mse = sput.cut_map(ds=ds_mse.ds[var_type_kelvin].sel(lev=plevel_mse),
                            lon_range=lon_range_c,
                            lat_range=lat_range_c,
                            dateline=False)

data_cut_wind = sput.cut_map(ds=ds_wind.ds[
    [u_var_type, v_var_type]].sel(lev=[plevel_u]),
    # [u_var_type, v_var_type]].sel(lev=[plevel_u, plevel_mse]),
    lon_range=lon_range_c,
    lat_range=lat_range_c,
    dateline=False)

# data_cut_mse = data_cut_wind

ncols = 3
im = cplt.create_multi_plot(nrows=3, ncols=ncols,
                            # hspace=0.4,
                            wspace=0.2,
                            projection='PlateCarree',
                            figsize=(18, 12),
                            lon_range=lon_range_c,
                            lat_range=lat_range_c,
                            # end_idx=9
                            )

plot_order = ['Canonical',
              'Eastward Blocked',
              'Quasi-stationary']

for idx, (group) in enumerate(plot_order):
    sel_tps = k_means_tps_lon_lat[group]
    # sel_tps_step = tu.add_time_step_tps(tps=sel_tps,
    #                                     time_step=step)
    sel_tps_step = tu.get_periods_tps(tps=sel_tps,
                                      start=0,
                                      end=step)

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_olr, tps=sel_tps_step)
    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_olr)
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_gph = cplt.plot_map(
        xr.where(mask, mean, np.nan),
        # mean,
        ax=im['ax'][idx*ncols],
        title='Kelvin wave pattern' if idx == 0 else None,
        vertical_title=f'{group}',
        cmap='RdBu_r',
        plot_type='contour',
        levels=9,
        vmin=vmin_olr, vmax=vmax_olr,
        orientation='horizontal',
        # significance_mask=mask,
        extend='both',
        lw=2,
        label=label_olr if idx == 2 else None,
        tick_step=3,
        # alpha=0.8
    )

    this_comp_ts_u = tu.get_sel_tps_ds(
        data_cut_wind[u_var_type].sel(lev=plevel_u), tps=sel_tps)
    this_comp_ts_v = tu.get_sel_tps_ds(
        data_cut_wind[v_var_type].sel(lev=plevel_v), tps=sel_tps)

    mean_u, pvalues_ttest_u = sut.ttest_field(
        this_comp_ts_u, data_cut_wind[u_var_type].sel(lev=plevel_u))
    mask_u = sut.field_significance_mask(
        pvalues_ttest_u, alpha=0.15, corr_type=None)
    mean_v, pvalues_ttest_v = sut.ttest_field(
        this_comp_ts_v, data_cut_wind[v_var_type].sel(lev=plevel_v))
    mask_v = sut.field_significance_mask(
        pvalues_ttest_v, alpha=0.15, corr_type=None)

    dict_w = cplt.plot_wind_field(ax=im['ax'][idx*ncols],
                                  u=xr.where(mask_u, mean_u, np.nan),
                                  v=xr.where(mask_u, mean_v, np.nan),
                                  #   u=mean_u,
                                  #   v=mean_v,
                                  scale=30,
                                  width=0.004,
                                  steps=2,
                                  key_length=1,
                                  )
    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_wind[v_var_type].sel(lev=plevel_v), tps=sel_tps_step)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_wind[v_var_type].sel(lev=plevel_v))
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_v = cplt.plot_map(mean,
                         ax=im['ax'][idx*ncols+1],
                         title='Low-level Rossby wave pattern' if idx == 0 else None,
                         cmap='PuOr',
                         centercolor='white',
                         plot_type='contourf',
                         levels=10,
                         vmin=vmin_vwind, vmax=vmax_vwind,
                         orientation='horizontal',
                         significance_mask=mask,
                         extend='both',
                         sci=None,
                         label=label_vwind if idx == 2 else None,
                         tick_step=2
                         )

    this_comp_ts = tu.get_sel_tps_ds(
        data_cut_wind[u_var_type].sel(lev=plevel_u), tps=sel_tps_step)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts, data_cut_wind[u_var_type].sel(lev=plevel_u))
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_v = cplt.plot_map(mean,
                         ax=im['ax'][idx*ncols+2],
                         title='Low-level zonal wind anomalies' if idx == 0 else None,
                         cmap='PuOr',
                         centercolor='white',
                         plot_type='contourf',
                         levels=10,
                         vmin=vmin_uwind, vmax=vmax_uwind,
                         orientation='horizontal',
                         significance_mask=mask,
                         extend='both',
                         sci=None,
                         label=label_uwind if idx == 2 else None,
                         tick_step=2
                         )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/Kelvin_Rossby_Waves.pdf"
cplt.save_fig(savepath, fig=im['fig'])
