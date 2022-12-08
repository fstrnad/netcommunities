#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
import matplotlib as mpl
import climnet.datasets.evs_dataset as cds
import geoutils.utils.indices_utils as iut
import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import xarray as xr
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import geoutils.tsa.time_series_analysis as tsa
import climnet.network.clim_networkx as cn
import geoutils.plotting.plots as cplt
import geoutils.geodata.wind_dataset as wds

# Run the null model
name = "mswep"
grid_type = "fibonacci"
grid_step = 0.5

grid_type = "fekete"
grid_step = 1

scale = "south_asia"
vname = "pr"

output_folder = "global_monsoon"

lat_range = [-15, 45]
lon_range = [55, 150]


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
# Import dataset and climnetworkX
reload(cds)
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_ds.nc"
ds = cds.EvsDataset(
    load_nc=dataset_file,
    rrevs=False,
    can=True,
    an_types=['dayofyear', 'month', 'JJAS']
)
nx_graph_file = output_dir + \
    f"{output_folder}/{name_prefix}_{q_sig}_lb_ES_nx.gml.gz"
cnx = cn.Clim_NetworkX(dataset=ds, nx_path_file=nx_graph_file)

# %%
# MSWEP precipitation
dataset_file = output_dir + \
    f"/climate_data/mswep_pr_{2.5}_ds.nc"
reload(cds)

ds_pr_mswep = cds.BaseRectDataset(load_nc=dataset_file,
                                  can=True,
                                  an_types=['JJAS'],
                                  )

# %%
# Use representative boxes
regions = [
    'EIO',
    'BoB',
    'MC',
    'SA',
    'WP',
]

cnx.ds.add_loc_dict(
    name="EIO",
    lname='Equatorial Indian Ocean',
    lon_range=(70, 80),
    lat_range=(-5, 5),
    color="tab:red",
    n_rep_ids=3,
    reset_loc_dict=True
)

cnx.ds.add_loc_dict(
    name="BoB",
    lname='Bay of Bengal',
    lon_range=(100, 110),
    lat_range=[-4, 6],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="MC",
    lname='Maritime Continent',
    lon_range=[120, 130],
    lat_range=[-3, 7],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="SA",
    lname='South Asia',
    lon_range=(108, 118),
    lat_range=[8, 18],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="WP",
    lname='West Pacific',
    lon_range=(125, 135),
    lat_range=[15, 25],
    color="tab:red",
    n_rep_ids=3,
)

regions = [
    'EIO',
    'BoB',
    'MC',
    'SA',
    'WP',
]

all_regions = [
    'EIO',
    'BoB',
    'MC',
    'SA',
    'WP',
    'NIC'
]

# %%
reload(tu)
B_max = 6
cd_folder = 'graph_tool'
savepath_loc = (
    plot_dir +
    f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_prob_maps.npy"
)


loc_dict = gut.load_np_dict(savepath_loc)

sregion = 'EIO'
comp_th = 0.95

# collect for different boxes
evs = loc_dict[sregion]['data'].evs
t_all = tsa.get_ee_ts(evs=evs)

t_jjas = tu.get_month_range_data(t_all, start_month='Jun',
                                 end_month='Sep')
tps_dict = gut.get_locmax_composite_tps(t_jjas, q=comp_th)
tps = tps_dict['peaks']
# else:
#     savepath_sync = plot_dir + \
#         f"{output_folder}/time_series/sync_times_{name_prefix}_{B_max}_{region1}_{region2}.nc"
#     ts_sync = xr.open_dataset(savepath_sync)

#     t_all = ts_sync['t']
#     ts12 = ts_sync['t12']
#     ts21 = ts_sync['t21']

#     times = ts_sync.time

#     t_jjas12 = tu.get_month_range_data(ts12, start_month='Jun',
#                                        end_month='Sep')
#     tps12_dict = gut.get_locmax_composite_tps(t_jjas12, q=comp_th)
#     tps = tps12_dict['peaks']  # Foreward directions

# %%
reload(tu)
num_days = 30
box_dict = cnx.ds.loc_dict
# box_dict = loc_dict
var = 'evs'
rm_tps = tu.remove_consecutive_tps(tps=tps, steps=1)

box_data = tu.get_box_propagation(ds=cnx.ds.ds,
                                  loc_dict=box_dict,
                                  tps=rm_tps,
                                  num_days=num_days,
                                  var=var,
                                  regions=regions,
                                  )


# %%
# ######################## Figure 1 ###########################################
reload(cplt)
time_range = ['1981-01-01', '2019-12-31']

fig = plt.figure(
    figsize=(9, 9)
)

# set up ax1 and plot the communities and event series there
proj = cplt.get_projection(projection='PlateCarree')
ax1 = fig.add_axes([0., 0.1, .7, 1],
                   projection=proj
                   )
ax2 = fig.add_axes([0.8, 0.35, 0.2, 0.47])
ax3 = fig.add_axes([0., 0., 0.2, 0.2])
ax4 = fig.add_axes([0.25, 0., 0.2, 0.2])
ax5 = fig.add_axes([0.5, 0., 0.2, 0.2])
ax6 = fig.add_axes([0.75, 0., 0.2, 0.2])
axs = [ax1, ax2, ax3, ax4, ax5, ax6]
cplt.enumerate_subplots(axs=axs)

legend_items = []
legend_item_names = []
for idx, region in enumerate(all_regions):
    this_dict = loc_dict[region]
    this_map = this_dict["map"]
    color = cplt.colors[idx]
    im = cplt.plot_map(
        this_map,
        ds=cnx.ds,
        ax=ax1,
        projection='PlateCarree',
        figsize=(9, 7),
        plt_grid=True if idx == 0 else False,
        plot_type='contour',
        ds_mask=True,
        color=color,
        levels=0, vmin=0, vmax=1,
        bar=False,
        lw=3,
        alpha=1,
        set_map=False if idx > 0 else True
    )

    im = cplt.plot_map(
        xr.where(this_map == 1, this_map, np.nan),
        ds=cnx.ds,
        ax=im['ax'],
        projection='PlateCarree',
        plt_grid=False,
        plot_type='contourf',
        color=color,
        cmap=None,
        significance_mask=True,
        levels=2,
        vmin=0, vmax=1,
        bar=False,
        alpha=0.6
    )
    ax = im['ax']
    legend_items.append(mpl.patches.Rectangle((0, 0), 1, 1,
                                              fc=color, alpha=0.5,
                                              fill=True,
                                              edgecolor=color,
                                              linewidth=2))
    legend_item_names.append(f"{this_dict['lname']} ({region})")

    # if region != 'NIC':
    #     this_box_dict = box_dict[region]
    #     cplt.plot_rectangle(ax=ax1,
    #                         lon_range=this_box_dict['lon_range'],
    #                         lat_range=this_box_dict['lat_range'],
    #                         color='k',
    #                         lw=2)

cplt.set_legend(ax=im['ax'],
                legend_items=legend_items,
                label_arr=legend_item_names,
                loc='outside',
                ncol_legend=2,
                box_loc=(0, 0))

# set up second axis right next to and plot box propagation
days = box_data.x
cplt.plot_2D(x=regions, y=days, z=box_data.T,
             ax=ax2,
             plot_type='colormesh',
             cmap='rainbow_r',
             orientation='vertical',
             #  label=f'Precipitation anomalies (wrt JJAS)',
             label=f'No. of EREs per 100 locations  ',
             vmin=0,
             vmax=21,
             extend='max',
             rot=90,
             ylabel='days',
             ylim=[-0, 30]
             #  ylim=[0, 26]
             )

# Plot the lead-lag correlations

region_pairs = [['EIO', 'BoB'],
                ['BoB', 'MC'],
                ['MC', 'SA'],
                ['SA', 'WP']]
q_min = 0.9
cutoff = 3
p = 0.05
for idx, (region1, region2) in enumerate(region_pairs):
    this_ax = axs[idx+2]
    evs1 = loc_dict[region1]['data'].evs
    t_ee1 = tsa.get_ee_ts(evs=evs1)
    t_ee1 = tu.get_sel_time_range(t_ee1, time_range=time_range,
                                  start_month='Jun',
                                  end_month='Sep')

    evs2 = loc_dict[region2]['data'].evs
    t_ee2 = tsa.get_ee_ts(evs=evs2)
    t_ee2 = tu.get_sel_time_range(t_ee2, time_range=time_range,
                                  start_month='Jun',
                                  end_month='Sep'
                                  )

    maxlags = 45
    ll_dict1 = tu.lead_lag_corr(ts1=t_ee1, ts2=t_ee2, maxlags=maxlags,
                                cutoff=cutoff, corr_method='spearman',
                                )
    tau_lag = ll_dict1['tau']
    lag_corr1 = ll_dict1['corr']
    p_vals = ll_dict1['p_val']
    cplt.plot_xy(
        ax=this_ax,
        x_arr=[tau_lag
               ],
        y_arr=[lag_corr1,
               ],
        title=f'{region1} - {region2}',
        y_title=1.1,
        xlabel=r'Time-Lag $\tau$ [days]',
        ylabel='Correlation' if idx == 0 else None,
        yticks=True if idx == 0 else None,
        ylim=(-0.4, 0.45),
        loc='upper right',
        lw_arr=[3],
        ls_arr=['-', '-', '-'],
        mk_arr=[None, None, None],
        stdize=False,
        set_grid=True
    )

    cplt.plot_hline(
        ax=this_ax,
        y=p,
        color='black',
        ls='dashed',
        lw=2,
        label="significance level")

    cplt.plot_hline(
        ax=this_ax,
        y=-p,
        color='black',
        ls='dashed',
        lw=2,
        label="significance level")

    # cplt.plot_vline(
    #     ax=im['ax'][idx],
    #     x=ll_dict1['tau_min'],
    #     color='red',
    #     ls='solid',
    #     lw=3,
    # )
    # cplt.plt_text(ax=im['ax'][idx],
    #               xpos=ll_dict1['tau_min']+0.5,
    #               ypos=-0.35,
    #               text=f'day {ll_dict1["tau_min"]}',
    #               )
    cplt.plot_vline(
        ax=this_ax,
        x=ll_dict1['tau_max'],
        color='red',
        ls='solid',
        lw=3,
    )
    cplt.plt_text(ax=this_ax,
                  xpos=ll_dict1['tau_max']+0.5,
                  ypos=-0.35,
                  text=f'day +{ll_dict1["tau_max"]}',
                  box=False,
                  )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/communities_lead_lag_{sregion}.png"
cplt.save_fig(savepath=savepath, fig=fig)

# %%
# ######################## Figure 2 ############################################
# ACTIVE and PHASE  BSISO
reload(tu)
reload(gut)
reload(cplt)
reload(iut)
reload(tsa)

B_max = 6
cd_folder = 'graph_tool'
savepath_loc = (
    plot_dir +
    f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_prob_maps.npy"
)

loc_dict = gut.load_np_dict(savepath_loc)

time_range = ['1981-01-01', '2019-12-31']
comp_th = 0.9
bsiso_th = 1.8
ncols = 3
nrows = 2
im = cplt.create_plot(nrows=nrows, ncols=ncols,
                      #   figsize=(10, 9),
                      figsize=(12, 6),
                      #   subplot_kw={'projection': 'polar'}
                      )

# Count EEs
for idx, region in enumerate(all_regions):
    this_dict = loc_dict[region]
    # if region != 'MC' else this_dict['data_range'].evs
    evs = this_dict['data'].evs
    # if region == 'Maritime':
    #     evs = sput.get_locations_in_range(def_map=evs,
    #                                       lat_range=[-10, 15],
    #                                       lon_range=[90, 150]
    #                                       )
    t_all = tsa.get_ee_ts(evs=evs)
    t_all = tu.get_sel_time_range(t_all, time_range=time_range,
                                  start_month='Jun', end_month='Sep',
                                  verbose=False)

    # Sync Times
    t_all = tu.get_sel_time_range(t_all, time_range=time_range,
                                  start_month='Jun',
                                  end_month='Sep',
                                  verbose=False)

    # Synchronous days
    times = t_all.time
    comp_th = 0.9
    tps_sync = gut.get_quantile_of_ts(t_all, q=comp_th)
    # tps_sync = gut.get_locmax_of_ts(t_all, q=comp_th)
    tps_not_sync = np.setxor1d(times, tps_sync.time)

    # BSISO Index Phase Active
    bsiso_index = iut.get_bsiso_index(time_range=time_range,
                                      start_month='Jun',
                                      end_month='Sep')

    bsiso_phase = bsiso_index['BSISO1-phase']
    ampl = bsiso_index['BSISO1']

    phase_sync = tu.get_sel_tps_ds(bsiso_phase, tps=tps_sync)

    # Get all active/break days
    act_days = bsiso_phase[np.where(ampl >= bsiso_th)].time
    phase_act = bsiso_phase[np.where(ampl >= bsiso_th)]
    phase_break = bsiso_phase[np.where(ampl < bsiso_th)]
    tps_not_sync_act = np.setxor1d(times, phase_act.time)

    p_a = len(phase_act) / len(bsiso_phase)
    p_b = len(phase_break) / len(bsiso_phase)

    # Joint prob, sync, active, phase
    phase_sync_act = tu.get_sel_tps_ds(ds=phase_act, tps=tps_sync)
    phase_not_sync_act = tu.get_sel_tps_ds(
        ds=bsiso_phase, tps=tps_not_sync_act)
    phase_sync_break = tu.get_sel_tps_ds(ds=phase_break, tps=tps_sync)

    phase_tps_sync = tu.get_sel_tps_ds(ds=bsiso_phase, tps=tps_sync)
    phase_tps_not_sync = tu.get_sel_tps_ds(ds=bsiso_phase, tps=tps_not_sync)

    phase_vals = np.arange(1, 9)
    # Determine time points
    count_phase = tsa.count_tps_occ(tps_arr=[bsiso_phase.data],
                                    count_arr=phase_vals,
                                    counter=None,
                                    rel_freq=False)

    count_phase_sync = tsa.count_tps_occ(tps_arr=[phase_tps_sync.data],
                                         count_arr=phase_vals,
                                         counter=None,
                                         rel_freq=False)
    count_phase_not_sync = tsa.count_tps_occ(tps_arr=[phase_tps_not_sync.data],
                                             count_arr=phase_vals,
                                             counter=None,
                                             rel_freq=False)

    # Tps active/break of phase = joint probabilites (P(p,a))
    count_phase_act = tsa.count_tps_occ(tps_arr=[phase_act.data],
                                        count_arr=phase_vals,
                                        counter=None,
                                        rel_freq=False)
    count_phase_break = tsa.count_tps_occ(tps_arr=[phase_break.data],
                                          count_arr=phase_vals,
                                          counter=None,
                                          rel_freq=False)

    # Sync + Active/Break
    count_phase_act_sync = tsa.count_tps_occ(tps_arr=[phase_sync_act.data],
                                             count_arr=phase_vals,
                                             counter=None,
                                             rel_freq=False)
    count_act_not_sync = tsa.count_tps_occ(tps_arr=[phase_not_sync_act.data],
                                           count_arr=phase_vals,
                                           counter=None,
                                           rel_freq=False)
    count_phase_break_sync = tsa.count_tps_occ(tps_arr=[phase_sync_break.data],
                                               count_arr=phase_vals,
                                               counter=None,
                                               rel_freq=False)

    p_s_1_act = count_phase_act_sync/count_phase_act
    p_s_1_break = count_phase_break_sync/count_phase_break
    p_s_1_phase = count_phase_sync/count_phase

    p_s_0_act = count_act_not_sync/count_phase_act
    p_s_0_phase = count_phase_not_sync/count_phase

    label_arr = ['P(EREs|phase, BSISO active):\nProb. of highly synchr. EREs per phase,\ngiven active BSISO',
                 'P(EREs|phase, BSISO inactive):\nProb. of highly synchr. EREs per phase,\ngiven inactive BSISO',
                 'P(EREs|phase)',
                 ]
    set_legend = False if idx < len(im['ax'])-1 else True

    im_bar = cplt.plot_xy(
        plot_type='bar',
        ax=im['ax'][idx],
        x_arr=phase_vals,
        xticks=phase_vals,
        y_arr=[p_s_1_act,
               p_s_1_break,
               #   p_s_1_phase,
               ],
        set_legend=False,
        label_arr=label_arr,
        ylim=(0, 0.65),
        color_arr=['steelblue', 'firebrick', 'darkgray'],
        title=f'{this_dict["lname"]}',  # {time_range[0]} - {time_range[1]}
        xlabel='BSISO Phase' if idx >= nrows*(ncols-1)-1 else None,
        ylabel='Likelihood' if idx % ncols == 0 else None)

    cplt.plot_hline(ax=im['ax'][idx],
                    y=1-comp_th,
                    label='Null Model',
                    lw=2
                    )

order = [1, 2, 0]
cplt.set_legend(ax=im['ax'][-1],
                fig=im['fig'],
                # order=order,
                loc='outside',
                ncol_legend=3,
                box_loc=(0.05, 0)
                )
savepath = plot_dir +\
    f"{output_folder}/paper_plots/conditioned_bsiso_phase.pdf"
cplt.save_fig(savepath, fig=im['fig'])


# %%
# ################################## Figure 4 ###############################
# Precipitation together with local Hadley circulation (convergence zone)
reload(cplt)
lon_range_c = [0, 179]
lat_range_c = [-60, 60]
name_prefix = 'sa'

an_type = 'JJAS'
plevel = 500

ds_comp = ds_hd


vmax = 3e4
# vmax = 1
# vmax = 8e2
# vmax = 2e2
# vmax = 6e4
vmin = -vmax

sci = 3

nrows = 5
ncols = 2
im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            figsize=(10, 20),
                            projection='PlateCarree',
                            orientation='horizontal',
                            hspace=0.05,
                            wspace=0.2,
                            end_idx=2*len(regions))

for i, region in enumerate(regions):

    # EE TS
    this_dict = loc_dict[region]
    evs = this_dict['data'].evs
    t_all = tsa.get_ee_ts(evs=evs)

    comp_th = 0.9
    t_jjas = tu.get_month_range_data(t_all, start_month='Jun',
                                     end_month='Sep')
    tps_dict = gut.get_locmax_composite_tps(t_jjas, q=comp_th)
    tps = tps_dict['peaks']
    for j, ds_comp in enumerate([ds_pr_mswep, ds_hd]):
        idx = i*ncols + j
        if idx % 2 == 0:
            ds_comp = ds_pr_mswep
            var_type = f'an_{an_type}'
            this_comp_ts = tu.get_sel_tps_ds(ds=ds_comp.ds, tps=tps)
        else:
            ds_comp = ds_hd
            this_comp_ts = tu.get_sel_tps_ds(
                ds=ds_comp.ds.sel(plevel=plevel), tps=tps)
            var_type = f'mf_v_an_{an_type}'

        mean_ts = this_comp_ts[var_type].mean(dim='time')

        mean_ts = sput.cut_map(mean_ts, lon_range=lon_range_c,
                               lat_range=lat_range_c)

        if idx % 2 == 0:
            im_pr = cplt.plot_map(mean_ts,
                                  ax=im['ax'][idx],
                                  title=f'{this_dict["lname"]}',
                                  vertical_title=True,
                                  title_fontweight='bold',
                                  cmap='RdBu',
                                  plot_type='contourf',
                                  levels=12,
                                  vmin=-5, vmax=5,
                                  projection='PlateCarree',
                                  bar=False,
                                  round_dec=1,
                                  plt_grid=True,
                                  orientation='horizontal',
                                  )
        else:
            im_hd = cplt.plot_map(mean_ts,
                                  ax=im['ax'][idx],
                                  #   title=f'{this_dict["lname"]}',
                                  cmap='PuOr',
                                  plot_type='contourf',
                                  levels=12,
                                  vmin=vmin, vmax=vmax,
                                  projection='PlateCarree',
                                  bar=False,
                                  round_dec=1,
                                  plt_grid=True,
                                  orientation='horizontal',
                                  )
            this_comp_wind = tu.get_sel_tps_ds(ds=ds_hd.ds.sel(plevel=200),
                                               tps=tps)
            this_comp_wind = this_comp_wind.mean(dim='time')
            mean_wind = sput.cut_map(this_comp_wind,
                                     lon_range=lon_range_c,
                                     lat_range=lat_range_c)

            dict_w = cplt.plot_wind_field(ax=im['ax'][idx],
                                          u=mean_wind[f'u_chi_an_{an_type}'],
                                          v=mean_wind[f'v_chi_an_{an_type}'],
                                          #   u=mean_wind[f'ewvf_an_{an_type}'],
                                          #   v=mean_wind[f'nwvf_an_{an_type}'],
                                          lw=1,
                                          steps=3,
                                          key_length=1,
                                          )
        loc_lons = this_dict['data'].lon
        loc_lats = this_dict['data'].lat

        locs = gut.zip_2_lists(loc_lons, loc_lats)

        loc_map = ds_wind.get_map_for_locs(locations=locs)
        loc_map = sput.cut_map(loc_map, lon_range=lon_range_c,
                               lat_range=lat_range_c)
        cplt.plot_map(loc_map,
                      ax=im['ax'][idx],
                      plot_type='contour',
                      color='magenta',
                      levels=1,
                      #   vmin=-5, vmax=5,
                      bar=False,
                      plt_grid=True,
                      intpol=False,
                      tick_step=2,
                      lw=3,
                      extend='both',
                      gl_plt=False)

label = f'Precipitation anomalies [mm/day]'
cbar = cplt.add_colorbar(im=im_pr,
                         fig=im['fig'],
                         x_pos=0.13,
                         y_pos=0.09, width=0.35,
                         height=0.01,
                         orientation='horizontal',
                         label=label,
                         round_dec=1,
                         sci=None,
                         tick_step=2,
                         )

label = rf'MSF anomalies ({plevel} hPa) [kg m/s]'
label = rf'MSF anomalies [kg m/s]  '
cbar = cplt.add_colorbar(im=im_hd,
                         fig=im['fig'],
                         x_pos=0.55,
                         y_pos=0.09, width=0.35,
                         height=0.01,
                         orientation='horizontal',
                         label=label,
                         #  round_dec=0,
                         sci=3,
                         tick_step=2,
                         )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/pr_{plevel}_msf_all.png"
cplt.save_fig(savepath,
              fig=im['fig'])


# %%
# Propagation plots of China community
# EE TS progression
reload(tu)

region = 'CH'
# EE TS
this_dict = loc_dict[region]
evs = this_dict['data'].evs
t_all = tsa.get_ee_ts(evs=evs)

comp_th = 0.9
t_jjas = tu.get_month_range_data(t_all, start_month='Jun',
                                 end_month='Sep')
tps_dict = gut.get_locmax_composite_tps(t_jjas, q=comp_th)
tps = tps_dict['peaks']

lon_range_c = [-90, 179]
lat_range_c = [-10, 70]

ds_comp = ds_hd

an_type = 'JJAS'
var_type = f'mf_v_an_{an_type}'
label = rf'Anomalies MSF v - $\chi$ (wrt {an_type}) {plevel} hPa [m/s]'

# var_type = f'v_chi_an_{an_type}'
# label = rf'Anomalies v - $\chi$ (wrt {an_type}) {plevel} hPa [m/s]'

var_type = f'v_psi_an_{an_type}'
label = rf'Anomalies v - $\psi$ (wrt {an_type}) {plevel} hPa [m/s]'

plevel = 150
vmax = 2
vmin = -vmax
sci = None

region = 'CH'

start_off = 4
end = 4
ncols = 3
step = 1
composite_arrs = tu.get_day_progression_arr(ds=ds_comp.ds.sel(plevel=plevel),
                                            tps=tps,
                                            # sps=sps, eps=eps,
                                            start=start_off,
                                            end=end,
                                            step=step,
                                            var=None)

nrows = int(np.ceil(len(composite_arrs.day)/ncols))

im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            figsize=(14, 6),
                            projection='PlateCarree',
                            orientation='horizontal',
                            # hspace=0.25,
                            wspace=0.25,
                            )
for idx, (day) in enumerate(composite_arrs.day):
    mean_ts = composite_arrs.sel(day=day)
    mean_wind = sput.cut_map(mean_ts, lon_range=lon_range_c,
                             lat_range=lat_range_c)
    im_comp = cplt.plot_map(ds_wind.ds,
                            mean_wind[var_type],
                            ax=im['ax'][idx],
                            title=f'Day {int(day)}',
                            plot_type='contourf',
                            cmap='RdBu_r',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            bar=False,
                            extend='both',
                            plt_grid=True,
                            intpol=False,
                            tick_step=2,
                            round_dec=2)

    # dict_w = cplt.plot_wind_field(ax=im['ax'][idx],
    #                               u=mean_wind[f'u_psi_an_{an_type}'],
    #                               v=mean_wind[f'v_psi_an_{an_type}'],
    #                               #   u=mean_wind[f'ewvf_an_{an_type}'],
    #                               #   v=mean_wind[f'nwvf_an_{an_type}'],
    #                               lw=1,
    #                               steps=3,
    #                               key_length=.5,
    #                               )

cplt.add_colorbar(im=im_comp,
                  fig=im['fig'],
                  x_pos=0.2,
                  y_pos=0.08, width=0.6, height=0.02,
                  orientation='horizontal',
                  label=label,
                  round_dec=2,
                  tick_step=2,
                  sci=sci
                  )


savepath = plot_dir +\
    f"{output_folder}/paper_plots/wind_{region}_{plevel}_{var_type}_day_progression.png"
cplt.save_fig(savepath, fig=im['fig'])


# %%
# Propagation plots of China community
# EE TS VIMD
# %%
# Moisture divergence
dataset_file = output_dir + \
    f"/climate_data/era5_vimd_{2.5}_ds.nc"
reload(cds)
ds_vimd = cds.BaseRectDataset(load_nc=dataset_file,
                              can=True,
                              an_types=['JJAS'],
                              )
# %%
# IWF
reload(wds)
dataset_file_ewf = output_dir + \
    f"/climate_data/era5_ewvf_{2.5}_ds.nc"
dataset_file_nwf = output_dir + \
    f"/climate_data/era5_nwvf_{2.5}_ds.nc"

ds_ivf = wds.Wind_Dataset(load_nc_arr_u=[dataset_file_ewf],
                          load_nc_arr_v=[dataset_file_nwf],
                          can=True,
                          an_types=['JJAS'],
                          u_name='ewvf',
                          v_name='nwvf'
                          )
# %%
# OLR data
reload(cds)
dataset_file = output_dir + \
    f"/climate_data/era5_ttr_{2.5}_ds.nc"

ds_olr = cds.BaseRectDataset(load_nc=dataset_file,
                             can=True,
                             an_types=['month', 'JJAS'],
                             )
# %%
# OLR
region = 'CH'
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

lon_range_c = [-90, 179]
lat_range_c = [-10, 70]

an_type = 'JJAS'
var_type = f'an_{an_type}'
label = f'Anomalies OLR (wrt {an_type}) [J/m$^2$]'

vmax = 6e4
vmin = -vmax
sci = 3

start_off = 4
end = 4
ncols = 3
step = 1
plevel = 150
composite_arrs = tu.get_day_progression_arr(ds=ds_olr.ds,
                                            tps=tps,
                                            # sps=sps, eps=eps,
                                            start=start_off,
                                            end=end,
                                            step=step,
                                            var=None)
composite_arrs_wind = tu.get_day_progression_arr(ds=ds_hd.ds.sel(plevel=plevel),
                                                 tps=tps,
                                                 # sps=sps, eps=eps,
                                                 start=start_off,
                                                 end=end,
                                                 step=step,
                                                 var=None)

nrows = int(np.ceil(len(composite_arrs.day)/ncols))

im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            figsize=(14, 6),
                            projection='PlateCarree',
                            orientation='horizontal',
                            # hspace=0.25,
                            wspace=0.25,
                            )
for idx, (day) in enumerate(composite_arrs.day):
    mean_ts = composite_arrs.sel(day=day)
    mean_data = sput.cut_map(mean_ts, lon_range=lon_range_c,
                             lat_range=lat_range_c)
    im_comp = cplt.plot_map(ds_olr.ds,
                            -1*mean_data[var_type],
                            ax=im['ax'][idx],
                            title=f'Day {int(day)}',
                            plot_type='contourf',
                            cmap='RdBu_r',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            bar=False,
                            extend='both',
                            plt_grid=True,
                            intpol=False,
                            tick_step=2,
                            round_dec=2)

    mean_wind = composite_arrs_wind.sel(day=day)
    mean_wind = sput.cut_map(mean_wind, lon_range=lon_range_c,
                             lat_range=lat_range_c)

    dict_w = cplt.plot_wind_field(ax=im['ax'][idx],
                                  u=mean_wind[f'u_psi_an_{an_type}'],
                                  v=mean_wind[f'v_psi_an_{an_type}'],
                                  lw=1,
                                  steps=4,
                                  key_length=2,
                                  key_loc=(0.95, -0.13)
                                  )

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
    f"{output_folder}/paper_plots/olr_{region}_{var_type}_day_progression.png"
cplt.save_fig(savepath, fig=im['fig'])


# %%
reload(tu)
region = 'CH'
# EE TS
this_dict = loc_dict[region]
evs = this_dict['data'].evs
t_all = tsa.get_ee_ts(evs=evs)

comp_th = 0.9
t_jjas = tu.get_month_range_data(t_all, start_month='Jun',
                                 end_month='Sep')
tps_dict = gut.get_locmax_composite_tps(t_jjas, q=comp_th)
tps = tps_dict['peaks']
sps = tps_dict['sps']
eps = tps_dict['eps']

lon_range_c = [20, 179]
lat_range_c = [-10, 70]

ds_comp = ds_ivf
var_type = f'windspeed_an_{an_type}'
label = rf'Anomalies IWF (wrt {an_type}) $[kg/m^2]$ '

vmax = 1e2
vmin = -vmax
sci = 1

region = 'CH'

start_off = 4
end = 4
ncols = 3
step = 1
composite_arrs = tu.get_day_progression_arr(ds=ds_comp.ds.sel(plevel=0),
                                            tps=tps,
                                            sps=sps, eps=eps,
                                            start=start_off,
                                            end=end,
                                            step=step,
                                            var=None)

nrows = int(np.ceil(len(composite_arrs.day)/ncols))

im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            figsize=(12, 8),
                            projection='PlateCarree',
                            orientation='horizontal',
                            hspace=0.1,
                            wspace=0.25,
                            )
for idx, (day) in enumerate(composite_arrs.day):
    mean_ts = composite_arrs.sel(day=day)
    mean_wind = sput.cut_map(mean_ts, lon_range=lon_range_c,
                             lat_range=lat_range_c)
    im_comp = cplt.plot_map(ds_wind.ds,
                            mean_wind[var_type],
                            ax=im['ax'][idx],
                            title=f'Day {int(day)}',
                            plot_type='contourf',
                            cmap='RdBu_r',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            bar=False,
                            extend='both',
                            plt_grid=True,
                            intpol=False,
                            tick_step=2,
                            round_dec=2)

    dict_w = cplt.plot_wind_field(ax=im['ax'][idx],
                                  #   u=mean_wind[f'u_psi_an_{an_type}'],
                                  #   v=mean_wind[f'v_psi_an_{an_type}'],
                                  u=mean_wind[f'ewvf_an_{an_type}'],
                                  v=mean_wind[f'nwvf_an_{an_type}'],
                                  lw=1,
                                  steps=3,
                                  key_length=20,
                                  wind_unit='kg/m s',
                                  key_loc=(0.95, -0.11)
                                  )

cplt.add_colorbar(im=im_comp,
                  fig=im['fig'],
                  x_pos=0.2,
                  y_pos=0.08, width=0.6, height=0.02,
                  orientation='horizontal',
                  label=label,
                  #   round_dec=2,
                  tick_step=2,
                  sci=sci
                  )


savepath = plot_dir +\
    f"{output_folder}/paper_plots/IWF_{region}_{var_type}_day_progression.png"
cplt.save_fig(savepath, fig=im['fig'])


# %% ############################### Appendix
# Appendix: Quantiles and EREs
ds_jjas = tu.get_month_range_data(ds.ds, start_month=start_month,
                                  end_month=end_month)
q_val_map, ee_map, data_above_quantile, rel_frac_q_map = tu.get_ee_ds(
    dataarray=ds_jjas[vname], q=q_ee, th=1, th_eev=1
)
# %%
# Plot together EEs and Q-values
reload(cplt)
fdic = cplt.create_multi_map_plot_gs(
    1, 2,
    figsize=(12, 5),
    wspace=0.15,
    orientation=None,
    projection="PlateCarree")
cplt.plot_map(
    ds,
    q_val_map,
    ax=fdic["ax"][0],
    plot_type="contourf",
    label=rf"$Q_{{pr}}({{{ds.q}}})$ [mm/day]",
    ds_mask=True,
    orientation="horizontal",
    vmin=0,
    vmax=35,
    projection="PlateCarree",
    plt_grid=True,
    levels=12,
    cmap="coolwarm_r",
    tick_step=2,
    round_dec=1,
)
cplt.plot_map(
    ds,
    ee_map,
    ax=fdic["ax"][1],
    plot_type="contourf",
    label=f"Number of EREs",
    ds_mask=True,
    orientation="horizontal",
    projection="PlateCarree",
    plt_grid=True,
    vmin=0,
    vmax=450,
    levels=12,
    tick_step=2,
    round_dec=1,
    cmap="coolwarm_r",
)

savepath = f"{plot_dir}/{output_folder}/paper_plots/ee_and_quantile.png"
cplt.save_fig(savepath)

# %%
# Analyze months
reload(tsa)

time_range = ['1981-01-01', '2019-12-31']
comp_th = 0.9
ncols = 2
nrows = 3
im = cplt.create_plot(nrows=nrows, ncols=ncols,
                      figsize=(10, 9),
                      #   subplot_kw={'projection': 'polar'}
                      )

# Count EEs
for idx, region in enumerate(all_regions):
    this_dict = loc_dict[region]
    evs = this_dict['data'].evs
    # if region == 'Maritime':
    #     evs = sput.get_locations_in_range(def_map=evs,
    #                                       lat_range=[-10, 15],
    #                                       lon_range=[90, 150]
    #                                       )
    t_all = tsa.get_ee_ts(evs=evs)
    t_all = tu.get_sel_time_range(t_all, time_range=time_range,
                                  start_month='Jun', end_month='Sep')

    # Synchronous days
    times = t_all.time
    comp_th = 0.9
    tps_sync = gut.get_quantile_of_ts(t_all, q=comp_th)

    mnths_rel = tsa.count_tps_occ_evs(
        evs=evs, counter='month', count_arr=tu.jjas_months)
    tps_rel = tsa.count_tps_occ(
        tps_arr=[tps_sync.time], counter='month', count_arr=tu.jjas_months)
    im_bars = cplt.plot_xy(
        ax=im['ax'][idx],
        x_arr=tu.jjas_months,
        y_arr=[mnths_rel, tps_rel],
        plot_type='bar',
        label_arr=['All Extreme Events', 'Most Synchronous Events'],
        figsize=(10, 6),
        title=f'{this_dict["lname"]}',
        # xlabel='Month',
        ylabel='relative frequency' if idx % ncols == 0 else None,
        loc='upper right',
        set_legend=False)


cplt.set_legend(ax=im['ax'][-1],
                fig=im['fig'],
                # order=order,
                loc='outside',
                ncol_legend=3,
                box_loc=(0.05, 0)
                )
savepath = plot_dir +\
    f"{output_folder}/paper_plots/months_communities_eres.pdf"
cplt.save_fig(savepath, fig=im['fig'])


# %%
# Appendix: Synchronous EEs

lat_range = [22, 25]
lon_range = [75, 80]

lat_range = [18, 22]
lon_range = [74, 88]

link_dict = cnx.get_edges_nodes_for_region(
    lon_range=lon_range, lat_range=lat_range, binary=False
)

# Plot nodes where edges go to
im = cplt.plot_map(
    cnx.ds,
    link_dict['target_map'],
    # label=f"Local degree",
    label=f"No. of links to node",
    projection="PlateCarree",
    plt_grid=True,
    plot_type="colormesh",
    ds_mask=True,
    cmap="Greens",
    vmin=0,
    vmax=20,
    levels=10,
    bar=True,
    alpha=0.7,
    size=10,
    extend='max',
    # tick_step=2,
    fillstyle="none",
)

im = cplt.plot_edges(
    cnx.ds,
    link_dict['el'][::10],
    ax=im["ax"],
    significant_mask=True,
    orientation="vertical",
    projection="EqualEarth",
    lw=0.2,
    alpha=0.6,
    color="grey",
    plt_grid=False,
)

cplt.plot_rectangle(
    ax=im["ax"],
    lon_range=lon_range,
    lat_range=lat_range,
    color="magenta",
    lw=3,
    zorder=11
)
savepath = f"{plot_dir}/{output_folder}/paper_plots/sync_ee_network_links.png"
cplt.save_fig(savepath)

# %%
#
ds_jjas = tu.get_month_range_data(cnx.ds.ds, start_month=start_month,
                                  end_month=end_month)
q_val_map, ee_map, data_above_quantile, rel_frac_q_map = tu.get_ee_ds(
    dataarray=ds_jjas[vname], q=q_ee, th=0, th_eev=0
)
# %%
# Plot together EEs and Q-values
reload(cplt)
fdic = cplt.create_multi_plot(
    nrows=1, ncols=2,
    figsize=(12, 5),
    wspace=0.15,
    orientation=None,
    projection="PlateCarree")
cplt.plot_map(
    q_val_map,
    ds=cnx.ds,
    ax=fdic["ax"][0],
    plot_type="contourf",
    label=rf"$Q_{{pr}}({{{ds.q}}})$ [mm/day]",
    significance_mask=True,
    orientation="horizontal",
    vmin=0,
    vmax=40,
    projection="PlateCarree",
    plt_grid=True,
    levels=12,
    cmap="coolwarm_r",
    tick_step=2,
    round_dec=1,
)
cplt.plot_map(
    ee_map,
    ds=cnx.ds,
    ax=fdic["ax"][1],
    plot_type="contourf",
    label=f"Number of EREs",
    significance_mask=True,
    orientation="horizontal",
    projection="PlateCarree",
    plt_grid=True,
    vmin=0,
    vmax=450,
    levels=12,
    tick_step=2,
    round_dec=1,
    cmap="coolwarm_r",
)

savepath = f"{plot_dir}/{output_folder}/ee_plots/ee_and_quantile.png"
cplt.save_fig(savepath)
# %%
cplt.plot_map(
    rel_frac_q_map,
    ds=cnx.ds,
    plot_type="colormesh",
    label=f"Contribution EREs to JJAS rainfall.",
    significance_mask=True,
    orientation="horizontal",
    projection="PlateCarree",
    plt_grid=True,
    vmin=0,
    vmax=1,
    levels=5,
    tick_step=1,
    round_dec=2,
    cmap="rainbow",
)

savepath = f"{plot_dir}/{output_folder}/ee_plots/rel_frac.png"
cplt.save_fig(savepath)


# %%
# TS progression precipitation
reload(tu)
lon_range_c = [-30, 179]
lat_range_c = [-70, 70]
an_type = 'JJAS'
ds_comp = ds_pr_mswep

region = 'EIO'

var_type = f'an_{an_type}'
label = rf'Anomalies Precipitation (wrt {an_type}) [mm/day]'

vmax = 3
vmin = -vmax
sci = None

# collect for different boxes
evs = loc_dict[region]['data'].evs
t_all = tsa.get_ee_ts(evs=evs)

comp_th = 0.9
t_jjas = tu.get_month_range_data(t_all, start_month='Jun',
                                 end_month='Sep')
# tps = tps_dict['peaks']
tps = gut.get_quantile_of_ts(t_jjas, q=comp_th)


start_off = 0
end = 20
ncols = 3
step = 2
composite_arrs = tu.get_day_progression_arr(ds=ds_comp.ds,
                                            tps=tps,
                                            # sps=sps, eps=eps,
                                            start=start_off,
                                            end=end,
                                            step=step,
                                            var=None)

nrows = int(np.ceil(len(composite_arrs.day)/ncols))

im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            projection='PlateCarree',
                            orientation='horizontal',
                            hspace=0.25, wspace=0.15,
                            end_idx=len(composite_arrs.day)
                            )
label = f'Anomalies (wrt {an_type}) {plevel}hPa [m/s]'
for idx, (day) in enumerate(composite_arrs.day):
    mean_ts = composite_arrs.sel(day=day)
    mean_ts = sput.cut_map(mean_ts, lon_range=lon_range_c,
                           lat_range=lat_range_c)
    im_comp = cplt.plot_map(mean_ts[var_type],
                            ax=im['ax'][idx],
                            title=f'Day {int(day)}',
                            plot_type='contourf',
                            cmap='PuOr',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            bar=False,
                            plt_grid=True,
                            intpol=False,
                            tick_step=2,
                            round_dec=2)

    # mean_wind = composite_arrs.sel(day=day)
    # mean_wind = sput.cut_map(mean_wind.sel(plevel=150),
    #                          lon_range=lon_range_c,
    #                          lat_range=lat_range_c)

    # dict_w = cplt.plot_wind_field(ax=im['ax'][idx],
    #                               u=mean_wind[f'u_chi_an_{an_type}'],
    #                               v=mean_wind[f'v_chi_an_{an_type}'],
    #                               #   u=mean_wind[f'ewvf_an_{an_type}'],
    #                               #   v=mean_wind[f'nwvf_an_{an_type}'],
    #                               lw=1,
    #                               steps=3,
    #                               key_length=.5,
    #                               )

cplt.add_colorbar(im=im_comp,
                  fig=im['fig'],
                  x_pos=0.2,
                  y_pos=0.08, width=0.6, height=0.01,
                  orientation='horizontal',
                  label=label,
                  round_dec=2,
                  sci=sci
                  )

savepath = plot_dir +\
    f"{output_folder}/paper_plots/pr_{var_type}_{region}_day_progression.png"
cplt.save_fig(savepath, fig=im['fig'])


# %%
# %%
# Atmospheric conditions for the clustered days
reload(cplt)
an_type = 'JJAS'
var_type = f'mf_v_an_{an_type}'
plevel = 500
label = rf'Anomalies MSF $\Psi_v$ (wrt {an_type}) {plevel} hPa [kg m/s]'

vmax = 5e4
# vmax = 1
# vmax = 8e2
# vmax = 2e2
# vmax = 6e4
vmin = -vmax
lon_range_c = [30, -160]
lat_range_c = [-50, 70]
wind_cut = sput.cut_map(ds=ds_hd.ds.sel(plevel=200),
                        lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=True)
data_cut = sput.cut_map(ds=ds_hd.ds.sel(plevel=plevel),
                        lon_range=lon_range_c,
                        lat_range=lat_range_c,
                        dateline=True)

wind_cut = tu.get_month_range_data(
    wind_cut, start_month='Jun', end_month='Sep')

im = cplt.create_multi_plot(nrows=2,
                            ncols=3,
                            figsize=(15, 8),
                            projection='PlateCarree',
                            orientation='horizontal',
                            hspace=0.15,
                            wspace=0.25,
                            end_idx=len(regions),
                            central_longitude=180
                            )

for idx, region in enumerate(regions):

    # EE TS
    this_dict = loc_dict[region]
    evs = this_dict['data'].evs
    t_all = tsa.get_ee_ts(evs=evs)

    comp_th = 0.9
    t_jjas = tu.get_month_range_data(t_all, start_month='Jun',
                                     end_month='Sep')
    tps_dict = gut.get_locmax_composite_tps(t_jjas, q=comp_th)
    tps = tps_dict['peaks']

    data_cut = tu.get_month_range_data(
        data_cut, start_month='Jun', end_month='Sep')
    this_comp_ts = tu.get_sel_tps_ds(
        data_cut, tps=tps)

    mean, pvalues_ttest = sut.ttest_field(
        this_comp_ts[var_type], data_cut[var_type])
    mask = sut.field_significance_mask(
        pvalues_ttest, alpha=0.05, corr_type=None)

    im_comp = cplt.plot_map(mean,
                            ax=im['ax'][idx],
                            title=f'{this_dict["lname"]}',
                            cmap='PuOr',
                            plot_type='contourf',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            projection='PlateCarree',
                            bar=False,
                            plt_grid=True,
                            orientation='horizontal',
                            significance_mask=mask,
                            extend='both'
                            )

    an_type = 'JJAS'
    u_var_type = f'u_chi_an_{an_type}'
    v_var_type = f'v_chi_an_{an_type}'
    this_comp_ts_u = tu.get_sel_tps_ds(
        wind_cut[u_var_type], tps=tps)
    this_comp_ts_v = tu.get_sel_tps_ds(
        wind_cut[v_var_type], tps=tps)

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
                                  #   lw=40,
                                  #   scale=50,
                                  steps=3,
                                  key_length=1,
                                  key_loc=(0.9, -0.08)
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
                  intpol=False,
                  tick_step=2,
                  lw=3,
                  extend='both',
                  gl_plt=False)


cplt.add_colorbar(im=im_comp,
                  fig=im['fig'],
                  sci=3,
                  label=label,
                  x_pos=0.1,
                  y_pos=0.03,
                  height=0.03,
                  tick_step=2,
                  )
savepath = plot_dir +\
    f"{output_folder}/propagation/msf_{var_type}_all_regions.png"
cplt.save_fig(savepath, fig=im['fig'])
