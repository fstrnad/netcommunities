# %%
import xarray as xr
from matplotlib.patches import Rectangle
import geoutils.utils.general_utils as gut
import climnet.community_detection.membership_likelihood as ml
import climnet.community_detection.graph_tool.es_graph_tool as egt
from climnet.community_detection.MTCOV.mtcov_climent import MTCOV_Climnet
import os
from climnet.datasets.evs_dataset import EvsDataset
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import geoutils.plotting.plots as cplt
import climnet.network.clim_networkx as cn

# Run the null model
name = "mswep"
grid_type = "fekete"
grid_step = 1

scale = "south_asia"
vname = "pr"

output_folder = "global_monsoon"
output_dir = "../outputs/"
plot_dir = "../outputs/images/"

q_ee = .9
q_sig = 0.95

name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"
start_month = "Jun"
end_month = "Sep"
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"


# %%
print("Loading Data")
graph_file = plot_dir + f"/graph_tool/{name_prefix}_{q_sig}.xml.gz"
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_ds.nc"
ds = EvsDataset(
    load_nc=dataset_file,
    rrevs=False
)
nx_graph_file = output_dir + \
    f"{output_folder}/{name_prefix}_{q_sig}_lb_ES_nx.gml.gz"
cnx = cn.Clim_NetworkX(dataset=ds, nx_path_file=nx_graph_file)

# %%
reload(egt)
cd = 'gt'
job_id = 0
if cd == 'gt':
    B_max = 6
    cd_folder = 'graph_tool'
    graph_file = plot_dir + f"/{cd_folder}/{name_prefix}_{q_sig}.xml.gz"
    sp_theta = (
        plot_dir +
        f"{output_folder}/{cd_folder}/{job_id}_{name_prefix}_{q_sig}_{B_max}.npy"
    )
    cplt.mk_plot_dir(sp_theta)
    ds_cd = egt.ES_Graph_tool(
        network=cnx, graph_file=graph_file, weighted=False, rcg=False)
elif cd == 'mtcov':
    B_max = 15
    cd_folder = 'MTCOV'
    sp_theta = (
        plot_dir +
        f"{output_folder}/MTCOV/{job_id}_{name_prefix}_{q_sig}_{B_max}.npy"
    )
    cplt.mk_plot_dir(sp_theta)
    ds_cd = MTCOV_Climnet(network=cnx, weighted=True,)


# %%
cnx.ds.add_loc_dict(
    name="EIO",
    lname='Equatorial Indian Ocean',
    lon_range=(60, 65),
    lat_range=(7, 10),
    color="tab:red",
    n_rep_ids=3,
    reset_loc_dict=True
)

cnx.ds.add_loc_dict(
    name="SA",
    lname='South Asia',
    lon_range=(75, 80),
    lat_range=[15, 25],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="NIC",
    lname='North India-China',
    lon_range=(110, 120),
    lat_range=[32, 37],
    color="tab:red",
    n_rep_ids=3,
)


cnx.ds.add_loc_dict(
    name="WP",
    lname='West Pacific',
    lon_range=(130, 135),
    lat_range=[20, 30],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="MC",
    lname='Maritime Continent',
    lon_range=(100, 135),
    lat_range=[-10, 0],
    color="tab:red",
    n_rep_ids=3,
)

cnx.ds.add_loc_dict(
    name="BoB",
    lname='Bay of Bengal',
    lon_range=(82, 92),
    lat_range=[8, 13],
    color="tab:red",
    n_rep_ids=3,
)

# %%
n_idx_list = []
region_dict = cnx.ds.loc_dict
for mname, mregion in region_dict.items():
    n_idx_list.append(mregion["ids"])
loc_map = cnx.ds.get_map_for_idx(np.concatenate(n_idx_list))
ax = cplt.plot_map(
    loc_map,
    ds=cnx.ds,
    projection="PlateCarree",
    plt_grid=True,
    plot_type="points",
    color="blue",
    levels=2,
    vmin=0,
    vmax=1,
    title="Selected Rep Ids",
    bar=False,
    alpha=1,
)
savepath = plot_dir + \
    f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_pids.png"

# cplt.save_fig(savepath)

# %%
# Get group numbers for rep ids
B_max = 6
sp_arr = []

for run in np.arange(110):
    sp_theta = (
        plot_dir +
        f"{output_folder}/{cd_folder}/{run}_{name_prefix}_{q_sig}_{B_max}.npy"
    )
    if os.path.exists(sp_theta):
        sp_arr.append(sp_theta)

ds_cd.load_sp_arr(sp_arr=sp_arr)
# %%
reload(cplt)
for run in [0, 1, 2, 100]:
    # for run in [0]:
    sp_theta = (
        plot_dir +
        f"{output_folder}/{cd_folder}/{run}_{name_prefix}_{q_sig}_{B_max}.npy"
    )
    theta, hard_cluster = ds_cd.load_communities(sp_theta)
    hc_map = cnx.ds.get_map(hard_cluster)
    im = cplt.plot_map(
        cnx.ds,
        hc_map,
        ds_mask=True,
        plot_type="discrete",
        # title=f"Run Number {run}",
        # title=f"Run Number {100}",
        projection="PlateCarree",
        extend="neither",
        bar=False,
        label="Group number",
        plt_grid=True,
        orientation="horizontal",
    )
    cplt.plt_text(ax=im["ax"], text=f"Run number {run}", fsize=16,
                  xpos=0.38, ypos=0.9,
                  geoaxis=True)

    savepath = (
        plot_dir
        + f"{output_folder}/{cd_folder}/{run}_hard_clusters_{name_prefix}_{B_max}_{q_sig}.png"
    )
    cplt.save_fig(savepath=savepath)

# %%
reload(ml)
reload(gut)
reload(cplt)

res_dict = ml.get_regions_data_prob_map(
    ds_cd=ds_cd,
    all_region_dict=region_dict,
    sig_th=0.75,
    exclude_outlayers=True,
    inrange=True,
)
savepath = (
    plot_dir +
    f"{output_folder}/{cd_folder}/{name_prefix}_{q_sig}_{B_max}_prob_maps.npy"
)
gut.save_np_dict(res_dict, savepath)

# %%
reload(cplt)
region = 'MC'
ax = cplt.plot_map(
    xr.where(res_dict[region]['map_range']==1, 1, np.nan),
    # res_dict[region]['map'],
    ds=cnx.ds,
    projection='PlateCarree',
    plt_grid=True,
    plot_type='contourf',
    significance_mask=True,
    # cmap='Reds',
    color='blue',
    cmap=None,
    levels=10, vmin=0, vmax=1,
    bar=False,
    #    alpha=1
)

# %%
reload(cplt)
prob_map = res_dict[region]['prob_map']
gr_map = cnx.ds.get_map(prob_map)

im = cplt.plot_map(
    gr_map,
    ds=ds_cd.ds,
    significance_mask=True,
    plot_type="contourf",
    cmap="Reds",
    levels=10,
    vmin=0,
    vmax=1,
    # title=f"Probability map for q={q_sig} {region}",
    projection="PlateCarree",
    extend="neither",
    bar=True,
    plt_grid=True,
    label="Membership Likelihood",
    orientation="horizontal",
    round_dec=3,
)
# cplt.text_box(ax=im["ax"], text=f"{start_month}-{end_month}")

savepath = (
    plot_dir
    + f"{output_folder}/{cd_folder}/msl_{region}_{q_sig}_{B_max}.png"
)
cplt.save_fig(savepath)

# %%
# Standard Deviation
prob_map_std = res_dict[region]['prob_map_std']
gr_map = cnx.ds.get_map(prob_map_std)
# gr_map = Net.ds.get_map(np.where(prob_map_std > 0.4, prob_map_std, prob_map_std-0.1))

im = cplt.plot_map(
    gr_map,
    ds=cnx.ds,
    plot_type="contourf",
    significance_mask=True,
    cmap="Reds",
    levels=10,
    vmin=0,
    vmax=1,
    title=f"Standard Deviation for q={q_sig} sign threshold",
    projection="PlateCarree",
    extend="neither",
    bar=True,
    plt_grid=True,
    label="Standard Deviation",
    orientation="horizontal",
    round_dec=3,
)

savepath = (
    plot_dir +
    f"{output_folder}/{cd_folder}/gt_ensemble_std_{name_prefix}_{q_sig}.png"
)

plt.savefig(savepath, bbox_inches="tight")


# %%
# Plot Msl of all communities
reload(cplt)
ax = None
legend_items = []
legend_item_names = []
regions = list(res_dict.keys())
for idx, region in enumerate(regions):

    this_map = res_dict[region]["map"]
    color = cplt.colors[idx]
    im = cplt.plot_map(
        this_map,
        ax=ax,
        projection='PlateCarree',
        figsize=(9, 7),
        plt_grid=True,
        plot_type='contour',
        color=color,
        levels=0, vmin=0, vmax=1,
        bar=False,
        lw=3,
        alpha=1,
        set_map=False if idx > 0 else True)

    im = cplt.plot_map(
        xr.where(this_map == 1, 1, np.nan),
        # this_map,
        ds=ds_cd.ds,
        significance_mask=True,
        ax=im['ax'],
        projection='PlateCarree',
        plt_grid=False,
        plot_type='contourf',
        color=color,
        cmap=None,
        levels=2,
        vmin=0, vmax=1,
        bar=False,
        alpha=0.6
    )
    ax = im['ax']
    legend_items.append(Rectangle((0, 0), 1, 1,
                                  fc=color, alpha=0.5,
                                  fill=True,
                                  edgecolor=color,
                                  linewidth=2))
    legend_item_names.append(f"{region}")

cplt.set_legend(ax=im['ax'],
                legend_items=legend_items,
                label_arr=legend_item_names,
                loc='outside',
                ncol_legend=3,
                box_loc=(0, 0))

savepath = (
    plot_dir +
    f"{output_folder}/{cd_folder}/msl_all_regions_{q_sig}.png"
)
# cplt.save_fig(savepath=savepath)
# %%
reload(cplt)
nrows = 2
ncols = 3
im = cplt.create_multi_plot(nrows=nrows,
                            ncols=ncols,
                            projection='PlateCarree',
                            orientation='horizontal',
                            hspace=0.25,
                            wspace=0.15,
                            end_idx=len(regions))

for idx, region in enumerate(regions):

    # EE TS
    this_dict = res_dict[region]
    prob_map = this_dict['prob_map']
    gr_map = cnx.ds.get_map(prob_map)

    im_comp = cplt.plot_map(
        ds_cd.ds,
        gr_map,
        ax=im['ax'][idx],
        plot_type="contourf",
        title=f'{this_dict["lname"]}',
        ds_mask=True,
        cmap="Reds",
        levels=10,
        vmin=0,
        vmax=1,
        # title=f"Probability map for q={q_sig} {region}",
        projection="PlateCarree",
        extend="neither",
        bar=False,
        plt_grid=True,
    )


cbar = cplt.add_colorbar(im=im_comp,
                         fig=im['fig'],
                         x_pos=0.2,
                         y_pos=0.05, width=0.6, height=0.02,
                         orientation='horizontal',
                         label='Membership Likelihood',
                         round_dec=2,
                         tick_step=1,
                         )

savepath = (
    plot_dir +
    f"{output_folder}/{cd_folder}/msl_single_regions_{q_sig}.png"
)
cplt.save_fig(savepath,
              fig=im['fig'])
