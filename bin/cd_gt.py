# %%
import climnet.community_detection.graph_tool.es_graph_tool as egt
import climnet.network.clim_networkx as cn
import climnet.datasets.evs_dataset as eds
from importlib import reload
import geoutils.plotting.plots as cplt
import geoutils.utils.general_utils as gut
import numpy as np

output_dir = "/home/strnad/data/climnet/outputs/"
plot_dir = "/home/strnad/data/climnet/plots/"

# %%
# Load Network file EE
reload(eds)
name = "mswep"
grid_type = "fekete"
grid_step = 1

q_ee = .9
output_folder = "summer_monsoon"
name_prefix = f"{name}_{grid_type}_{grid_step}_{q_ee}"

start_month = "Jun"
end_month = "Sep"
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"

# %%
# Graph tool
job_id = gut.get_job_id()

# %%
reload(eds)
q_sig = 0.95
lon_range = [-80, 170]
lat_range = [-30, 50]
nx_path_file = output_dir + \
        f"{output_folder}/{name_prefix}_{q_sig}_lat_{lat_range}_ES_nx.gml.gz"
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_1979_2021_lat_{lat_range}_ds.nc"

ds = eds.EvsDataset(
    load_nc=dataset_file,
    rrevs=False
)
# %%
cnx = cn.Clim_NetworkX(dataset=ds, nx_path_file=nx_path_file)
# %%


# %%
# Init GT
reload(egt)
B_max = 10
num_runs = 10
for run in np.arange(num_runs):
    this_id = job_id*num_runs + run
    gut.myprint(f'Compute Run {this_id}!')
    sp_theta = (
        plot_dir
        + f"{output_folder}/graph_tool/{this_id}_{name_prefix}_{q_sig}_{B_max}.npy"
    )
    cplt.mk_plot_dir(sp_theta)

    graph_file = plot_dir + f"/graph_tool/{name_prefix}_{q_sig}.xml.gz"
    ds_gt = egt.ES_Graph_tool(
            network=cnx, graph_file=graph_file, weighted=False, rcg=True)

    theta = ds_gt.apply_SBM(
        g=ds_gt.graph, B_max=B_max, savepath=sp_theta, multi_level=False,
    )
    ds_gt.save_communities(sp_theta, theta)


# %%
B_max = 10

sp_theta = (
    plot_dir
    + f"{output_folder}/graph_tool/{1}_{name_prefix}_{q_sig}_{B_max}.npy"
)
theta, hard_cluster = ds_gt.load_communities(sp_theta)

# %%
# Plot one run

reload(cplt)
hc_map = cnx.ds.get_map(hard_cluster)
im = cplt.plot_map(
    hc_map,
    plot_type="discrete",
    cmap='Paired',
    title="Hard clustering Graph Tool",
    projection="PlateCarree",
    ds=cnx.ds,
    significance_mask=cnx.ds.mask,
    extend="neither",
    label="Group number",
    plt_grid=True,
    orientation="horizontal",
)
cplt.text_box(ax=im['ax'], text=f'{start_month}-{end_month}')
savepath = f"{plot_dir}/summer_monsoon/graph_tool/gt_{name_prefix}_{B_max}.png"
cplt.save_fig(savepath)
# %%
