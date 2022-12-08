# %%
from climnet.community_detection.graph_tool.es_graph_tool import ES_Graph_tool
import climnet.network.clim_networkx as cn
from climnet.datasets.evs_dataset import EvsDataset
from importlib import reload
import geoutils.plotting.plots as cplt
import geoutils.utils.general_utils as gut
# Run the null model
name = "mswep"
grid_type = "fekete"
grid_step = 1

scale = "global"
vname = "pr"

output_folder = "global_monsoon"
output_dir = "../outputs/"
plot_dir = "../outputs/images/"

name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}"

print("Loading Data")
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_ds.nc"
min_evs = 20
start_month = "Jun"
end_month = "Sep"
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"
    min_evs = 10

# %%
# Graph tool
# for job array
reload(gut)
job_id = gut.get_job_id()

B_max = 6
q_sig = 0.95

# %%
# Run job array on multiple networks
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
num_runs = 10

for run in range(num_runs):
    this_id = job_id*num_runs + run
    sp_theta = (
        plot_dir
        + f"{output_folder}/graph_tool/{this_id}_{name_prefix}_{q_sig}_{B_max}.npy"
    )
    gut.myprint(f'Compute {sp_theta}!')

    cplt.mk_plot_dir(sp_theta)
    ds_gt = ES_Graph_tool(
        network=cnx, graph_file=graph_file, weighted=False, rcg=True)

    theta = ds_gt.apply_SBM(
        g=ds_gt.graph, B_max=B_max, savepath=sp_theta,
        multi_level=False,
    )
    ds_gt.save_communities(sp_theta, theta)

