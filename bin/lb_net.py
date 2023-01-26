# %%
import climnet.network.clim_networkx as nx
import os
import climnet.datasets.evs_dataset as eds
from importlib import reload


name = "mswep"
grid_type = "fekete"
grid_step = 1

output_folder = "summer_monsoon"


output_dir = "/home/strnad/data/climnet/outputs/"
plot_dir = "/home/strnad/data/climnet/plots/"


# %%
q_ee = 0.9
name_prefix = f"{name}_{grid_type}_{grid_step}_{q_ee}"

start_month = "Jun"
end_month = "Sep"
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_1979_2021_ds.nc"


# %%
reload(nx)
reload(eds)
nn_lb = None
num_rand_permutations = 1000
q_sig = 0.95

networkfile = (
    output_dir + f"{output_folder}/{name_prefix}_{q_sig}_ES_nx.gml.gz"
)

networkfile_lb = (
    output_dir +
    f"{output_folder}/{name_prefix}_{q_sig}_lb_ES_nx.gml.gz"
)

if os.path.exists(networkfile):
    print(f"Use q sign = {q_sig}")
    ds = eds.EvsDataset(
        load_nc=dataset_file,
        rrevs=False,
    )
    if os.path.exists(networkfile_lb):
        cnx = nx.Clim_NetworkX(ds,
                               nx_path_file=networkfile_lb)
    else:
        cnx = nx.Clim_NetworkX(ds,
                               nx_path_file=networkfile)

        lb_folder = f"{name_prefix}_{num_rand_permutations}/"
        cnx.link_bundles(
            num_rand_permutations=num_rand_permutations,
            nn_points_bw=nn_lb,
            link_bundle_folder=lb_folder,
        )
        cnx.save(networkfile_lb)
    cnx.compute_link_lengths_edges()
    # cnx.compute_curvature(c_type='forman')
    # cnx.compute_network_attrs('degree', rc_attr=True)
    cnx.save(networkfile_lb)
else:
    print(f"Networkfile {networkfile} does not exist!")

# %%
