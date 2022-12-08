# %%
import os
import climnet.network.clim_networkx as nx
import climnet.datasets.evs_dataset as cds
from importlib import reload
import numpy as np
import geoutils.utils.time_utils as tu
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



# %%
reload(cds)
q_ee = 0.9
name_prefix = f"{name}_{scale}_{grid_type}_{grid_step}_{q_ee}"

start_month = "Jun"
end_month = "Sep"
min_evs = 20
if start_month != "Jan" or end_month != "Dec":
    name_prefix += f"_{start_month}_{end_month}"
    min_evs = 10
dataset_file = output_dir + f"/{output_folder}/{name_prefix}_ds.nc"
if gut.exist_file(dataset_file):
    ds = cds.EvsDataset(
        load_nc=dataset_file,
        rrevs=False,
    )
else:
    name_prefix_load = f"{name}_{scale}_{grid_type}_{grid_step}"
    dataset_file_load = output_dir + f"/{output_folder}/{name_prefix_load}_ds.nc"
    ds = cds.EvsDataset(
        load_nc=dataset_file_load,
        rrevs=True,
        month_range=[start_month, end_month],
        can=False,
        min_evs=min_evs,
        q=q_ee
    )
    ds.save(dataset_file)
    sys.exit(0)
# %%
taumax = 10
n_pmts = 1000
weighted = True
networkfile = output_dir + f"{output_folder}/{name_prefix}_ES_net.npz"

E_matrix_folder = (
    f"{output_folder}/{name_prefix}_{q_ee}_{min_evs}/"
)
time_steps = len(
    tu.get_month_range_data(
        ds.ds.time, start_month=start_month, end_month=end_month)
)
null_model_file = f'null_model_ts_{time_steps}_taumax_{taumax}_npmts_{n_pmts}_q_{q_ee}_directed.npy'
# %%
Net = EventSyncClimNet(ds, taumax=taumax, weighted=weighted,)
# %%

# q_sign_arr = np.around(np.arange(0.95, 1, 0.05), decimals=2)
# q_sign_arr = np.around(
#     np.array([0.25, 0.5, 0.75, 0.95, 0.98, 0.99, 0.995, 0.999]), decimals=3)
q_sign_arr = np.around(
    np.array([0.9, 0.91, 0.92, 0.93, 0.94, 0.95,
              0.96, 0.97, 0.98, 0.99, 0.995, 0.999]),
    decimals=3,
)
spars_arr = {q: None for q in q_sign_arr}
# %%
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
        print(f"Use q = {q_sig}", flush=True)
        Net.create(
            method='es',
            null_model_file=null_model_file,
            E_matrix_folder=E_matrix_folder,
            q_sig=q_sig
        )
        Net.save(nx_path_file)
# %%
sp_arr = plot_dir + \
    f"/null_model/compare_sign_thresh_density_gs_{grid_step}.npy"

