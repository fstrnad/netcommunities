# %%
import geoutils.utils.general_utils as gut
import geoutils.plotting.plots as cplt
import geoutils.utils.time_utils as tu
import geoutils.tsa.event_synchronization as es
import geoutils.geodata.base_dataset as bds
import os
from importlib import reload
import numpy as np
PATH = os.path.dirname(os.path.abspath(__file__))

# Run the null model

vname = 'pr'
name = 'mswep'
start_month = 'Jun'
end_month = 'Sep'

output_dir = '/home/strnad/data/climnet/outputs/'
plot_dir = '/home/strnad/data/climnet/plots/'
data_dir = "/home/strnad/data/climnet/outputs/climate_data/"


# %%
# MSWEP precipitation
dataset_file = data_dir + 'mswep_pr_1_1979_2021_ds.nc'
reload(bds)

ds_mswep = bds.BaseDataset(load_nc=dataset_file,
                           )
# %%
# Define number of time points in JJAS
ts_jjas = tu.get_month_range_data(dataset=ds_mswep.ds,
                                  start_month=start_month,
                                  end_month=end_month)
time_steps = tu.get_num_tps(ts_jjas)

# %%
reload(tu)
reload(es)
n_pmts = 1000
taumax = 10
q_ee = 0.9
q_range = np.around(np.arange(0.9, 1, 0.001), decimals=3)
q_range = np.array([0.25, 0.5, 0.75, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999])

max_num_events = int(np.ceil(time_steps*(1-q_ee)))
null_model_folder = f'{PATH}/../../climnet/network/es_files/null_model/'
null_model_file = f'null_model_ts_{time_steps}_taumax_{taumax}_npmts_{n_pmts}_q_{q_ee}_directed.npy'
nmfp = null_model_folder + null_model_file
# %%
reload(es)

q_dict = es.null_model_distribution(length_time_series=time_steps,
                                    max_num_events=max_num_events,
                                    num_permutations=n_pmts,
                                    taumax=taumax,
                                    min_num_events=1,
                                    q=q_range,
                                    savepath=nmfp)

# %%
# plot Null model
reload(cplt)

n_pmts = 1000
taumax = 10
time_steps = 5124
q_ee = 0.9

null_model_folder = f'{PATH}/../../climnet/network/es_files/null_model/'
null_model_file = f'null_model_ts_{time_steps}_taumax_{taumax}_npmts_{n_pmts}_q_{q_ee}_directed.npy'

q_dict = gut.load_np_dict(null_model_folder+null_model_file)
q_map = q_dict[0.999]

cplt.plot_2D(x=np.arange(len(q_map[0])),
             y=np.arange(len(q_map[1])),
             z=q_map,
             xlabel='Number of events i',
             ylabel='Number of events j',
             plot_type='colormesh',
             label='Number of sync. events',
             orientation='vertical',
             levels=8,
             tick_step=2,
             extend='both'
             )
