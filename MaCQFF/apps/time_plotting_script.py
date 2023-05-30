import numpy as np
import matplotlib.pyplot as plt

import pickle


"""
Assumptions: 
            - the number of agents used in both methods is the same for one .pkl file
            - the number of envs is the same and they are named the same
            - there are 2 param folders: smcc_MacOpt_GP and smcc_MaCQFF
"""

workspace = "MaCQFF"
time_save_path = workspace + "/experiments/GP/environments/"


def plot_time(T_macopt, T_macqff, avg_iter_times_macopt, avg_iter_times_macqff):

    iterations_macopt = np.arange(1, T_macopt+1, 1) # include the 0th point?
    iterations_macqff = np.arange(1, T_macqff+1, 1)

    macopt_line, = plt.plot(iterations_macopt, avg_iter_times_macopt, '--')
    macopt_line.set_label('MaCOpt')
    macqff_line, = plt.plot(iterations_macqff, avg_iter_times_macqff, '--')
    macqff_line.set_label('MaCQFF')

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("CPU time [s]")
    plt.savefig("time.pdf")
    plt.close()
    #plt.show()


def plot_cum_time(T_macopt, T_macqff, avg_cum_times_macopt, avg_cum_times_macqff):

    iterations_macopt = np.arange(1, T_macopt+1, 1) # include the 0th point?
    iterations_macqff = np.arange(1, T_macqff+1, 1)

    macopt_line, = plt.plot(iterations_macopt, avg_cum_times_macopt, '--')
    macopt_line.set_label('MaCOpt')
    macqff_line, = plt.plot(iterations_macqff, avg_cum_times_macqff, '--')
    macqff_line.set_label('MaCQFF')

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative CPU time [s]")
    plt.savefig("cum_time.pdf")
    plt.close()
    #plt.show()


with open(time_save_path + '0_time_data.pkl', 'rb') as fp1:
    time_dict = pickle.load(fp1)

with open(time_save_path + '0_cum_time_data.pkl', 'rb') as fp2:
    cum_time_dict = pickle.load(fp2)

# with open("MaCQFF" + "/experiments/GP/environments/env_0/smcc_MaCQFF/" + 'time_dict.pkl', 'rb') as fp:
#     time_dict = pickle.load(fp)

num_agents = len(time_dict['env_0']['smcc_MaCQFF']['0'].keys())

list_of_arrays_macopt = []
list_of_arrays_macqff = []

# CPU time per iteration
for env in time_dict.keys():
    for params in time_dict[env].keys():
        if params == 'smcc_MacOpt_GP':
            for i in time_dict[env][params].keys():
                for agent in time_dict[env][params][i].keys():
                    if agent == 0:
                        avg_iter_time_macopt = np.array(time_dict[env][params][i][0])
                    else:
                        avg_iter_time_macopt += np.array(time_dict[env][params][i][agent])
                avg_iter_time_macopt = avg_iter_time_macopt / num_agents
                list_of_arrays_macopt.append(avg_iter_time_macopt)
        elif params == 'smcc_MaCQFF':
            for i in time_dict[env][params].keys():
                for agent in time_dict[env][params][i].keys():
                    if agent == 0:
                        avg_iter_time_macqff = np.array(time_dict[env][params][i][0])
                    else:
                        avg_iter_time_macqff += np.array(time_dict[env][params][i][agent])
                avg_iter_time_macqff = avg_iter_time_macqff / num_agents
                list_of_arrays_macqff.append(avg_iter_time_macqff)

# CPU times per iteration of MaCOpt
max_num_iters_macopt = max(len(array) for array in list_of_arrays_macopt)
iter_times_macopt = []
for array in list_of_arrays_macopt:
    processed_array = np.pad(array, (0, max_num_iters_macopt-len(array)), constant_values=np.nan)
    iter_times_macopt.append(processed_array)
avg_iter_time_macopt = np.nanmean(np.array(iter_times_macopt), axis=0)

# CPU times per iteration of MaCQFF
max_num_iters_macqff = max(len(array) for array in list_of_arrays_macqff)
iter_times_macqff = []
for array in list_of_arrays_macqff:
    processed_array = np.pad(array, (0, max_num_iters_macqff-len(array)), constant_values=np.nan)
    iter_times_macqff.append(processed_array)
avg_iter_time_macqff = np.nanmean(np.array(iter_times_macqff), axis=0)

# cumulative CPU time
list_of_arrays_macopt_2 = []
list_of_arrays_macqff_2 = []

for env in cum_time_dict.keys():
    for params in cum_time_dict[env].keys():
        if params == 'smcc_MacOpt_GP':
            for i in cum_time_dict[env][params].keys():
                for agent in cum_time_dict[env][params][i].keys():
                    if agent == 0:
                        avg_cum_time_macopt = np.array(cum_time_dict[env][params][i][0])
                    else:
                        avg_cum_time_macopt += np.array(cum_time_dict[env][params][i][agent])
                avg_cum_time_macopt = avg_cum_time_macopt / num_agents
                list_of_arrays_macopt_2.append(avg_cum_time_macopt)
        elif params == 'smcc_MaCQFF':
            for i in cum_time_dict[env][params].keys():
                for agent in cum_time_dict[env][params][i].keys():
                    if agent == 0:
                        avg_cum_time_macqff = np.array(cum_time_dict[env][params][i][0])
                    else:
                        avg_cum_time_macqff += np.array(cum_time_dict[env][params][i][agent])
                avg_cum_time_macqff = avg_cum_time_macqff / num_agents
                list_of_arrays_macqff_2.append(avg_cum_time_macqff)

# cumulative CPU times per iteration of MaCOpt
max_num_iters_macopt = max(len(array) for array in list_of_arrays_macopt_2)
cum_times_macopt = []
for array in list_of_arrays_macopt_2:
    processed_array = np.pad(array, (0, max_num_iters_macopt-len(array)), constant_values=np.nan)
    cum_times_macopt.append(processed_array)
avg_cum_time_macopt = np.nanmean(np.array(cum_times_macopt), axis=0)

# cumulative CPU times per iteration of MaCQFF
max_num_iters_macqff = max(len(array) for array in list_of_arrays_macqff_2)
cum_times_macqff = []
for array in list_of_arrays_macqff_2:
    processed_array = np.pad(array, (0, max_num_iters_macqff-len(array)), constant_values=np.nan)
    cum_times_macqff.append(processed_array)
avg_cum_time_macqff = np.nanmean(np.array(cum_times_macqff), axis=0)

plot_time(max_num_iters_macopt, max_num_iters_macqff, avg_iter_time_macopt, avg_iter_time_macqff)
plot_cum_time(max_num_iters_macopt, max_num_iters_macqff, avg_cum_time_macopt, avg_cum_time_macqff)