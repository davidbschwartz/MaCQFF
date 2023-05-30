import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from env_helper import grid_world_graph
from greedy_helper import greedy_algorithm, stochastic_greedy_algorithm, greedy_algorithm_opti

def plot_time(n_soln, cum_time_greedy, cum_time_opti_greedy, cum_time_stochastic_greedy):

    iterations = np.arange(1, n_soln+1, 1)

    greedy_line, = plt.plot(iterations, cum_time_greedy, '--')
    greedy_line.set_label("Greedy Algorithm")
    greedy_line, = plt.plot(iterations, cum_time_opti_greedy, '--')
    greedy_line.set_label("Optimized Greedy Algorithm")
    stochastic_greedy_line, = plt.plot(iterations, cum_time_stochastic_greedy, '--')
    stochastic_greedy_line.set_label("Stochastic-Greedy Algorithm")

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative CPU time [s]")
    # _ = plt.title("Runtime Comparison of Greedy and Stochastic-Greedy")
    plt.savefig('stochastic-greedy.pdf', format='pdf')
    plt.show()

# init problem
num_agents = 50
disk_size = 0 # 0 to select a single node

# create domain
Nx = 500
Ny = 500
graph = grid_world_graph((Nx, Ny))
num_nodes = Nx * Ny

# create density
density = torch.rand(num_nodes)

# greedy algorithm
st_greedy = time.process_time()
sol_greedy, _, _, cum_time_greedy = greedy_algorithm(density.clone(), graph, num_agents, disk_size)
et_greedy = time.process_time()
cpu_time_greedy = et_greedy - st_greedy
print('Greedy solution:', sol_greedy)
print('CPU time of Greedy:', cpu_time_greedy, 's')
greedy_coverage = 0
for sol in sol_greedy:
    greedy_coverage += density[sol].item()
print("Greedy coverage:", greedy_coverage)

# stochastic greedy algorithm
st_stochastic_greedy = time.process_time()
sol_stochastic_greedy, _, _, cum_time_stochastic_greedy = stochastic_greedy_algorithm(density.clone(), graph, num_agents, disk_size)
et_stochastic_greedy = time.process_time()
cpu_time_stochastic_greedy = et_stochastic_greedy - st_stochastic_greedy
print('Stochastic-Greedy solution:', sol_stochastic_greedy)
print('CPU time of Stochastic-Greedy:', cpu_time_stochastic_greedy, 's')
stoch_greedy_coverage = 0
for sol in sol_stochastic_greedy:
    stoch_greedy_coverage += density[sol].item()
print("Stochastic-Greedy coverage:", stoch_greedy_coverage)

# optimized greedy algorithm
st_opti_greedy = time.process_time()
sol_opti_greedy, _, _, cum_time_opti_greedy = greedy_algorithm_opti(density.clone(), graph, num_agents, disk_size)
et_opti_greedy = time.process_time()
cpu_time_opti_greedy = et_opti_greedy - st_opti_greedy
print('Optimized Greedy solution:', sol_opti_greedy)
print('CPU time of Optimized Greedy:', cpu_time_opti_greedy, 's')

# plot the time performance
plot_time(num_agents, cum_time_greedy, cum_time_opti_greedy, cum_time_stochastic_greedy)