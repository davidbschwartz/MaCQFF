import numpy as np
import torch
import networkx as nx
from random import sample
import time

from env_helper import coverage_oracle


def greedy_algorithm(density, graph, n_soln, disk_size):
    """Compute greedy solution on sum of UCB.

    Args:
        density (torch.Tensor Nx1): Upper confidence bound of density for whole domain
        graph (nx.DiGraph()): dynamics graph on which greedy solution needs to be evaluated e.g., pessi, opti
        n_soln (int): number of greedy solutions, typically number of agents
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location

    Returns:
        idx_x_curr (list): list of agent index
        M_dist (list): list of coverage gain at each location over the whole domain
        ____  (list): list is marginal gain for each agent
    """
    # Greedy algorithm initializations
    idx_x_curr = []
    M_dist = []
    record_margin_gain = []
    max_marginal_gain = 0  # if n_sol=0, we still need to pass this value
    non_covered_density = torch.empty_like(density).copy_(density)

    cum_time_greedy = 0
    runtimes_greedy = []

    # 1) Compute greedy solution, n_soln times
    for k in range(n_soln):

        st_greedy = time.process_time()

        # 1.1) Initialize
        idx_x_curr.append(0)
        M_dist.append(density)
        max_marginal_gain = -np.inf

        # 1.2) Compute marginal gain at each node (cell) of the domain and pick the node which give maximum marginal gain
        for node in graph.nodes:
            idx_x_curr[-1] = node  # filling in the last due to append
            marginal_gain = coverage_oracle(
                node, non_covered_density, graph, disk_size)
            M_dist[-1][node] = marginal_gain
            if marginal_gain > max_marginal_gain:
                max_marginal_gain = marginal_gain
                best_pos_k = node

        # 1.3) Save the greedy solution for kth agent and record marginal gain
        idx_x_curr[-1] = best_pos_k
        record_margin_gain.append(max_marginal_gain)

        # 1.4) Mark covered location as 0 (since they do not provide new information)
        non_covered_density[
            list(
                nx.single_source_shortest_path_length(
                    graph, idx_x_curr[-1], cutoff=disk_size
                )
            )
        ] = 0.0

        et_greedy = time.process_time()
        cum_time_greedy += et_greedy - st_greedy
        runtimes_greedy.append(cum_time_greedy)

    return idx_x_curr, M_dist, torch.sum(torch.stack(record_margin_gain)), runtimes_greedy


def stochastic_greedy_algorithm(density, graph, n_soln, disk_size):
    """Compute (approximate) greedy solution on sum of UCB.

    Args:
        density (torch.Tensor Nx1): Upper confidence bound of density for whole domain
        graph (nx.DiGraph()): dynamics graph on which greedy solution needs to be evaluated e.g., pessi, opti
        n_soln (int): number of greedy solutions, typically number of agents
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location

    Returns:
        idx_x_curr (list): list of agent index
        M_dist (list): list of coverage gain at each location over the whole domain
        ____  (list): list is marginal gain for each agent
    """

    # Greedy algorithm initializations
    idx_x_curr = []
    M_dist = []
    record_margin_gain = []
    max_marginal_gain = 0  # if n_sol=0, we still need to pass this value
    non_covered_density = torch.empty_like(density).copy_(density)

    epsilon = 0.05
    random_set_size = (len(graph.nodes)/n_soln) * np.log10(1/epsilon)
    effective_random_set_size = int(np.ceil(random_set_size))

    cum_time_stochastic_greedy = 0
    runtimes_stochastic_greedy = []

    # 1) Compute greedy solution, n_soln times
    for k in range(n_soln):

        st_stochastic_greedy = time.process_time()

        # 1.1) Initialize
        idx_x_curr.append(0)
        M_dist.append(density)
        max_marginal_gain = -np.inf

        # alternative 1.2) Create the random subset R and pick to node wich gives the highest marginal gain
        random_subset = sample(list(graph.nodes), effective_random_set_size)
        random_subgraph = graph.subgraph(random_subset)
        for node in random_subgraph.nodes:
            idx_x_curr[-1] = node  # filling in the last due to append
            marginal_gain = coverage_oracle(node, non_covered_density, graph, disk_size)
            M_dist[-1][node] = marginal_gain
            if marginal_gain > max_marginal_gain:
                max_marginal_gain = marginal_gain
                best_pos_k = node

        # 1.3) Save the greedy solution for kth agent and record marginal gain
        idx_x_curr[-1] = best_pos_k
        record_margin_gain.append(max_marginal_gain)

        # 1.4) Mark covered location as 0 (since they do not provide new information)
        non_covered_density[list(nx.single_source_shortest_path_length(graph, idx_x_curr[-1], cutoff=disk_size))] = 0.0

        et_stochastic_greedy = time.process_time()
        cum_time_stochastic_greedy += et_stochastic_greedy - st_stochastic_greedy
        runtimes_stochastic_greedy.append(cum_time_stochastic_greedy)

    return idx_x_curr, M_dist, torch.sum(torch.stack(record_margin_gain)), runtimes_stochastic_greedy


def greedy_algorithm_opti(density, graph, n_soln, disk_size):
    """Compute greedy solution on sum of UCB. (_opti) The code is optimized to not scale linearly with number of agents

    Args:
        density (torch.Tensor Nx1): Upper confidence bound of density for whole domain
        graph (nx.DiGraph()): dynamics graph on which greedy solution needs to be evaluated e.g., pessi, opti
        n_soln (int): number of greedy solutions, typically number of agents
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location

    Returns:
        idx_x_curr (list): list of agent index
        M_dist (list): list of coverage gain at each location over the whole domain
        ____  (list): list is marginal gain for each agent
    """
    # Greedy algorithm initializations
    idx_x_curr = []
    M_dist = []
    record_margin_gain = []
    non_covered_density = torch.empty_like(density).copy_(density)
    node_coverage_gain = {}

    cum_time_opti_greedy = 0
    runtimes_opti_greedy = []

    # 1) Compute marginal gain at each node (cell) of the domain

    st_opti_greedy_1 = time.process_time()

    for node in graph.nodes:
        marginal_gain = coverage_oracle(
            node, non_covered_density, graph, disk_size)
        node_coverage_gain[node] = marginal_gain

    et_opti_greedy_1 = time.process_time()
    time_first_iter = et_opti_greedy_1 - st_opti_greedy_1

    # 2) Compute greedy solution, n_soln times
    for k in range(n_soln):

        st_opti_greedy = time.process_time()

        idx = max(node_coverage_gain, key=node_coverage_gain.get)
        idx_x_curr.append(idx)
        record_margin_gain.append(node_coverage_gain[idx])
        M_dist.append(node_coverage_gain)

        # 2.2) Recompute marinal gain, given 'k' agents has already been picked. This method is optimized to not recompute the unaffected locations
        if k < n_soln - 1:

            # 2.2.1) Mark covered location as 0 (since they do not provide new information)
            non_covered_density[
                list(
                    nx.single_source_shortest_path_length(
                        graph, idx_x_curr[-1], cutoff=disk_size
                    )
                )
            ] = 0.0

            # 2.2.2) Compute affected locations coordinates. All the locations upto 2* radius of last picked agent is affected
            affected_locs = list(
                nx.single_source_shortest_path_length(
                    graph, idx_x_curr[-1], cutoff=2 * disk_size
                )
            )

            # 2.2.3) Recompute marinal gain, given 'k' agents has already been picked.
            for node in affected_locs:
                marginal_gain = coverage_oracle(
                    node, non_covered_density, graph, disk_size
                )
                node_coverage_gain[node] = marginal_gain
        
        et_opti_greedy = time.process_time()
        cum_time_opti_greedy += et_opti_greedy - st_opti_greedy
        runtimes_opti_greedy.append(cum_time_opti_greedy)

    runtimes_opti_greedy_final = [x+time_first_iter for x in runtimes_opti_greedy]

    return idx_x_curr, M_dist, torch.sum(torch.stack(record_margin_gain)), runtimes_opti_greedy_final
