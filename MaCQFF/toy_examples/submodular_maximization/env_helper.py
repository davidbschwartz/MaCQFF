import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx


def nodes_to_states(nodes, world_shape, step_size):
    """Convert node numbers to physical states.
    Parameters
    ----------
    nodes: np.array
        Node indices of the grid world
    world_shape: tuple
        The size of the grid_world
    step_size: np.array
        The step size of the grid world
    Returns
    -------
    states: np.array
        The states in physical coordinates
    """
    nodes = torch.as_tensor(nodes)
    step_size = torch.as_tensor(step_size)
    return (
        torch.vstack(
            ((nodes // world_shape["y"]), (nodes % world_shape["y"]))).T
        * step_size
    )


def grid(world_shape, step_size, start_loc):
    """
    Creates grids of coordinates and indices of state space
    Parameters
    ----------
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Phyiscal step size in the grid world
    Returns
    -------
    states_ind: np.array
        (n*m) x 2 array containing the indices of the states
    states_coord: np.array
        (n*m) x 2 array containing the coordinates of the states
    """
    nodes = torch.arange(0, world_shape["x"] * world_shape["y"])
    return nodes_to_states(nodes, world_shape, step_size) + start_loc

class CentralGraph:
    def __init__(self, env_params) -> None:
        self.Nx = env_params["shape"]["x"]
        self.Ny = env_params["shape"]["y"]
        self.graph = grid_world_graph((self.Nx, self.Ny))
        self.base_graph = grid_world_graph((self.Nx, self.Ny))

    def UpdateSafeInGraph(self):
        return 1


def expansion_operator(graph, true_constraint, init_node, thresh, Lc):
    # print("init_node", init_node)
    # Total safet set
    total_safe_nodes = torch.arange(0, true_constraint.shape[0])[
        true_constraint > thresh
    ]
    total_safe_nodes = torch.cat([total_safe_nodes, init_node.reshape(-1)])
    total_safe_nodes = torch.unique(total_safe_nodes)
    total_safe_graph = graph.subgraph(total_safe_nodes.numpy())
    edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(
        total_safe_graph, init_node.item()
    )

    connected_nodes = [init_node.item()] + [v for u, v in edges]
    reachable_safe_graph = graph.subgraph(np.asarray(connected_nodes))

    return reachable_safe_graph


def grid_world_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # action 1: go right
    graph.add_edges_from(
        zip(grid_nodes[:, :-1].reshape(-1), grid_nodes[:, 1:].reshape(-1)), action=1
    )

    # action 2: go down
    graph.add_edges_from(
        zip(grid_nodes[:-1, :].reshape(-1), grid_nodes[1:, :].reshape(-1)), action=2
    )

    # action 3: go left
    graph.add_edges_from(
        zip(grid_nodes[:, 1:].reshape(-1), grid_nodes[:, :-1].reshape(-1)), action=3
    )

    # action 4: go up
    graph.add_edges_from(
        zip(grid_nodes[1:, :].reshape(-1), grid_nodes[:-1, :].reshape(-1)), action=4
    )

    return graph


def diag_grid_world_graph(world_size):
    """Create a graph that represents a grid world.
    In the grid world there are four actions, (1, 2, 3, 4), which correspond
    to going (up, right, down, left) in the x-y plane. The states are
    ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
    corresponds to a matrix where increasing the row index corresponds to the
    x direction in the graph, and increasing y index corresponds to the y
    direction.
    Parameters
    ----------
    world_size: tuple
        The size of the grid world (rows, columns)
    Returns
    -------
    graph: nx.DiGraph()
        The directed graph representing the grid world.
    """
    nodes = np.arange(np.prod(world_size))
    grid_nodes = nodes.reshape(world_size)

    graph = nx.DiGraph()

    # action 1: go right
    graph.add_edges_from(
        zip(grid_nodes[:, :-1].reshape(-1), grid_nodes[:, 1:].reshape(-1)), action=1
    )

    # action 2: go down
    graph.add_edges_from(
        zip(grid_nodes[:-1, :].reshape(-1), grid_nodes[1:, :].reshape(-1)), action=2
    )

    # action 3: go left
    graph.add_edges_from(
        zip(grid_nodes[:, 1:].reshape(-1), grid_nodes[:, :-1].reshape(-1)), action=3
    )

    # action 4: go up
    graph.add_edges_from(
        zip(grid_nodes[1:, :].reshape(-1), grid_nodes[:-1, :].reshape(-1)), action=4
    )

    graph.add_edges_from(
        zip(grid_nodes[:-1, :-1].reshape(-1), grid_nodes[1:, 1:].reshape(-1)), action=5
    )
    graph.add_edges_from(
        zip(grid_nodes[:-1, 1:].reshape(-1), grid_nodes[1:, :-1].reshape(-1)), action=6
    )
    graph.add_edges_from(
        zip(grid_nodes[1:, :-1].reshape(-1), grid_nodes[:-1, 1:].reshape(-1)), action=7
    )
    graph.add_edges_from(
        zip(grid_nodes[1:, 1:].reshape(-1), grid_nodes[:-1, :-1].reshape(-1)), action=8
    )

    return graph

def coverage_oracle(idx_x_curr, density, graph, disk_size):
    """_summary_ This method computes upper bound of the coverage using sum of ucb of density

    Args:
        idx_x_curr (list): node of current agent whose coverage upper bound is querried
        density (torch.Tensor Nx1): Upper confidence bound of density for whole domain
        graph (nx.DiGraph()): dynamics graph for getting list of covered nodes
        disk_size (int): maximum number of connecting edges that an agent can travel from its current location

    Returns:
        Fs (torch.Tensor 1x1): upper bound of the coverage function at the node
    """
    # 1) Get a list of nodes being covered
    coverage_area_idx = list(
        nx.single_source_shortest_path_length(
            graph, idx_x_curr, cutoff=disk_size)
    )

    # 2) \sum \mu + \sum \sigma_ii
    Fs = torch.sum(density[coverage_area_idx])
    return Fs
