import torch


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