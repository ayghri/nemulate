import numpy as np


def stack_grid(grid_data):
    middle = np.c_[grid_data, grid_data, grid_data]
    symmetric = middle[::-1][::, ::-1]
    return np.r_[symmetric, middle, symmetric]


def stack_map(one_map):
    """
    Rearranges a 2D array by stacking and mirroring it.

    This function takes a 2D numpy array, reverses it, and concatenates
    sections to create a stacked output.

    Parameters:
    one_map (np.ndarray): A 2D numpy array to be processed.

    Returns:
    np.ndarray: A new 2D numpy array that is a stacked and mirrored version of the input.
    """
    mid = one_map
    top = np.concatenate([mid[:, 180:], mid[:, :180]], axis=1)[::-1]
    center_column = np.concatenate([top, mid, top], axis=0)
    stack = np.concatenate([center_column, center_column, center_column], axis=1)
    return stack
