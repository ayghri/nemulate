from icomesh.mesh import Mesh
from numpy import typing as ntp

# from icomesh.utils import lonlat_to_cartesian
# from sklearn.neighbors import NearestNeighbors


def transfer_to_mesh(
    mesh: Mesh,
    source_lonlat: ntp.NDArray,
    source_vals: ntp.NDArray,
    n_neighbors: int = 5,
):
    """

    Args:
        source_grid: shape (K, 2) longitude, latitude
        source_vals: shape (K, D1, D2...,Dd)
        target_grid: shape (M, 2) target longitude, latitude or (M,N,2)
        n_neighbors: int
    Returns:
        (M,D1,D2..) or (M,N,D1...)

    """

    # (K,) = source_lonlat.shape[:1]
    # Ds = source_vals.shape[1:]
    assert source_lonlat.shape[:1] == source_vals.shape[:1]
    # target_shape = target_grid.shape[:-1]
    # nearest_idx, _ = mesh_map(source_lonlat, target_grid, n_neighbors)
    nearest_idx, _ = mesh.nearest_nodes_to_mesh_nodes(
        source_lonlat, n_neighbors=n_neighbors
    )
    # print(nearest_idx.shape, source_vals.shape, source_lonlat.shape)
    gather_source_vals = source_vals[nearest_idx]  #
    # print(gather_source_vals.shape)
    target_vals = gather_source_vals.mean(1)
    # print(target_vals.shape)
    return target_vals


def transfer_from_mesh(
    mesh: Mesh,
    target_lonlat: ntp.NDArray,
    mesh_vals: ntp.NDArray,
    n_neighbors: int = 5,
):
    """

    Args:
        source_grid: shape (K, 2) longitude, latitude
        source_vals: shape (K, D1, D2...,Dd)
        target_grid: shape (M, 2) target longitude, latitude or (M,N,2)
        n_neighbors: int
    Returns:
        (M,D1,D2..) or (M,N,D1...)

    """

    # (K,) = target_lonlat.shape[:1]
    # Ds = mesh_vals.shape[1:]
    # assert target_lonlat.shape[:1] == mesh_vals.shape[:1]
    # target_shape = target_grid.shape[:-1]
    # nearest_idx, _ = mesh_map(source_lonlat, target_grid, n_neighbors)
    nearest_idx, _ = mesh.nearest_mesh_nodes_to_target_nodes(
        target_lonlat, n_neighbors=n_neighbors
    )
    # print(nearest_idx.shape, target_lonlat.shape, mesh_vals.shape)
    gather_source_vals = mesh_vals[nearest_idx]  #
    target_vals = gather_source_vals.mean(1)
    # print(gather_source_vals.shape, target_vals.shape)
    return target_vals

    # return source_vals[nearest_idx]


# def mesh_map(
#     source_grid: ntp.NDArray,
#     target_grid: ntp.NDArray,
#     n_neighbors: int = 5,
# ):
#     """
#
#     Args:
#         source_grid: shape (K, 2) longitude, latitude
#         target_grid: shape (M, 2) target longitude, latitude or (M,N,2)
#         n_neighbors: int
#     Returns:
#         (M, n_neighbors)
#
#     """
#
#     (K,) = source_grid.shape[:1]
#     (M,) = target_grid.shape[:1]
#     source_vectors = lonlat_to_cartesian(source_grid.reshape(K, 2))
#     target_vectors = lonlat_to_cartesian(target_grid.reshape(M, 2))
#     neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(source_vectors)
#     nearest_dist, nearest_idx = neigh.kneighbors(target_vectors)
#     return nearest_idx, nearest_dist
