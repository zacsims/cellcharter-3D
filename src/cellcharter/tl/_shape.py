from __future__ import annotations

import io
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import sknw
from anndata import AnnData
from anndata._io.specs.registry import _REGISTRY, IOSpec
from h5py import Dataset, Group
from matplotlib.path import Path
from rasterio import features
from scipy.spatial import Delaunay
from shapely import geometry, wkb
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union
from skimage.morphology import skeletonize
from squidpy._docs import d


# ============================================================================
# Dimension Detection Utility
# ============================================================================


def _detect_ndim(adata: AnnData, spatial_key: str = "spatial", ndim: int | None = None) -> int:
    """
    Detect the dimensionality of spatial coordinates.

    Parameters
    ----------
    adata
        AnnData object containing spatial coordinates.
    spatial_key
        Key in adata.obsm where spatial coordinates are stored.
    ndim
        If provided, use this value instead of auto-detecting.

    Returns
    -------
    int
        The number of spatial dimensions (2 or 3).

    Raises
    ------
    ValueError
        If spatial coordinates are not found or have invalid dimensionality.
    """
    if ndim is not None:
        if ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
        return ndim

    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial key '{spatial_key}' not found in adata.obsm")

    coords = adata.obsm[spatial_key]
    detected_ndim = coords.shape[1]

    if detected_ndim not in (2, 3):
        raise ValueError(f"Spatial coordinates must have 2 or 3 dimensions, got {detected_ndim}")

    return detected_ndim


def _get_spatial_coords(
    adata: AnnData, mask: np.ndarray, spatial_key: str = "spatial", ndim: int | None = None
) -> np.ndarray:
    """
    Get spatial coordinates for a subset of cells, handling 2D/3D appropriately.

    Parameters
    ----------
    adata
        AnnData object containing spatial coordinates.
    mask
        Boolean mask or index array to select cells.
    spatial_key
        Key in adata.obsm where spatial coordinates are stored.
    ndim
        Number of dimensions to use. If None, uses all available dimensions.

    Returns
    -------
    np.ndarray
        Spatial coordinates for the selected cells.
    """
    coords = adata.obsm[spatial_key][mask]
    if ndim is not None:
        return coords[:, :ndim]
    return coords

# ============================================================================
# Serialization: 2D Polygons (WKB format)
# ============================================================================

# 1. Define a custom encoding spec
polygon_spec = IOSpec(encoding_type="polygon", encoding_version="1.0.0")


# 2. Writer: Polygon → WKB → uint8 vlen array (object-path)
@_REGISTRY.register_write(Group, Polygon, polygon_spec)
def _write_polygon(group: Group, key: str, poly: Polygon, *, _writer, dataset_kwargs):
    # 2.1 Serialize to WKB bytes
    raw: bytes = wkb.dumps(poly)
    # 2.2 View as a 1D uint8 array
    arr: np.ndarray = np.frombuffer(raw, dtype=np.uint8)
    # 2.3 Create a vlen dtype over uint8
    dt = h5py.special_dtype(vlen=np.dtype("uint8"))
    # 2.4 Create an empty length-1 dataset with that dtype
    dset = group.create_dataset(key, shape=(1,), dtype=dt)
    # 2.5 Assign element-wise to invoke object-path conversion
    dset[0] = arr


# 3. Reader: uint8 vlen array → bytes → Polygon
@_REGISTRY.register_read(Dataset, polygon_spec)
def _read_polygon(dataset: Dataset, *, _reader) -> Polygon:
    # 3.1 dataset[0] returns the inner uint8 array
    arr: np.ndarray = dataset[0]
    # 3.2 Recover raw WKB and load
    return wkb.loads(arr.tobytes())


# ============================================================================
# Serialization: 3D Meshes (PLY format via trimesh)
# ============================================================================


class Mesh3D:
    """
    A wrapper class for 3D mesh objects to enable AnnData serialization.

    This class wraps a trimesh.Trimesh object and provides serialization
    via the PLY format. The wrapper is needed because AnnData's IOSpec
    registry requires specific types to be registered.

    Parameters
    ----------
    mesh : trimesh.Trimesh, optional
        The trimesh object to wrap. If None, creates an empty mesh.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        The underlying trimesh object.
    vertices : np.ndarray
        Mesh vertices (N x 3 array).
    faces : np.ndarray
        Mesh faces (M x 3 array of vertex indices).
    """

    def __init__(self, mesh=None):
        if mesh is None:
            try:
                import trimesh

                self.mesh = trimesh.Trimesh()
            except ImportError:
                self.mesh = None
        else:
            self.mesh = mesh

    @property
    def vertices(self) -> np.ndarray:
        """Get mesh vertices."""
        return self.mesh.vertices if self.mesh is not None else np.array([])

    @property
    def faces(self) -> np.ndarray:
        """Get mesh faces."""
        return self.mesh.faces if self.mesh is not None else np.array([])

    @classmethod
    def from_trimesh(cls, trimesh_obj) -> "Mesh3D":
        """Create a Mesh3D from a trimesh.Trimesh object."""
        return cls(mesh=trimesh_obj)

    def to_bytes(self) -> bytes:
        """Serialize mesh to PLY format bytes."""
        if self.mesh is None:
            return b""
        buffer = io.BytesIO()
        self.mesh.export(buffer, file_type="ply")
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Mesh3D":
        """Deserialize mesh from PLY format bytes."""
        if len(data) == 0:
            return cls()
        try:
            import trimesh

            buffer = io.BytesIO(data)
            mesh = trimesh.load(buffer, file_type="ply")
            return cls(mesh=mesh)
        except ImportError:
            warnings.warn("trimesh not installed. Cannot load 3D mesh.", stacklevel=2)
            return cls()

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are inside the mesh.

        Parameters
        ----------
        points
            Array of shape (N, 3) with 3D coordinates.

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,) indicating containment.
        """
        if self.mesh is None or len(self.vertices) == 0:
            return np.zeros(len(points), dtype=bool)
        return self.mesh.contains(points)

    @property
    def volume(self) -> float:
        """Get mesh volume."""
        if self.mesh is None or not self.mesh.is_watertight:
            return 0.0
        return self.mesh.volume

    @property
    def area(self) -> float:
        """Get mesh surface area."""
        if self.mesh is None:
            return 0.0
        return self.mesh.area


# 4. Define a custom encoding spec for 3D meshes
mesh3d_spec = IOSpec(encoding_type="mesh3d", encoding_version="1.0.0")


# 5. Writer: Mesh3D → PLY bytes → uint8 vlen array
@_REGISTRY.register_write(Group, Mesh3D, mesh3d_spec)
def _write_mesh3d(group: Group, key: str, mesh: Mesh3D, *, _writer, dataset_kwargs):
    # 5.1 Serialize to PLY bytes
    raw: bytes = mesh.to_bytes()
    # 5.2 View as a 1D uint8 array
    arr: np.ndarray = np.frombuffer(raw, dtype=np.uint8)
    # 5.3 Create a vlen dtype over uint8
    dt = h5py.special_dtype(vlen=np.dtype("uint8"))
    # 5.4 Create an empty length-1 dataset with that dtype
    dset = group.create_dataset(key, shape=(1,), dtype=dt)
    # 5.5 Assign element-wise to invoke object-path conversion
    dset[0] = arr


# 6. Reader: uint8 vlen array → bytes → Mesh3D
@_REGISTRY.register_read(Dataset, mesh3d_spec)
def _read_mesh3d(dataset: Dataset, *, _reader) -> Mesh3D:
    # 6.1 dataset[0] returns the inner uint8 array
    arr: np.ndarray = dataset[0]
    # 6.2 Recover raw PLY bytes and load
    return Mesh3D.from_bytes(arr.tobytes())


# ============================================================================
# 2D Alpha Shape Computation
# ============================================================================


def _alpha_shape(coords, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.

    Adapted from `here <https://web.archive.org/web/20200726174718/http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/>`_.

    Parameters
    ----------
    coords : np.array
        Array of 2D coordinates of points (N x 2).
    alpha : float
        Alpha value to influence the gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers. Too large, and you lose
        everything!

    Returns
    -------
    concave_hull : shapely.geometry.Polygon
        Concave hull of the points.
    triangles : list
        List of triangles in the alpha shape.
    edge_points : np.ndarray
        Array of edge point indices.
    """
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < alpha]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0)  # .tolist()
    m = geometry.MultiLineString(edge_points.tolist())
    triangles = list(polygonize(m.geoms))
    return unary_union(triangles), triangles, edge_points


# ============================================================================
# 3D Alpha Shape Computation
# ============================================================================


def _alpha_shape_3d(coords: np.ndarray, alpha: float | None = None) -> Mesh3D:
    """
    Compute the 3D alpha shape (concave hull) of a set of 3D points.

    Uses the alphashape library which handles alpha parameter optimization.

    Parameters
    ----------
    coords : np.ndarray
        Array of 3D coordinates of points (N x 3).
    alpha : float, optional
        Alpha value for the alpha shape. If None, uses automatic optimization
        from the alphashape library.

    Returns
    -------
    Mesh3D
        A Mesh3D object containing the alpha shape boundary.

    Notes
    -----
    Requires the alphashape and trimesh packages to be installed.
    Install with: pip install "cellcharter[3d]"
    """
    try:
        import alphashape
        import trimesh
    except ImportError as e:
        raise ImportError(
            "3D alpha shape computation requires 'alphashape' and 'trimesh'. "
            "Install them with:\n    pip install 'cellcharter[3d]'"
        ) from e

    # Compute alpha shape
    if alpha is None:
        # Let alphashape optimize the alpha parameter
        alpha_shape = alphashape.alphashape(coords, 0)  # 0 = convex hull as starting point
    else:
        alpha_shape = alphashape.alphashape(coords, alpha)

    # Convert to trimesh
    if hasattr(alpha_shape, "exterior"):
        # It's a shapely polygon (shouldn't happen in 3D but just in case)
        raise ValueError("alphashape returned a 2D polygon for 3D input")

    if isinstance(alpha_shape, trimesh.Trimesh):
        mesh = alpha_shape
    else:
        # Try to convert from vertices/faces if it's a different format
        try:
            mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces)
        except AttributeError:
            raise ValueError(f"Unexpected alpha shape type: {type(alpha_shape)}")

    return Mesh3D.from_trimesh(mesh)


def _process_component_3d(
    points: np.ndarray, component, alpha: float | None = None, use_convex_hull: bool = False
) -> tuple:
    """
    Process a single 3D component to compute its boundary.

    Parameters
    ----------
    points : np.ndarray
        Array of 3D coordinates (N x 3).
    component
        Component identifier.
    alpha : float, optional
        Alpha parameter for alpha shape. If None, uses automatic optimization.
    use_convex_hull : bool, default=False
        If True, use convex hull instead of alpha shape (faster but less accurate).

    Returns
    -------
    tuple
        (component, Mesh3D boundary)
    """
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "3D boundary computation requires 'trimesh'. " "Install with:\n    pip install 'cellcharter[3d]'"
        ) from e

    if len(points) < 4:
        # Need at least 4 points for a 3D hull
        warnings.warn(f"Component {component} has fewer than 4 points, skipping boundary computation.", stacklevel=2)
        return component, Mesh3D()

    if use_convex_hull:
        # Use convex hull (faster but less accurate)
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(points)
            # Use all points as vertices and hull.simplices as faces
            # (simplices index into the original points array)
            mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
            return component, Mesh3D.from_trimesh(mesh)
        except Exception as e:
            warnings.warn(f"Convex hull computation failed for component {component}: {e}", stacklevel=2)
            return component, Mesh3D()
    else:
        # Use alpha shape
        try:
            return component, _alpha_shape_3d(points, alpha=alpha)
        except Exception as e:
            warnings.warn(f"Alpha shape computation failed for component {component}: {e}. Falling back to convex hull.", stacklevel=2)
            # Fall back to convex hull
            try:
                from scipy.spatial import ConvexHull

                hull = ConvexHull(points)
                # Use all points as vertices and hull.simplices as faces
                mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
                return component, Mesh3D.from_trimesh(mesh)
            except Exception:
                return component, Mesh3D()


def _process_component_2d(points, component, hole_area_ratio=0.1, alpha_start=2000):
    """
    Process a single 2D component to compute its boundary.

    Parameters
    ----------
    points : np.ndarray
        Array of 2D coordinates (N x 2).
    component
        Component identifier.
    hole_area_ratio : float
        Minimum ratio between the area of a hole and the area of the boundary.
    alpha_start : int
        Starting value for the alpha parameter.

    Returns
    -------
    tuple
        (component, Polygon boundary)
    """
    alpha = alpha_start
    polygon, triangles, edge_points = _alpha_shape(points, alpha)

    while (
        type(polygon) is not geometry.polygon.Polygon
        or type(polygon) is geometry.MultiPolygon
        or edge_points.shape[0] < 10
    ):
        alpha *= 2
        polygon, triangles, edge_points = _alpha_shape(points, alpha)

    boundary_with_holes = max(triangles, key=lambda triangle: triangle.area)
    boundary = polygon

    for interior in boundary_with_holes.interiors:
        interior_polygon = geometry.Polygon(interior)
        hole_to_boundary_ratio = interior_polygon.area / boundary.area
        if hole_to_boundary_ratio > hole_area_ratio:
            try:
                difference = boundary.difference(interior_polygon)
                if isinstance(difference, geometry.Polygon):
                    boundary = difference
            except Exception:  # noqa: B902
                pass
    return component, boundary


# Legacy alias for backward compatibility
def _process_component(points, component, hole_area_ratio=0.1, alpha_start=2000):
    """Deprecated alias for _process_component_2d."""
    return _process_component_2d(points, component, hole_area_ratio, alpha_start)


@d.dedent
def boundaries(
    adata: AnnData,
    cluster_key: str = "component",
    min_hole_area_ratio: float = 0.1,
    alpha_start: int = 2000,
    spatial_key: str = "spatial",
    ndim: int | None = None,
    alpha_3d: float | None = None,
    use_convex_hull: bool = False,
    copy: bool = False,
) -> None | dict:
    """
    Compute the topological boundaries of sets of cells.

    Automatically detects 2D or 3D data based on the spatial coordinates.
    For 2D data, computes alpha shape polygons. For 3D data, computes alpha
    shape meshes using the alphashape library.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    min_hole_area_ratio
        Minimum ratio between the area of a hole and the area of the boundary.
        Only used for 2D data.
    alpha_start
        Starting value for the alpha parameter of the alpha shape algorithm.
        Only used for 2D data.
    spatial_key
        Key in :attr:`anndata.AnnData.obsm` where the spatial coordinates are stored.
    ndim
        Number of dimensions (2 or 3). If None, automatically detected from
        the spatial coordinates.
    alpha_3d
        Alpha parameter for 3D alpha shape computation. If None, uses automatic
        optimization. Only used for 3D data.
    use_convex_hull
        If True, use convex hull instead of alpha shape for 3D data. Faster but
        less accurate. Only used for 3D data.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys
    and the boundaries as values (Polygon for 2D, Mesh3D for 3D).

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['boundary']`` - the above mentioned :class:`dict`.

    Notes
    -----
    For 3D data, requires the alphashape and trimesh packages:
        pip install "cellcharter[3d]"
    """
    # Detect dimensionality
    detected_ndim = _detect_ndim(adata, spatial_key=spatial_key, ndim=ndim)

    if detected_ndim == 2:
        return _boundaries_2d(
            adata,
            cluster_key=cluster_key,
            min_hole_area_ratio=min_hole_area_ratio,
            alpha_start=alpha_start,
            spatial_key=spatial_key,
            copy=copy,
        )
    else:
        return _boundaries_3d(
            adata,
            cluster_key=cluster_key,
            spatial_key=spatial_key,
            alpha=alpha_3d,
            use_convex_hull=use_convex_hull,
            copy=copy,
        )


def _boundaries_2d(
    adata: AnnData,
    cluster_key: str = "component",
    min_hole_area_ratio: float = 0.1,
    alpha_start: int = 2000,
    spatial_key: str = "spatial",
    copy: bool = False,
) -> None | dict[int, geometry.Polygon]:
    """
    Compute 2D topological boundaries using alpha shapes.

    Internal function called by boundaries() for 2D data.
    """
    assert 0 <= min_hole_area_ratio <= 1, "min_hole_area_ratio must be between 0 and 1"
    assert alpha_start > 0, "alpha_start must be greater than 0"

    clusters = [cluster for cluster in adata.obs[cluster_key].unique() if cluster != "-1" and not pd.isnull(cluster)]

    result_boundaries = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _process_component_2d,
                adata.obsm[spatial_key][adata.obs[cluster_key] == cluster, :2],
                cluster,
                min_hole_area_ratio,
                alpha_start,
            ): cluster
            for cluster in clusters
        }

        for future in as_completed(futures):
            component, boundary = future.result()
            result_boundaries[component] = boundary

    if copy:
        return result_boundaries

    adata.uns[f"shape_{cluster_key}"] = {"boundary": result_boundaries}


def _boundaries_3d(
    adata: AnnData,
    cluster_key: str = "component",
    spatial_key: str = "spatial",
    alpha: float | None = None,
    use_convex_hull: bool = False,
    copy: bool = False,
) -> None | dict[int, Mesh3D]:
    """
    Compute 3D topological boundaries using alpha shapes.

    Internal function called by boundaries() for 3D data.

    Parameters
    ----------
    adata
        AnnData object with spatial coordinates.
    cluster_key
        Key in adata.obs for cluster labels.
    spatial_key
        Key in adata.obsm for spatial coordinates.
    alpha
        Alpha parameter for alpha shape. None for auto-optimization.
    use_convex_hull
        If True, use convex hull instead of alpha shape.
    copy
        If True, return boundaries dict instead of modifying adata.

    Returns
    -------
    dict or None
        Dictionary mapping cluster labels to Mesh3D objects if copy=True.
    """
    clusters = [cluster for cluster in adata.obs[cluster_key].unique() if cluster != "-1" and not pd.isnull(cluster)]

    result_boundaries = {}

    # Process each component (not using ProcessPoolExecutor for 3D due to
    # potential issues with trimesh serialization across processes)
    for cluster in clusters:
        points = adata.obsm[spatial_key][adata.obs[cluster_key] == cluster, :3]
        component, boundary = _process_component_3d(
            points, cluster, alpha=alpha, use_convex_hull=use_convex_hull
        )
        result_boundaries[component] = boundary

    if copy:
        return result_boundaries

    adata.uns[f"shape_{cluster_key}"] = {"boundary": result_boundaries}


def _find_dangling_branches(graph, total_length, min_ratio=0.05):
    total_length = np.sum(list(nx.get_edge_attributes(graph, "weight").values()))
    adj = nx.to_numpy_array(graph, weight=None)
    adj_w = nx.to_numpy_array(graph)

    n_neighbors = np.sum(adj, axis=1)
    node_total_dist = np.sum(adj_w, axis=1)
    dangling_nodes = np.argwhere((node_total_dist < min_ratio * total_length) & (n_neighbors == 1))
    if dangling_nodes.shape[0] != 1:
        dangling_nodes = dangling_nodes.squeeze()
    else:
        dangling_nodes = dangling_nodes[0]
    return dangling_nodes


def _remove_dangling_branches(graph, min_ratio=0.05):
    total_length = np.sum(list(nx.get_edge_attributes(graph, "weight").values()))

    dangling_branches = _find_dangling_branches(graph, total_length=total_length, min_ratio=min_ratio)

    while len(dangling_branches) > 0:
        idx2node = dict(enumerate(graph.nodes))
        for i in dangling_branches:
            graph.remove_node(idx2node[i])

        dangling_branches = _find_dangling_branches(graph, total_length=total_length, min_ratio=min_ratio)


def _longest_path_from_node(graph, u):
    visited = dict.fromkeys(graph.nodes)
    distance = {i: -1 for i in list(graph.nodes)}
    idx2node = dict(enumerate(graph.nodes))

    try:
        adj_lil = nx.to_scipy_sparse_matrix(graph, format="lil")
    except AttributeError:
        adj_lil = nx.to_scipy_sparse_array(graph, format="lil")
    adj = {i: [idx2node[neigh] for neigh in neighs] for i, neighs in zip(graph.nodes, adj_lil.rows)}
    weight = nx.get_edge_attributes(graph, "weight")

    distance[u] = 0
    queue = deque()
    queue.append(u)
    visited[u] = True
    while queue:
        front = queue.popleft()
        for i in adj[front]:
            if not visited[i]:
                visited[i] = True
                source, target = min(i, front), max(i, front)
                distance[i] = distance[front] + weight[(source, target)]
                queue.append(i)

    farthest_node = max(distance, key=distance.get)

    longest_path_length = distance[farthest_node]
    return farthest_node, longest_path_length


def _longest_path_length(graph):
    # first DFS to find one end point of longest path
    node, _ = _longest_path_from_node(graph, list(graph.nodes)[0])
    # second DFS to find the actual longest path
    _, longest_path_length = _longest_path_from_node(graph, node)
    return longest_path_length


# ============================================================================
# 2D Shape Metric Helpers
# ============================================================================


def _linearity(boundary, height=1000, min_ratio=0.05):
    """Compute 2D linearity from a polygon boundary."""
    img, _ = _rasterize(boundary, height=height)
    skeleton = skeletonize(img).astype(int)

    graph = sknw.build_sknw(skeleton.astype(np.uint16))
    graph = graph.to_undirected()

    _remove_dangling_branches(graph, min_ratio=min_ratio)

    cycles = nx.cycle_basis(graph)
    cycles_len = [nx.path_weight(graph, cycle + [cycle[0]], "weight") for cycle in cycles]

    longest_path_length = _longest_path_length(graph)
    longest_length = np.max(cycles_len + [longest_path_length])

    return longest_length / np.sum(list(nx.get_edge_attributes(graph, "weight").values()))


# ============================================================================
# 3D Shape Metric Helpers
# ============================================================================


def _elongation_3d(points: np.ndarray) -> float:
    """
    Compute 3D elongation using PCA eigenvalue ratio.

    Parameters
    ----------
    points : np.ndarray
        Array of 3D coordinates (N x 3).

    Returns
    -------
    float
        Elongation score between 0 (sphere) and 1 (line).
    """
    if len(points) < 4:
        return 0.0

    # Center the points
    centered = points - points.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Get eigenvalues (sorted in ascending order)
    eigenvalues = np.linalg.eigvalsh(cov)

    # Sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Elongation is 1 - (smallest / largest eigenvalue)
    if eigenvalues[0] == 0:
        return 0.0

    return 1 - eigenvalues[2] / eigenvalues[0]


def _purity_3d(
    boundary: Mesh3D, cluster_points: np.ndarray, all_points: np.ndarray, cluster_labels: np.ndarray, cluster_id
) -> float:
    """
    Compute 3D purity using mesh containment.

    Parameters
    ----------
    boundary : Mesh3D
        The 3D mesh boundary.
    cluster_points : np.ndarray
        Points belonging to the cluster.
    all_points : np.ndarray
        All points in the sample.
    cluster_labels : np.ndarray
        Cluster labels for all points.
    cluster_id
        The cluster identifier.

    Returns
    -------
    float
        Purity score (ratio of cluster cells within boundary to total cells within boundary).
    """
    if boundary.mesh is None or len(boundary.vertices) == 0:
        return np.nan

    try:
        # Check which points are inside the mesh
        within_mask = boundary.contains(all_points)

        if np.sum(within_mask) == 0:
            return np.nan

        # Compute purity
        return np.sum(cluster_labels[within_mask] == cluster_id) / np.sum(within_mask)
    except Exception:
        return np.nan


def _is_3d_boundary(boundary) -> bool:
    """Check if a boundary is a 3D mesh or 2D polygon."""
    return isinstance(boundary, Mesh3D)


def _rasterize(boundary, height=1000):
    minx, miny, maxx, maxy = boundary.bounds
    poly = shapely.affinity.translate(boundary, -minx, -miny)
    if maxx - minx > maxy - miny:
        scale_factor = height / poly.bounds[2]
    else:
        scale_factor = height / poly.bounds[3]
    poly = shapely.affinity.scale(poly, scale_factor, scale_factor, origin=(0, 0, 0))
    return features.rasterize([poly], out_shape=(height, int(height * (maxx - minx) / (maxy - miny)))), scale_factor


def linearity(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "linearity",
    height: int = 1000,
    min_ratio: float = 0.05,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the linearity of the topological boundaries of sets of cells.

    This function is deprecated. Please use `linearity_metric` instead.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    height
        Height of the rasterized image.
    min_ratio
        Minimum ratio between the length of a branch and the total length of the skeleton.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the linearity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.
    """
    warnings.warn(
        "linearity is deprecated and will be removed in the next release. " "Please use `linearity_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return linearity_metric(
        adata=adata,
        cluster_key=cluster_key,
        out_key=out_key,
        height=height,
        min_ratio=min_ratio,
        copy=copy,
    )


@d.dedent
def linearity_metric(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "linearity",
    height: int = 1000,
    min_ratio: float = 0.05,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the linearity of the topological boundaries of sets of cells.

    For 2D data, rasterizes the polygon and computes the skeleton of the
    rasterized image. Then, computes the longest path in the skeleton and
    divides it by the total length of the skeleton.

    For 3D data, this metric is not yet implemented and will raise a warning.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    height
        Height of the rasterized image. The width is computed automatically to
        preserve the aspect ratio of the polygon. Higher values lead to more
        precise results but also higher memory usage. Only used for 2D data.
    min_ratio
        Minimum ratio between the length of a branch and the total length of the
        skeleton to be considered a real branch and not be removed. Only used for 2D data.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the linearity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.

    Notes
    -----
    For 3D data, linearity (tubularity) metric is not yet implemented.
    Clusters with 3D boundaries will have NaN linearity values.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    linearity_score = {}
    has_3d = False
    for cluster, boundary in boundaries.items():
        if _is_3d_boundary(boundary):
            has_3d = True
            linearity_score[cluster] = np.nan
        else:
            linearity_score[cluster] = _linearity(boundary, height=height, min_ratio=min_ratio)

    if has_3d:
        warnings.warn(
            "Linearity metric is not yet implemented for 3D data. "
            "3D clusters will have NaN linearity values.",
            UserWarning,
            stacklevel=2,
        )

    if copy:
        return linearity_score

    adata.uns[f"shape_{cluster_key}"][out_key] = linearity_score


def _elongation(boundary):
    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*boundary.minimum_rotated_rectangle.exterior.coords.xy))

    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [geometry.LineString((mbr_points[i], mbr_points[i + 1])).length for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    minor_axis = min(mbr_lengths)
    major_axis = max(mbr_lengths)
    return 1 - minor_axis / major_axis


def elongation(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "elongation",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the elongation of the topological boundaries of sets of cells.

    This function is deprecated. Please use `elongation_metric` instead.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the elongation as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.
    """
    warnings.warn(
        "elongation is deprecated and will be removed in the next release. " "Please use `elongation_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return elongation_metric(
        adata=adata,
        cluster_key=cluster_key,
        out_key=out_key,
        copy=copy,
    )


@d.dedent
def elongation_metric(
    adata: AnnData,
    cluster_key: str = "component",
    spatial_key: str = "spatial",
    out_key: str = "elongation",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the elongation of the topological boundaries of sets of cells.

    For 2D data, computes the minimum bounding rectangle and divides the length
    of the minor axis by the length of the major axis.

    For 3D data, uses PCA eigenvalue ratios to measure elongation.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    spatial_key
        Key in :attr:`anndata.AnnData.obsm` where the spatial coordinates are stored.
        Used for 3D elongation computation.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the elongation as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    elongation_score = {}
    for cluster, boundary in boundaries.items():
        if _is_3d_boundary(boundary):
            # 3D: use PCA-based elongation on point cloud
            points = adata.obsm[spatial_key][adata.obs[cluster_key] == cluster, :3]
            elongation_score[cluster] = _elongation_3d(points)
        else:
            # 2D: use minimum bounding rectangle
            elongation_score[cluster] = _elongation(boundary)

    if copy:
        return elongation_score
    adata.uns[f"shape_{cluster_key}"][out_key] = elongation_score


def _axes(boundary):
    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*boundary.minimum_rotated_rectangle.exterior.coords.xy))
    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [geometry.LineString((mbr_points[i], mbr_points[i + 1])).length for i in range(len(mbr_points) - 1)]
    return min(mbr_lengths), max(mbr_lengths)


def _curl(boundary):
    factor = boundary.length**2 - 16 * boundary.area
    if factor < 0:
        factor = 0
    fibre_length = boundary.area / ((boundary.length - np.sqrt(factor)) / 4)

    _, length = _axes(boundary)
    if fibre_length < length:
        return 0
    else:
        return 1 - length / fibre_length


def curl(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "curl",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the curl score of the topological boundaries of sets of cells.

    This function is deprecated. Please use `curl_metric` instead.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the curl score as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.
    """
    warnings.warn(
        "curl is deprecated and will be removed in the next release. " "Please use `curl_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return curl_metric(
        adata=adata,
        cluster_key=cluster_key,
        out_key=out_key,
        copy=copy,
    )


@d.dedent
def curl_metric(
    adata: AnnData,
    cluster_key: str = "component",
    out_key: str = "curl",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the curl score of the topological boundaries of sets of cells.

    For 2D data, computes the curl score of each cluster as one minus the ratio
    between the length of the major axis of the minimum bounding rectangle and
    the fiber length of the polygon.

    For 3D data, this metric is not yet implemented and will raise a warning.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the curl score as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.

    Notes
    -----
    For 3D data, curl (tortuosity) metric is not yet implemented.
    Clusters with 3D boundaries will have NaN curl values.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]
    curl_score = {}
    has_3d = False
    for cluster, boundary in boundaries.items():
        if _is_3d_boundary(boundary):
            has_3d = True
            curl_score[cluster] = np.nan
        else:
            curl_score[cluster] = _curl(boundary)

    if has_3d:
        warnings.warn(
            "Curl metric is not yet implemented for 3D data. "
            "3D clusters will have NaN curl values.",
            UserWarning,
            stacklevel=2,
        )

    if copy:
        return curl_score
    adata.uns[f"shape_{cluster_key}"][out_key] = curl_score


def purity(
    adata: AnnData,
    cluster_key: str = "component",
    library_key: str = "sample",
    out_key: str = "purity",
    exterior: bool = False,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the purity of the topological boundaries of sets of cells.

    This function is deprecated. Please use `purity_metric` instead.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    library_key
        Key in :attr:`anndata.AnnData.obs` where the sample labels are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    exterior
        If ``True``, the computation of the purity ignores the polygon's internal holes.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the purity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.
    """
    warnings.warn(
        "purity is deprecated and will be removed in the next release. " "Please use `purity_metric` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return purity_metric(
        adata=adata,
        cluster_key=cluster_key,
        library_key=library_key,
        out_key=out_key,
        exterior=exterior,
        copy=copy,
    )


@d.dedent
def purity_metric(
    adata: AnnData,
    cluster_key: str = "component",
    library_key: str = "sample",
    spatial_key: str = "spatial",
    out_key: str = "purity",
    exterior: bool = False,
    copy: bool = False,
) -> None | dict[int, float]:
    """
    Compute the purity of the topological boundaries of sets of cells.

    It computes the purity of each cluster as the ratio between the number of
    cells of the cluster that are within the boundary and the total number of
    cells within the boundary.

    For 2D data, uses point-in-polygon tests.
    For 3D data, uses mesh containment tests via trimesh.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels are stored.
    library_key
        Key in :attr:`anndata.AnnData.obs` where the sample labels are stored.
    spatial_key
        Key in :attr:`anndata.AnnData.obsm` where the spatial coordinates are stored.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    exterior
        If ``True``, the computation of the purity ignores the polygon's internal holes.
        Only used for 2D data.
    %(copy)s

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the purity as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - the above mentioned :class:`dict`.
    """
    boundaries = adata.uns[f"shape_{cluster_key}"]["boundary"]

    purity_score = {}
    for cluster, boundary in boundaries.items():
        sample = adata[adata.obs[cluster_key] == cluster].obs[library_key][0]
        adata_sample = adata[adata.obs[library_key] == sample]

        if _is_3d_boundary(boundary):
            # 3D: use mesh containment
            points = adata_sample.obsm[spatial_key][:, :3]
            purity_score[cluster] = _purity_3d(
                boundary,
                adata.obsm[spatial_key][adata.obs[cluster_key] == cluster, :3],
                points,
                adata_sample.obs[cluster_key].values,
                cluster,
            )
        else:
            # 2D: use point-in-polygon
            points = adata_sample.obsm[spatial_key][:, :2]
            within_mask = np.zeros(points.shape[0], dtype=bool)
            if type(boundary) is geometry.multipolygon.MultiPolygon:
                for p in boundary.geoms:
                    path = Path(np.array(p.exterior.coords.xy).T)
                    within_mask |= np.array(path.contains_points(points))
            else:
                path = Path(np.array(boundary.exterior.coords.xy).T)
                within_mask |= np.array(path.contains_points(points))
                if not exterior:
                    for interior in boundary.interiors:
                        path = Path(np.array(interior.coords.xy).T)
                        within_mask &= ~np.array(path.contains_points(points))

            purity_score[cluster] = np.sum(adata_sample.obs[cluster_key][within_mask] == cluster) / np.sum(within_mask)

    if copy:
        return purity_score
    adata.uns[f"shape_{cluster_key}"][out_key] = purity_score


@d.dedent
def relative_component_size_metric(
    adata: AnnData,
    neighborhood_key: str,
    cluster_key: str = "component",
    library_key: str | None = None,
    out_key: str = "rcs",
    copy: bool = False,
) -> None | dict[int, float]:
    """
    The Relative Component Size (RCS) metric compares a component's cell count to the average component size in its cellular neighborhood, indicating whether it is larger or smaller than expected given the neighborhood's total cells and component count.

    Parameters
    ----------
    %(adata)s
    neighborhood_key
        Key in :attr:`anndata.AnnData.obs` where the neighborhood labels are stored.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` where the cluster labels from cc.gr.connected_components are stored.
    library_key
        Key in :attr:`anndata.AnnData.obs` where the sample or condition labels are stored. If None, the average is computed across all samples.
    out_key
        Key in :attr:`anndata.AnnData.obs` where the metric values are stored if ``copy = False``.
    %(copy)s
    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the cluster labels as keys and the RCS as values.

    Otherwise, modifies the ``adata`` with the following key:
        - :attr:`anndata.AnnData.uns` ``['shape_{{cluster_key}}']['{{out_key}}']`` - - the above mentioned :class:`dict`.
    """
    count = adata.obs[cluster_key].value_counts().to_dict()
    df = pd.DataFrame(count.items(), columns=[cluster_key, "count"])
    df = pd.merge(df, adata.obs[[cluster_key, neighborhood_key]].drop_duplicates().dropna(), on=cluster_key)

    if library_key is not None:
        df = pd.merge(df, adata.obs[[cluster_key, library_key]].drop_duplicates().dropna(), on=cluster_key)
        group_by = [library_key, neighborhood_key]
    else:
        group_by = [neighborhood_key]

    nbh_counts = adata.obs.groupby(group_by, observed=False).size().reset_index(name="total_neighborhood_cells_image")
    df = df.merge(nbh_counts, on=group_by, how="left")
    unique_counts = (
        adata.obs.groupby(group_by, observed=False)[cluster_key]
        .nunique()
        .reset_index()
        .rename(columns={cluster_key: "unique_components_neighborhood_image"})
    )
    df = df.merge(unique_counts, on=group_by, how="left")

    df["rcs"] = df["count"] / (df["total_neighborhood_cells_image"] / df["unique_components_neighborhood_image"])

    if copy:
        return df.set_index(cluster_key)["rcs"].to_dict()
    adata.uns[f"shape_{cluster_key}"][out_key] = df.set_index(cluster_key)["rcs"].to_dict()
