"""3D visualization module for CellCharter spatial data.

This module provides PyVista-based 3D visualization functions for
spatial clustering results and boundaries.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from squidpy._docs import d

from ..tl._shape import Mesh3D


def _check_pyvista():
    """Check if PyVista is installed."""
    try:
        import pyvista as pv

        return pv
    except ImportError as e:
        raise ImportError(
            "3D visualization requires 'pyvista'. Install it with:\n"
            "    pip install 'cellcharter[vis3d]'"
        ) from e


@d.dedent
def spatial_3d(
    adata: AnnData,
    color: str | None = None,
    spatial_key: str = "spatial",
    size: float = 5.0,
    opacity: float = 1.0,
    cmap: str = "viridis",
    background: str = "white",
    show_axes: bool = True,
    title: str | None = None,
    save: str | Path | None = None,
    off_screen: bool = False,
    return_plotter: bool = False,
    **kwargs,
):
    """
    Create a 3D scatter plot of cells colored by a feature.

    Parameters
    ----------
    %(adata)s
    color
        Key in ``adata.obs`` or ``adata.var_names`` to color cells by.
        If None, all cells are colored the same.
    spatial_key
        Key in ``adata.obsm`` where the 3D spatial coordinates are stored.
    size
        Size of the points.
    opacity
        Opacity of the points (0-1).
    cmap
        Colormap to use for continuous values or categorical palette name.
    background
        Background color of the plot.
    show_axes
        Whether to show the axes.
    title
        Title of the plot.
    save
        Path to save the plot (supports PNG, JPG, SVG, PDF).
    off_screen
        If True, render off-screen (for saving without display).
    return_plotter
        If True, return the PyVista plotter object instead of showing the plot.
    **kwargs
        Additional arguments passed to ``pyvista.Plotter.add_points``.

    Returns
    -------
    pyvista.Plotter or None
        If ``return_plotter=True``, returns the plotter object.
        Otherwise, displays the plot and returns None.

    Examples
    --------
    >>> import cellcharter as cc
    >>>
    >>> # Plot cells colored by cluster
    >>> cc.pl.spatial_3d(adata, color="cluster", spatial_key="spatial_3d")
    >>>
    >>> # Save to file
    >>> cc.pl.spatial_3d(adata, color="cluster", save="spatial_3d.png")
    """
    pv = _check_pyvista()

    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm")

    coords = adata.obsm[spatial_key]
    if coords.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates, got {coords.shape[1]} dimensions")

    # Create point cloud
    point_cloud = pv.PolyData(coords)

    # Get color values
    if color is not None:
        if color in adata.obs.columns:
            values = adata.obs[color].values
        elif color in adata.var_names:
            values = adata[:, color].X.flatten()
        else:
            raise KeyError(f"Color key '{color}' not found in adata.obs or adata.var_names")

        # Handle categorical data
        if pd.api.types.is_categorical_dtype(values) or isinstance(values[0], str):
            categories = pd.Categorical(values)
            scalars = categories.codes.astype(float)
            # Map -1 (NaN) to NaN
            scalars[scalars == -1] = np.nan
            point_cloud["scalars"] = scalars
            annotations = {i: cat for i, cat in enumerate(categories.categories)}
        else:
            point_cloud["scalars"] = values
            annotations = None
    else:
        point_cloud["scalars"] = np.ones(len(coords))
        annotations = None

    # Create plotter
    plotter = pv.Plotter(off_screen=off_screen or save is not None)
    plotter.set_background(background)

    # Add points
    plotter.add_points(
        point_cloud,
        scalars="scalars",
        point_size=size,
        opacity=opacity,
        cmap=cmap,
        render_points_as_spheres=True,
        **kwargs,
    )

    if annotations is not None:
        # Add legend for categorical data
        legend_entries = [(str(cat), cmap) for cat in annotations.values()]
        if len(legend_entries) <= 20:  # Only show legend if not too many categories
            try:
                plotter.add_legend(legend_entries[:10])  # Limit to 10 entries
            except Exception:
                pass  # Legend might fail for some colormaps

    if show_axes:
        plotter.show_axes()

    if title is not None:
        plotter.add_title(title)

    if save is not None:
        plotter.screenshot(str(save))
        if not off_screen and not return_plotter:
            plotter.close()
            return None

    if return_plotter:
        return plotter

    plotter.show()
    return None


@d.dedent
def boundaries_3d(
    adata: AnnData,
    cluster_key: str = "component",
    spatial_key: str = "spatial",
    library_key: str | None = None,
    sample: str | None = None,
    opacity: float = 0.5,
    show_cells: bool = True,
    cell_size: float = 3.0,
    cell_opacity: float = 0.8,
    cmap: str = "tab20",
    background: str = "white",
    show_axes: bool = True,
    title: str | None = None,
    save: str | Path | None = None,
    off_screen: bool = False,
    return_plotter: bool = False,
    **kwargs,
):
    """
    Render 3D mesh boundaries of spatial clusters.

    Parameters
    ----------
    %(adata)s
    cluster_key
        Key in ``adata.obs`` where the cluster labels are stored.
    spatial_key
        Key in ``adata.obsm`` where the 3D spatial coordinates are stored.
    library_key
        Key in ``adata.obs`` where the sample labels are stored.
        Required if filtering by sample.
    sample
        Sample to plot. If None and library_key is provided, plots all samples.
    opacity
        Opacity of the mesh boundaries (0-1).
    show_cells
        Whether to show the cells as points.
    cell_size
        Size of the cell points.
    cell_opacity
        Opacity of the cell points (0-1).
    cmap
        Colormap to use for cluster colors.
    background
        Background color of the plot.
    show_axes
        Whether to show the axes.
    title
        Title of the plot.
    save
        Path to save the plot (supports PNG, JPG, SVG, PDF).
    off_screen
        If True, render off-screen (for saving without display).
    return_plotter
        If True, return the PyVista plotter object instead of showing the plot.
    **kwargs
        Additional arguments passed to ``pyvista.Plotter.add_mesh``.

    Returns
    -------
    pyvista.Plotter or None
        If ``return_plotter=True``, returns the plotter object.
        Otherwise, displays the plot and returns None.

    Examples
    --------
    >>> import cellcharter as cc
    >>>
    >>> # Plot 3D boundaries
    >>> cc.pl.boundaries_3d(adata, cluster_key="component", spatial_key="spatial_3d")
    >>>
    >>> # Save to file
    >>> cc.pl.boundaries_3d(adata, save="boundaries_3d.png")
    """
    pv = _check_pyvista()

    # Check if boundaries exist
    shape_key = f"shape_{cluster_key}"
    if shape_key not in adata.uns or "boundary" not in adata.uns[shape_key]:
        raise KeyError(
            f"Boundaries not found. Please run `cc.tl.boundaries(adata, cluster_key='{cluster_key}')` first."
        )

    boundaries = adata.uns[shape_key]["boundary"]

    # Filter by sample if needed
    if sample is not None and library_key is not None:
        adata_subset = adata[adata.obs[library_key] == sample]
        clusters_in_sample = adata_subset.obs[cluster_key].unique()
        boundaries = {k: v for k, v in boundaries.items() if k in clusters_in_sample}
    elif library_key is not None and sample is None:
        # Use all data
        adata_subset = adata
    else:
        adata_subset = adata

    # Check that boundaries are 3D
    first_boundary = next(iter(boundaries.values()), None)
    if first_boundary is None:
        raise ValueError("No boundaries found for the specified clusters")

    if not isinstance(first_boundary, Mesh3D):
        raise ValueError(
            "Boundaries are not 3D meshes. Please run `cc.tl.boundaries()` on 3D spatial data "
            "or ensure you're using the correct spatial_key."
        )

    # Create plotter
    plotter = pv.Plotter(off_screen=off_screen or save is not None)
    plotter.set_background(background)

    # Get colormap
    import matplotlib.pyplot as plt

    if isinstance(cmap, str):
        cmap_obj = plt.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap

    n_clusters = len(boundaries)
    colors = [cmap_obj(i / max(n_clusters - 1, 1))[:3] for i in range(n_clusters)]

    # Add cell points if requested
    if show_cells and spatial_key in adata_subset.obsm:
        coords = adata_subset.obsm[spatial_key]
        if coords.shape[1] >= 3:
            point_cloud = pv.PolyData(coords[:, :3])

            # Color by cluster
            cluster_values = adata_subset.obs[cluster_key].values
            if pd.api.types.is_categorical_dtype(cluster_values):
                scalars = cluster_values.codes.astype(float)
            else:
                categories = pd.Categorical(cluster_values)
                scalars = categories.codes.astype(float)

            scalars[scalars == -1] = np.nan
            point_cloud["cluster"] = scalars

            plotter.add_points(
                point_cloud,
                scalars="cluster",
                point_size=cell_size,
                opacity=cell_opacity,
                cmap=cmap,
                render_points_as_spheres=True,
                show_scalar_bar=False,
            )

    # Add mesh boundaries
    for i, (cluster, boundary) in enumerate(boundaries.items()):
        if boundary.mesh is None or len(boundary.vertices) == 0:
            warnings.warn(f"Empty boundary for cluster {cluster}, skipping.", stacklevel=2)
            continue

        try:
            # Convert trimesh to PyVista
            mesh = pv.PolyData(boundary.vertices, faces=_faces_to_pyvista(boundary.faces))
            plotter.add_mesh(
                mesh,
                color=colors[i % len(colors)],
                opacity=opacity,
                label=str(cluster),
                **kwargs,
            )
        except Exception as e:
            warnings.warn(f"Failed to add mesh for cluster {cluster}: {e}", stacklevel=2)

    # Add legend
    if n_clusters <= 20:
        try:
            plotter.add_legend()
        except Exception:
            pass

    if show_axes:
        plotter.show_axes()

    if title is not None:
        plotter.add_title(title)

    if save is not None:
        plotter.screenshot(str(save))
        if not off_screen and not return_plotter:
            plotter.close()
            return None

    if return_plotter:
        return plotter

    plotter.show()
    return None


def _faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    """
    Convert trimesh faces to PyVista format.

    PyVista expects faces in the format [n, v1, v2, v3, n, v1, v2, v3, ...]
    where n is the number of vertices per face.

    Parameters
    ----------
    faces : np.ndarray
        Trimesh faces array of shape (n_faces, 3).

    Returns
    -------
    np.ndarray
        PyVista-compatible faces array.
    """
    n_faces = len(faces)
    pv_faces = np.column_stack([np.full(n_faces, 3), faces]).flatten()
    return pv_faces
