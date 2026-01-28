"""Graph utilities."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def _assert_distances_key(adata: AnnData, key: str) -> None:
    if key not in adata.obsp:
        key_added = key.replace("_distances", "")
        raise KeyError(
            f"Spatial distances key `{key}` not found in `adata.obsp`. "
            f"Please run `squidpy.gr.spatial_neighbors(..., key_added={key_added!r})` first."
        )


def construct_3d_coordinates(
    adata: AnnData,
    section_key: str,
    z_spacing: float = 1.0,
    spatial_key: str = "spatial",
    output_key: str = "spatial_3d",
    section_order: list | None = None,
    copy: bool = False,
) -> np.ndarray | None:
    """
    Construct 3D spatial coordinates from 2D serial sections.

    This utility takes 2D spatial data from multiple sections (e.g., serial tissue
    slices) and combines them into unified 3D coordinates by adding z-values based
    on section ordering.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing 2D spatial coordinates.
    section_key : str
        Key in ``adata.obs`` containing section/slice identifiers.
    z_spacing : float, default=1.0
        Distance between sections in the z-direction. This should be set according
        to your experimental setup (e.g., slice thickness in micrometers).
    spatial_key : str, default="spatial"
        Key in ``adata.obsm`` where the 2D spatial coordinates are stored.
    output_key : str, default="spatial_3d"
        Key in ``adata.obsm`` where the 3D coordinates will be stored.
    section_order : list, optional
        Explicit ordering of sections along the z-axis. If None, sections are
        sorted alphanumerically. Use this to specify the correct physical ordering
        of your tissue sections.
    copy : bool, default=False
        If True, return the 3D coordinates without modifying adata.

    Returns
    -------
    np.ndarray or None
        If ``copy=True``, returns an (n_cells, 3) array of 3D coordinates.
        Otherwise, modifies ``adata.obsm[output_key]`` in place and returns None.

    Examples
    --------
    >>> import cellcharter as cc
    >>> import squidpy as sq
    >>>
    >>> # Combine serial sections into 3D
    >>> cc.gr.construct_3d_coordinates(adata, section_key="section", z_spacing=10.0)
    >>>
    >>> # Build 3D spatial graph
    >>> sq.gr.spatial_neighbors(adata, spatial_key="spatial_3d", coord_type="generic")
    >>>
    >>> # Run CellCharter pipeline
    >>> cc.gr.aggregate_neighbors(adata)
    >>> cc.tl.Cluster(n_clusters=10).fit(adata)
    >>> cc.tl.boundaries(adata, spatial_key="spatial_3d")  # Now computes 3D boundaries

    Notes
    -----
    The function assumes that all sections have the same XY coordinate system.
    If sections need alignment, this should be done before calling this function.
    """
    if section_key not in adata.obs:
        raise KeyError(f"Section key '{section_key}' not found in adata.obs")

    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm")

    coords_2d = adata.obsm[spatial_key]
    if coords_2d.shape[1] < 2:
        raise ValueError(f"Spatial coordinates must have at least 2 dimensions, got {coords_2d.shape[1]}")

    # Get unique sections
    sections = adata.obs[section_key].unique()

    # Determine section order
    if section_order is not None:
        # Validate that all sections are in the order
        missing = set(sections) - set(section_order)
        if missing:
            raise ValueError(f"Sections {missing} are in data but not in section_order")
        ordered_sections = section_order
    else:
        # Sort alphanumerically
        ordered_sections = sorted(sections, key=lambda x: (str(x), x))

    # Create z-coordinate mapping
    section_to_z = {section: i * z_spacing for i, section in enumerate(ordered_sections)}

    # Build 3D coordinates
    z_coords = adata.obs[section_key].map(section_to_z).values
    coords_3d = np.column_stack([coords_2d[:, :2], z_coords])

    if copy:
        return coords_3d

    adata.obsm[output_key] = coords_3d


def get_spatial_dimensions(adata: AnnData, spatial_key: str = "spatial") -> int:
    """
    Get the number of spatial dimensions from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial coordinates.
    spatial_key : str, default="spatial"
        Key in ``adata.obsm`` where spatial coordinates are stored.

    Returns
    -------
    int
        Number of spatial dimensions (2 or 3).

    Raises
    ------
    KeyError
        If spatial_key is not found in adata.obsm.
    ValueError
        If spatial coordinates have fewer than 2 or more than 3 dimensions.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm")

    ndim = adata.obsm[spatial_key].shape[1]
    if ndim < 2 or ndim > 3:
        raise ValueError(f"Expected 2 or 3 spatial dimensions, got {ndim}")

    return ndim
