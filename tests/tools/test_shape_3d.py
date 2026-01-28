"""Tests for 3D shape analysis functions."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import cellcharter as cc
from cellcharter.tl._shape import Mesh3D, _detect_ndim, _elongation_3d


# ============================================================================
# Helper functions (must be defined before test classes that use them)
# ============================================================================


def _has_3d_deps() -> bool:
    """Check if 3D dependencies are available."""
    try:
        import alphashape  # noqa: F401
        import trimesh  # noqa: F401
        return True
    except ImportError:
        return False


def _create_sphere_adata(n_cells: int) -> AnnData:
    """Create AnnData with spherical 3D cluster."""
    import squidpy as sq

    # Generate points in a sphere
    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, n_cells)
    phi = np.random.uniform(0, np.pi, n_cells)
    r = np.random.uniform(0, 1, n_cells) ** (1/3)  # Uniform within sphere

    points = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ]) * 10  # Scale up

    adata = AnnData(X=np.random.rand(n_cells, 10))
    adata.obsm["spatial"] = points
    adata.obs["cluster"] = pd.Categorical(["cluster_0"] * n_cells)

    # Build spatial graph
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=15)

    return adata


def _create_elongated_cluster_adata() -> AnnData:
    """Create AnnData with elongated 3D cluster."""
    import squidpy as sq

    np.random.seed(42)
    n_cells = 200

    # Generate points along a tube
    t = np.linspace(0, 10, n_cells)
    points = np.column_stack([
        t,
        np.random.normal(0, 0.5, n_cells),
        np.random.normal(0, 0.5, n_cells)
    ])

    adata = AnnData(X=np.random.rand(n_cells, 10))
    adata.obsm["spatial"] = points
    adata.obs["cluster"] = pd.Categorical(["cluster_0"] * n_cells)

    # Build spatial graph
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=15)

    return adata


# ============================================================================
# Test classes
# ============================================================================


class TestDimensionDetection:
    """Tests for dimension detection utility."""

    def test_detect_2d(self):
        """Test detection of 2D coordinates."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 2)
        assert _detect_ndim(adata) == 2

    def test_detect_3d(self):
        """Test detection of 3D coordinates."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 3)
        assert _detect_ndim(adata) == 3

    def test_explicit_ndim(self):
        """Test explicit ndim parameter."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 3)
        assert _detect_ndim(adata, ndim=2) == 2
        assert _detect_ndim(adata, ndim=3) == 3

    def test_missing_spatial_key(self):
        """Test error when spatial key is missing."""
        adata = AnnData(X=np.random.rand(100, 10))
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            _detect_ndim(adata)

    def test_invalid_ndim(self):
        """Test error for invalid dimensions."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 4)
        with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
            _detect_ndim(adata)


class TestMesh3D:
    """Tests for Mesh3D class."""

    def test_empty_mesh(self):
        """Test creation of empty mesh."""
        mesh = Mesh3D()
        assert mesh.volume == 0.0
        assert len(mesh.vertices) == 0

    def test_serialization_roundtrip(self):
        """Test mesh serialization and deserialization."""
        pytest.importorskip("trimesh")
        import trimesh

        # Create a simple cube mesh
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=float)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        wrapped = Mesh3D.from_trimesh(mesh)

        # Serialize and deserialize
        data = wrapped.to_bytes()
        loaded = Mesh3D.from_bytes(data)

        # Check that vertices and faces match
        assert np.allclose(wrapped.vertices, loaded.vertices)
        assert np.array_equal(wrapped.faces, loaded.faces)


class TestElongation3D:
    """Tests for 3D elongation metric."""

    def test_sphere_low_elongation(self):
        """Test that a spherical point cloud has low elongation."""
        # Generate points INSIDE a sphere (uniform distribution)
        np.random.seed(42)
        n_points = 1000

        # Uniform points in a sphere using rejection sampling
        points = []
        while len(points) < n_points:
            p = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1:
                points.append(p)
        points = np.array(points)

        elongation = _elongation_3d(points)
        # Sphere should have low elongation (close to 0)
        assert elongation < 0.3

    def test_line_high_elongation(self):
        """Test that a line has high elongation."""
        # Generate points along a line
        n_points = 100
        points = np.column_stack([
            np.linspace(0, 10, n_points),
            np.random.normal(0, 0.1, n_points),
            np.random.normal(0, 0.1, n_points)
        ])

        elongation = _elongation_3d(points)
        # Line should have high elongation (close to 1)
        assert elongation > 0.9

    def test_disk_high_elongation(self):
        """Test that a flat disk has high elongation (one dimension is nearly zero)."""
        np.random.seed(42)
        # Generate points on a flat disk (2D circle embedded in 3D)
        n_points = 500
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        r = np.random.uniform(0, 1, n_points)

        points = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.random.normal(0, 0.01, n_points)  # Very flat in z
        ])

        elongation = _elongation_3d(points)
        # Flat disk should have high elongation because z dimension is tiny
        assert elongation > 0.9


class TestConstruct3DCoordinates:
    """Tests for construct_3d_coordinates utility."""

    def test_basic_construction(self):
        """Test basic 3D coordinate construction."""
        # Create test data with 2 sections
        n_cells = 100
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 50 + ["B"] * 50)

        cc.gr.construct_3d_coordinates(adata, section_key="section", z_spacing=10.0)

        assert "spatial_3d" in adata.obsm
        assert adata.obsm["spatial_3d"].shape == (n_cells, 3)

        # Check z coordinates
        z_coords = adata.obsm["spatial_3d"][:, 2]
        assert all(z_coords[:50] == 0.0)  # Section A at z=0
        assert all(z_coords[50:] == 10.0)  # Section B at z=10

    def test_custom_section_order(self):
        """Test custom section ordering."""
        n_cells = 100
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 50 + ["B"] * 50)

        # Reverse the order
        cc.gr.construct_3d_coordinates(
            adata, section_key="section", z_spacing=10.0, section_order=["B", "A"]
        )

        z_coords = adata.obsm["spatial_3d"][:, 2]
        assert all(z_coords[:50] == 10.0)  # Section A now at z=10
        assert all(z_coords[50:] == 0.0)  # Section B now at z=0

    def test_copy_mode(self):
        """Test copy mode returns coordinates without modifying adata."""
        n_cells = 100
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 50 + ["B"] * 50)

        coords = cc.gr.construct_3d_coordinates(
            adata, section_key="section", z_spacing=10.0, copy=True
        )

        assert coords is not None
        assert coords.shape == (n_cells, 3)
        assert "spatial_3d" not in adata.obsm

    def test_missing_section_key(self):
        """Test error when section key is missing."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 2)

        with pytest.raises(KeyError, match="not found in adata.obs"):
            cc.gr.construct_3d_coordinates(adata, section_key="nonexistent")


class TestBoundaries3D:
    """Tests for 3D boundary computation."""

    @pytest.mark.skipif(
        not _has_3d_deps(),
        reason="3D dependencies (alphashape, trimesh) not installed"
    )
    def test_boundaries_3d_sphere(self):
        """Test 3D boundary computation on a spherical cluster."""
        # Create synthetic data with a spherical cluster
        n_cells = 500
        adata = _create_sphere_adata(n_cells)

        cc.gr.connected_components(adata, cluster_key="cluster", min_cells=10)
        cc.tl.boundaries(adata, spatial_key="spatial")

        boundaries = adata.uns["shape_component"]["boundary"]
        assert len(boundaries) > 0

        # Check that boundaries are Mesh3D objects
        for boundary in boundaries.values():
            assert isinstance(boundary, Mesh3D)

    @pytest.mark.skipif(
        not _has_3d_deps(),
        reason="3D dependencies (alphashape, trimesh) not installed"
    )
    def test_boundaries_3d_convex_hull(self):
        """Test 3D boundary computation with convex hull fallback."""
        n_cells = 100
        adata = _create_sphere_adata(n_cells)

        cc.gr.connected_components(adata, cluster_key="cluster", min_cells=10)
        cc.tl.boundaries(adata, spatial_key="spatial", use_convex_hull=True)

        boundaries = adata.uns["shape_component"]["boundary"]
        assert len(boundaries) > 0


class TestShapeMetrics3D:
    """Tests for 3D shape metrics."""

    @pytest.mark.skipif(
        not _has_3d_deps(),
        reason="3D dependencies (alphashape, trimesh) not installed"
    )
    def test_elongation_3d_metric(self):
        """Test 3D elongation metric computation."""
        adata = _create_elongated_cluster_adata()

        cc.gr.connected_components(adata, cluster_key="cluster", min_cells=10)
        cc.tl.boundaries(adata, spatial_key="spatial")
        cc.tl.elongation_metric(adata, spatial_key="spatial")

        elongation = adata.uns["shape_component"]["elongation"]
        assert len(elongation) > 0

        # All values should be between 0 and 1
        for val in elongation.values():
            assert 0 <= val <= 1

    @pytest.mark.skipif(
        not _has_3d_deps(),
        reason="3D dependencies (alphashape, trimesh) not installed"
    )
    def test_purity_3d_metric(self):
        """Test 3D purity metric computation."""
        adata = _create_sphere_adata(200)
        adata.obs["sample"] = "sample1"

        cc.gr.connected_components(adata, cluster_key="cluster", min_cells=10)
        cc.tl.boundaries(adata, spatial_key="spatial")

        # This may fail if mesh is not watertight, so we catch the warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cc.tl.purity_metric(adata, library_key="sample", spatial_key="spatial")

        purity = adata.uns["shape_component"]["purity"]
        assert len(purity) > 0
