"""Tests for graph utility functions."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import cellcharter as cc


class TestConstruct3DCoordinates:
    """Tests for construct_3d_coordinates function."""

    def test_basic_construction(self):
        """Test basic 3D coordinate construction from 2D sections."""
        n_cells = 100
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2) * 100
        adata.obs["section"] = pd.Categorical(
            ["section_1"] * 30 + ["section_2"] * 40 + ["section_3"] * 30
        )

        cc.gr.construct_3d_coordinates(adata, section_key="section", z_spacing=10.0)

        assert "spatial_3d" in adata.obsm
        coords_3d = adata.obsm["spatial_3d"]

        # Check shape
        assert coords_3d.shape == (n_cells, 3)

        # Check z values
        assert all(coords_3d[:30, 2] == 0.0)  # section_1
        assert all(coords_3d[30:70, 2] == 10.0)  # section_2
        assert all(coords_3d[70:, 2] == 20.0)  # section_3

        # Check XY coordinates preserved
        assert np.allclose(coords_3d[:, :2], adata.obsm["spatial"])

    def test_custom_z_spacing(self):
        """Test with different z spacing values."""
        n_cells = 60
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 30 + ["B"] * 30)

        # Test with larger z_spacing
        cc.gr.construct_3d_coordinates(adata, section_key="section", z_spacing=50.0)

        z_coords = adata.obsm["spatial_3d"][:, 2]
        assert z_coords[0] == 0.0
        assert z_coords[30] == 50.0

    def test_custom_section_order(self):
        """Test custom ordering of sections."""
        n_cells = 90
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 30 + ["B"] * 30 + ["C"] * 30)

        # Custom order: C, A, B
        cc.gr.construct_3d_coordinates(
            adata,
            section_key="section",
            z_spacing=10.0,
            section_order=["C", "A", "B"]
        )

        z_coords = adata.obsm["spatial_3d"][:, 2]
        assert all(z_coords[:30] == 10.0)  # A is second in order
        assert all(z_coords[30:60] == 20.0)  # B is third in order
        assert all(z_coords[60:] == 0.0)  # C is first in order

    def test_custom_output_key(self):
        """Test custom output key name."""
        n_cells = 50
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 25 + ["B"] * 25)

        cc.gr.construct_3d_coordinates(
            adata,
            section_key="section",
            z_spacing=10.0,
            output_key="my_3d_coords"
        )

        assert "my_3d_coords" in adata.obsm
        assert "spatial_3d" not in adata.obsm

    def test_copy_returns_array(self):
        """Test that copy=True returns array without modifying adata."""
        n_cells = 50
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 25 + ["B"] * 25)

        result = cc.gr.construct_3d_coordinates(
            adata,
            section_key="section",
            z_spacing=10.0,
            copy=True
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (n_cells, 3)
        assert "spatial_3d" not in adata.obsm

    def test_missing_section_key_error(self):
        """Test error when section key doesn't exist."""
        adata = AnnData(X=np.random.rand(50, 10))
        adata.obsm["spatial"] = np.random.rand(50, 2)

        with pytest.raises(KeyError, match="not found in adata.obs"):
            cc.gr.construct_3d_coordinates(adata, section_key="missing_key")

    def test_missing_spatial_key_error(self):
        """Test error when spatial key doesn't exist."""
        adata = AnnData(X=np.random.rand(50, 10))
        adata.obs["section"] = pd.Categorical(["A"] * 50)

        with pytest.raises(KeyError, match="not found in adata.obsm"):
            cc.gr.construct_3d_coordinates(adata, section_key="section")

    def test_invalid_section_order_error(self):
        """Test error when section_order is missing sections."""
        n_cells = 60
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical(["A"] * 30 + ["B"] * 30)

        with pytest.raises(ValueError, match="in data but not in section_order"):
            cc.gr.construct_3d_coordinates(
                adata,
                section_key="section",
                section_order=["A"]  # Missing B
            )

    def test_numeric_sections(self):
        """Test with numeric section identifiers."""
        n_cells = 60
        adata = AnnData(X=np.random.rand(n_cells, 10))
        adata.obsm["spatial"] = np.random.rand(n_cells, 2)
        adata.obs["section"] = pd.Categorical([1] * 20 + [2] * 20 + [3] * 20)

        cc.gr.construct_3d_coordinates(adata, section_key="section", z_spacing=5.0)

        z_coords = adata.obsm["spatial_3d"][:, 2]
        assert all(z_coords[:20] == 0.0)
        assert all(z_coords[20:40] == 5.0)
        assert all(z_coords[40:] == 10.0)


class TestGetSpatialDimensions:
    """Tests for get_spatial_dimensions function."""

    def test_2d_detection(self):
        """Test detection of 2D spatial data."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 2)

        assert cc.gr.get_spatial_dimensions(adata) == 2

    def test_3d_detection(self):
        """Test detection of 3D spatial data."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 3)

        assert cc.gr.get_spatial_dimensions(adata) == 3

    def test_custom_key(self):
        """Test with custom spatial key."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["custom_coords"] = np.random.rand(100, 3)

        assert cc.gr.get_spatial_dimensions(adata, spatial_key="custom_coords") == 3

    def test_missing_key_error(self):
        """Test error when spatial key is missing."""
        adata = AnnData(X=np.random.rand(100, 10))

        with pytest.raises(KeyError, match="not found in adata.obsm"):
            cc.gr.get_spatial_dimensions(adata)

    def test_invalid_dimensions_error(self):
        """Test error for invalid number of dimensions."""
        adata = AnnData(X=np.random.rand(100, 10))
        adata.obsm["spatial"] = np.random.rand(100, 4)

        with pytest.raises(ValueError, match="Expected 2 or 3"):
            cc.gr.get_spatial_dimensions(adata)
