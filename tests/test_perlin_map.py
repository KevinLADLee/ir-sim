"""
Tests for grid map generators, resolve_obstacle_map, and build_grid_from_generator.

Covers:
- Perlin2dMap / PerlinGridGenerator (+ generate_perlin_noise internals)
- resolve_obstacle_map: the single-point resolution dispatch
- build_grid_from_generator: YAML spec -> ndarray
- ImageGridGenerator: image-file based grid loading
"""

import os
import tempfile

import numpy as np
import pytest

from irsim.world.map import (
    GridMapGenerator,
    ImageGridGenerator,
    Perlin2dMap,
    PerlinGridGenerator,
    build_grid_from_generator,
    resolve_obstacle_map,
)
from irsim.world.map.perlin_map_generator import generate_perlin_noise


class TestPerlin2dMap:
    """Tests for Perlin2dMap."""

    def test_perlin2dmap_generate_returns_self(self):
        """Test that generate() returns self for chaining."""
        pmap = Perlin2dMap(50, 50, seed=42)
        out = pmap.generate()
        assert out is pmap

    def test_perlin2dmap_grid_shape(self):
        """Test that Perlin2dMap.grid has correct shape."""
        pmap = Perlin2dMap(100, 80, seed=42).generate()
        assert pmap.grid.shape == (100, 80)

    def test_perlin2dmap_grid_lazy(self):
        """Test that accessing .grid triggers generation if not yet built."""
        pmap = Perlin2dMap(50, 50, seed=42)
        assert pmap._grid is None
        grid = pmap.grid
        assert grid.shape == (50, 50)
        assert pmap._grid is not None

    def test_perlin2dmap_save_as_image(self):
        """Test that save_as_image creates a file."""
        pmap = Perlin2dMap(50, 50, seed=42).generate()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "perlin2d.png")
            pmap.save_as_image(filepath)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0

    def test_perlin2dmap_same_params_same_grid(self):
        """Test that same params produce identical grid."""
        pmap1 = Perlin2dMap(100, 100, fill=0.38, seed=42).generate()
        pmap2 = Perlin2dMap(100, 100, fill=0.38, seed=42).generate()
        np.testing.assert_array_equal(pmap1.grid, pmap2.grid)


class TestPerlinNoiseGeneration:
    """Tests for generate_perlin_noise."""

    def test_generate_noise_shape(self):
        """Test that noise array has correct shape."""
        noise = generate_perlin_noise(100, 80)
        assert noise.shape == (100, 80)

    def test_generate_noise_range(self):
        """Test that noise values are in [0, 1] range."""
        noise = generate_perlin_noise(100, 100, seed=42)
        assert noise.min() >= 0.0
        assert noise.max() <= 1.0

    def test_noise_seed_reproducibility(self):
        """Test that same seed produces identical noise."""
        noise1 = generate_perlin_noise(50, 50, seed=123)
        noise2 = generate_perlin_noise(50, 50, seed=123)
        np.testing.assert_array_equal(noise1, noise2)

    def test_noise_different_seeds(self):
        """Test that different seeds produce different noise."""
        noise1 = generate_perlin_noise(50, 50, seed=1)
        noise2 = generate_perlin_noise(50, 50, seed=2)
        assert not np.allclose(noise1, noise2)


class TestPerlinMapGeneration:
    """Tests for Perlin2dMap grid generation."""

    def test_generate_map_shape(self):
        """Test that map has correct shape."""
        grid = Perlin2dMap(100, 80).generate().grid
        assert grid.shape == (100, 80)

    def test_generate_map_value_range(self):
        """Test that map values are in [0, 100] range."""
        grid = Perlin2dMap(100, 100, seed=42).generate().grid
        assert grid.min() >= 0
        assert grid.max() <= 100

    def test_generate_map_binary_like(self):
        """Test that map is binary-like (0 or 100 values only)."""
        grid = Perlin2dMap(100, 100, seed=42).generate().grid
        unique_values = np.unique(grid)
        assert len(unique_values) == 2
        assert 0.0 in unique_values
        assert 100.0 in unique_values

    def test_map_seed_reproducibility(self):
        """Test that same seed produces identical maps."""
        grid1 = Perlin2dMap(50, 50, seed=42).generate().grid
        grid2 = Perlin2dMap(50, 50, seed=42).generate().grid
        np.testing.assert_array_equal(grid1, grid2)


class TestFillParameter:
    """Tests for fill parameter effects."""

    def test_fill_affects_obstacle_ratio(self):
        """Test that higher fill produces more obstacles."""
        grid_low = Perlin2dMap(100, 100, fill=0.3, seed=42).generate().grid
        grid_high = Perlin2dMap(100, 100, fill=0.7, seed=42).generate().grid

        obstacles_low = np.sum(grid_low > 50)
        obstacles_high = np.sum(grid_high > 50)

        assert obstacles_low < obstacles_high

    def test_fill_matches_obstacle_ratio(self):
        """Test that fill=0.38 produces ~38% obstacles."""
        grid = Perlin2dMap(100, 100, fill=0.38, seed=42).generate().grid
        obstacle_ratio = np.sum(grid > 50) / grid.size
        assert 0.36 <= obstacle_ratio <= 0.40

    def test_extreme_fill_high(self):
        """Test high fill creates mostly obstacles."""
        grid = Perlin2dMap(100, 100, fill=0.9, seed=42).generate().grid
        obstacle_ratio = np.sum(grid > 50) / grid.size
        assert obstacle_ratio > 0.85

    def test_extreme_fill_low(self):
        """Test low fill creates mostly free space."""
        grid = Perlin2dMap(100, 100, fill=0.1, seed=42).generate().grid
        obstacle_ratio = np.sum(grid > 50) / grid.size
        assert obstacle_ratio < 0.15


class TestComplexityParameter:
    """Tests for complexity parameter."""

    def test_complexity_affects_feature_size(self):
        """Test that different complexity produces different patterns."""
        grid_small = Perlin2dMap(100, 100, complexity=0.02, seed=42).generate().grid
        grid_large = Perlin2dMap(100, 100, complexity=0.2, seed=42).generate().grid

        assert not np.array_equal(grid_small, grid_large)


class TestFractalParameter:
    """Tests for fractal parameter."""

    def test_fractal_affects_detail(self):
        """Test that more fractal layers add more detail."""
        grid_few = Perlin2dMap(100, 100, fractal=1, seed=42).generate().grid
        grid_many = Perlin2dMap(100, 100, fractal=6, seed=42).generate().grid

        assert not np.array_equal(grid_few, grid_many)


class TestSaveMapAsImage:
    """Tests for save_as_image."""

    def test_save_creates_file(self):
        """Test that save_as_image creates a file."""
        pmap = Perlin2dMap(50, 50, seed=42).generate()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_map.png")
            pmap.save_as_image(filepath)
            assert os.path.exists(filepath)

    def test_save_file_not_empty(self):
        """Test that saved file has content."""
        pmap = Perlin2dMap(50, 50, seed=42).generate()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_map.png")
            pmap.save_as_image(filepath)
            assert os.path.getsize(filepath) > 0

    def test_save_invert_option(self):
        """Test that invert option works."""
        pmap = Perlin2dMap(50, 50, seed=42).generate()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_normal = os.path.join(tmpdir, "normal.png")
            filepath_inverted = os.path.join(tmpdir, "inverted.png")
            pmap.save_as_image(filepath_normal, invert=False)
            pmap.save_as_image(filepath_inverted, invert=True)
            assert os.path.getsize(filepath_normal) > 0
            assert os.path.getsize(filepath_inverted) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_map(self):
        """Test generation of very small map."""
        grid = Perlin2dMap(10, 10, seed=42).generate().grid
        assert grid.shape == (10, 10)

    def test_non_square_map(self):
        """Test generation of non-square map."""
        grid = Perlin2dMap(50, 100, seed=42).generate().grid
        assert grid.shape == (50, 100)

    def test_single_fractal(self):
        """Test generation with single fractal layer."""
        grid = Perlin2dMap(50, 50, fractal=1, seed=42).generate().grid
        assert grid.shape == (50, 50)

    def test_many_fractal(self):
        """Test generation with many fractal layers."""
        grid = Perlin2dMap(50, 50, fractal=8, seed=42).generate().grid
        assert grid.shape == (50, 50)


# ---------------------------------------------------------------------------
# resolve_obstacle_map tests
# ---------------------------------------------------------------------------


class TestResolveObstacleMap:
    """Tests for the central resolve_obstacle_map dispatch function."""

    def test_none_returns_none(self):
        """None input produces None output."""
        assert resolve_obstacle_map(None) is None

    def test_ndarray_passthrough(self):
        """An ndarray is returned as float64."""
        arr = np.ones((10, 10), dtype=np.int32) * 50
        result = resolve_obstacle_map(arr)
        assert result is not None
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, arr.astype(np.float64))

    def test_dict_spec_perlin(self):
        """A dict with name='perlin' builds an ndarray via the registry."""
        spec = {"name": "perlin", "width": 30, "height": 30, "seed": 7}
        result = resolve_obstacle_map(spec)
        assert isinstance(result, np.ndarray)
        assert result.shape == (30, 30)
        assert result.dtype == np.float64

    def test_object_with_grid_attribute(self):
        """An object with .grid is resolved to its grid ndarray."""
        gen = Perlin2dMap(20, 20, seed=1).generate()
        result = resolve_obstacle_map(gen)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, gen.grid)

    def test_unknown_input_returns_none(self):
        """Unrecognised types return None."""
        assert resolve_obstacle_map(12345) is None
        assert resolve_obstacle_map([1, 2, 3]) is None

    def test_dict_without_name_returns_none(self):
        """A dict missing the 'name' key returns None."""
        assert resolve_obstacle_map({"width": 10}) is None


# ---------------------------------------------------------------------------
# build_grid_from_generator tests
# ---------------------------------------------------------------------------


class TestBuildGridFromGenerator:
    """Tests for build_grid_from_generator."""

    def test_builds_perlin_grid(self):
        """Correctly builds an ndarray from a perlin spec."""
        spec = {"name": "perlin", "width": 40, "height": 40, "seed": 99}
        grid = build_grid_from_generator(spec)
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (40, 40)
        assert grid.dtype == np.float64

    def test_unknown_name_raises(self):
        """Raises ValueError for an unregistered generator name."""
        with pytest.raises(ValueError, match="Unknown"):
            build_grid_from_generator({"name": "nonexistent_generator"})

    def test_missing_name_raises(self):
        """Raises ValueError when name key is absent."""
        with pytest.raises(ValueError, match="Unknown"):
            build_grid_from_generator({"width": 10})

    def test_extra_keys_ignored(self):
        """Keys not in yaml_param_names are silently ignored."""
        spec = {
            "name": "perlin",
            "width": 20,
            "height": 20,
            "seed": 1,
            "unknown_param": 999,
        }
        grid = build_grid_from_generator(spec)
        assert grid.shape == (20, 20)

    def test_perlin_registered(self):
        """PerlinGridGenerator is auto-registered under 'perlin'."""
        assert "perlin" in GridMapGenerator.registry
        assert GridMapGenerator.registry["perlin"] is PerlinGridGenerator

    def test_image_registered(self):
        """ImageGridGenerator is auto-registered under 'image'."""
        assert "image" in GridMapGenerator.registry
        assert GridMapGenerator.registry["image"] is ImageGridGenerator


# ---------------------------------------------------------------------------
# ImageGridGenerator tests
# ---------------------------------------------------------------------------


class TestImageGridGenerator:
    """Tests for ImageGridGenerator."""

    def test_load_grayscale_png(self):
        """Load a known grayscale PNG and verify grid properties."""
        # tests/cave.png is 500x500 grayscale
        cave_path = os.path.join(os.path.dirname(__file__), "cave.png")
        gen = ImageGridGenerator(path=cave_path).generate()
        assert gen.grid.shape[0] > 0
        assert gen.grid.shape[1] > 0
        assert gen.grid.dtype == np.float64
        assert gen.grid.min() >= 0
        assert gen.grid.max() <= 100

    def test_missing_file_raises(self):
        """FileNotFoundError raised for a nonexistent image path."""
        gen = ImageGridGenerator(path="this_image_does_not_exist_xyz.png")
        with pytest.raises(FileNotFoundError):
            gen.generate()

    def test_lazy_grid_access(self):
        """Accessing .grid without calling .generate() triggers generation."""
        cave_path = os.path.join(os.path.dirname(__file__), "cave.png")
        gen = ImageGridGenerator(path=cave_path)
        assert gen._grid is None
        grid = gen.grid  # should auto-generate
        assert grid is not None
        assert grid.shape[0] > 0


# ---------------------------------------------------------------------------
# RNG isolation tests
# ---------------------------------------------------------------------------


class TestRNGIsolation:
    """Verify that Perlin noise generation does not mutate global RNG."""

    def test_perlin_does_not_affect_global_rng(self):
        """generate_perlin_noise with seed should not alter global RNG state."""
        from irsim.util.random import rng

        # Sample from global rng, then run perlin with a seed, then sample again
        rng_state_before = rng.random()
        _ = generate_perlin_noise(30, 30, seed=42)
        rng_state_after = rng.random()

        # If perlin mutated global rng via set_seed, the second sample
        # would be deterministic (always the same). Run a second time:
        rng_state_before2 = rng.random()
        _ = generate_perlin_noise(30, 30, seed=42)
        rng_state_after2 = rng.random()

        # The global rng should keep advancing independently
        # (not reset by perlin's seed)
        assert rng_state_after != rng_state_after2
