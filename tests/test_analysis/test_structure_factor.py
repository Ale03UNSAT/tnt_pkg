"""
Pytest tests for structure_factor.py.
"""

import numpy as np
import pandas as pd
import pytest

from tntlib.analysis import (
    radial_structure_factor
)


def test_radial_wavevector_and_column_order():
    df = pd.DataFrame({
        "kx": [0.0, 1.0, 2.0],
        "ky": [0.0, 0.0, 0.0],
        "S": [10.0, 20.0, 30.0],
    })

    result = radial_structure_factor(df, nbins=2)

    assert isinstance(result, pd.DataFrame)
    assert result.columns[0] == "k"
    assert list(result.columns) == ["k", "S"]

    np.testing.assert_allclose(result["k"], [0.5, 2.0])
    np.testing.assert_allclose(result["S"], [15.0, 30.0])
    

def test_radial_averaging_of_points_in_same_bin():
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 2.0, 0.0],
        "ky": [0.0, 1.0, 0.0, 2.0],
        "S": [2.0, 4.0, 6.0, 8.0],
    })

    result = radial_structure_factor(df, nbins=2)

    assert len(result) == 2
    np.testing.assert_allclose(result["k"], [1.0, 2.0])
    np.testing.assert_allclose(result["S"], [3.0, 7.0])


def test_all_observables_are_radially_averaged():
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 2.0, 0.0],
        "ky": [0.0, 1.0, 0.0, 2.0],
        "S": [2.0, 4.0, 6.0, 8.0],
        "temperature": [10.0, 14.0, 20.0, 24.0],
    })

    result = radial_structure_factor(df, nbins=2)

    assert list(result.columns) == ["k", "S", "temperature"]
    np.testing.assert_allclose(result["S"], [3.0, 7.0])
    np.testing.assert_allclose(result["temperature"], [12.0, 22.0])


def test_empty_bins_are_removed():
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 5.0],
        "ky": [0.0, 2.0, 0.0],
        "S": [10.0, 20.0, 30.0],
    })

    result = radial_structure_factor(df, nbins=5)

    assert len(result) == 3
    assert result["S"].notna().all()


def test_default_number_of_bins():
    df = pd.DataFrame({
        "kx": [0.0, 1.0, 2.0],
        "ky": [0.0, 0.0, 0.0],
        "S": [1.0, 2.0, 3.0],
    })

    result = radial_structure_factor(df)

    assert isinstance(result, pd.DataFrame)
    assert "k" in result.columns
    assert result["S"].notna().all()


@pytest.mark.parametrize(
    "velocity_columns",
    [("vx", "vy"), ("ux", "uy"), ("u", "v")],
)
def test_kinetic_energy_is_constructed(velocity_columns):
    vx_name, vy_name = velocity_columns
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 2.0, 0.0],
        "ky": [0.0, 1.0, 0.0, 2.0],
        vx_name: [2.0, 4.0, 6.0, 8.0],
        vy_name: [0.0, 0.0, 0.0, 0.0],
    })

    result = radial_structure_factor(df, nbins=2)

    assert "Ekin" in result.columns
    np.testing.assert_allclose(result["Ekin"], [5.0, 25.0])


def test_existing_kinetic_energy_is_not_overwritten():
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 2.0, 0.0],
        "ky": [0.0, 1.0, 0.0, 2.0],
        "vx": [2.0, 4.0, 6.0, 8.0],
        "vy": [0.0, 0.0, 0.0, 0.0],
        "Ekin": [100.0, 200.0, 300.0, 400.0],
    })

    result = radial_structure_factor(df, nbins=2)

    np.testing.assert_allclose(result["Ekin"], [150.0, 350.0])


def test_no_kinetic_energy_without_complete_velocity_pair():
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 2.0],
        "ky": [0.0, 1.0, 0.0],
        "vx": [2.0, 4.0, 6.0],
    })

    result = radial_structure_factor(df, nbins=2)

    assert "Ekin" not in result.columns


def test_input_dataframe_is_not_modified():
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 2.0, 0.0],
        "ky": [0.0, 1.0, 0.0, 2.0],
        "S": [2.0, 4.0, 6.0, 8.0],
        "vx": [1.0, 2.0, 3.0, 4.0],
        "vy": [4.0, 3.0, 2.0, 1.0],
    })
    original = df.copy(deep=True)

    radial_structure_factor(df, nbins=2)

    pd.testing.assert_frame_equal(df, original)


def test_result_has_reset_index():
    df = pd.DataFrame({
        "kx": [1.0, 0.0, 2.0, 0.0],
        "ky": [0.0, 1.0, 0.0, 2.0],
        "S": [2.0, 4.0, 6.0, 8.0],
    }, index=[10, 20, 30, 40])

    result = radial_structure_factor(df, nbins=2)

    assert result.index.equals(pd.RangeIndex(len(result)))


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"kx": [1.0, 2.0]}),
    ],
)
def test_rejects_fewer_than_two_columns(df):
    with pytest.raises(ValueError, match="at least two columns"):
        radial_structure_factor(df)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"x": [1.0, 2.0], "ky": [0.0, 1.0]}),
        pd.DataFrame({"ky": [0.0, 1.0], "kx": [1.0, 2.0]}),
        pd.DataFrame({"kx": [1.0, 2.0], "y": [0.0, 1.0]}),
    ],
)
def test_rejects_incorrect_first_two_columns(df):
    with pytest.raises(ValueError, match="first two columns"):
        radial_structure_factor(df)
