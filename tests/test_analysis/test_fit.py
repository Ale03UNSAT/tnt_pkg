"""
Pytest tests for fit.py.
"""

import numpy as np
import pytest

from tntlib.analysis import (
    weighted_mean,
    lin_cov,
    lin_var,
    res_lin,
    chi2_lin,
    _line,
    _exponential,
    _powerlaw,
    y_unc,
    LinearFit,
    ExponentialFit,
    PowerLawFit
)


# ---------------------------------------------------------------------
# Basic weighted statistics
# ---------------------------------------------------------------------

def test_weighted_mean():
    x = np.array([1.0, 2.0, 4.0])
    w = np.array([1.0, 2.0, 1.0])

    assert weighted_mean(x, w) == pytest.approx(2.25)


def test_lin_cov():
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([2.0, 1.0, 3.0])
    w = np.array([1.0, 2.0, 1.0])

    expected = weighted_mean(x * y, w) - (
        weighted_mean(x, w) * weighted_mean(y, w)
    )

    assert lin_cov(x, y, w) == pytest.approx(expected)


def test_lin_var():
    x = np.array([1.0, 2.0, 4.0])
    w = np.array([1.0, 2.0, 1.0])

    assert lin_var(x, w) == pytest.approx(lin_cov(x, x, w))


# ---------------------------------------------------------------------
# Residuals and chi-square
# ---------------------------------------------------------------------

def test_res_lin():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.1, 2.0, 3.2])
    sigma_y = np.array([0.1, 0.2, 0.2])

    expected = (y - 1.0 * x - 1.0) / sigma_y

    np.testing.assert_allclose(
        res_lin(x, y, 1.0, 1.0, sigma_y),
        expected,
    )


def test_res_lin_verbose(capsys):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    sigma_y = np.ones(3)

    res_lin(x, y, 1.0, 1.0, sigma_y, verbose=True)

    output = capsys.readouterr().out
    assert "e_1" in output
    assert "e_2" in output
    assert "e_3" in output


def test_chi2_lin():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.1, 2.0, 3.2])
    sigma_y = np.array([0.1, 0.2, 0.2])

    residuals = (y - (x + 1.0)) / sigma_y

    assert chi2_lin(x, y, 1.0, 1.0, sigma_y) == pytest.approx(
        np.sum(residuals**2)
    )


# ---------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------

def test_line_model():
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(_line(x, 2.0, 1.0), [1.0, 3.0, 5.0])


def test_exponential_model():
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        _exponential(x, 0.5, 2.0),
        2.0 * np.exp(0.5 * x),
    )


def test_powerlaw_model():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(_powerlaw(x, 2.0, 3.0), 3.0 * x**2)


# ---------------------------------------------------------------------
# Uncertainty on the fitted line
# ---------------------------------------------------------------------

def test_y_unc_scalar():
    x = 2.0
    u_m = 0.1
    u_c = 0.2
    cov_mc = 0.005

    expected = np.sqrt(x**2 * u_m**2 + u_c**2 + 2 * x * cov_mc)

    assert y_unc(x, u_m, u_c, cov_mc) == pytest.approx(expected)


def test_y_unc_array():
    x = np.array([0.0, 1.0, 2.0])
    result = y_unc(x, 0.1, 0.2, 0.0)

    expected = np.sqrt(x**2 * 0.1**2 + 0.2**2)
    np.testing.assert_allclose(result, expected)


def test_y_unc_clips_negative_variance_to_zero():
    # The function explicitly clips negative variances to zero.
    assert y_unc(1.0, 0.0, 0.0, -1.0) == 0.0


# ---------------------------------------------------------------------
# Linear fit
# ---------------------------------------------------------------------

@pytest.fixture
def linear_data():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    sigma_y = np.array([0.2, 0.2, 0.2, 0.2])
    return x, y, sigma_y


def test_linear_fit_recovers_exact_line(linear_data):
    x, y, sigma_y = linear_data

    result = LinearFit(x, y, sigma_y, verbose_fit=False)
    slope, u_slope, intercept, u_intercept, covariance, chi2, rho, residuals = result

    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert u_slope > 0
    assert u_intercept > 0
    assert covariance < 0
    assert chi2 == pytest.approx(0.0, abs=1e-12)
    assert -1.0 <= rho <= 1.0
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)


def test_linear_fit_matches_weighted_uncertainty_formulas(linear_data):
    x, y, sigma_y = linear_data

    result = LinearFit(x, y, sigma_y, verbose_fit=False)
    slope, u_slope, intercept, u_intercept, covariance, *_ = result

    w = 1.0 / sigma_y**2
    sum_w = np.sum(w)
    x_w = np.sum(w * x) / sum_w
    x2_w = np.sum(w * x**2) / sum_w
    var_x = x2_w - x_w**2

    expected_u_slope = np.sqrt(1.0 / (sum_w * var_x))
    expected_u_intercept = np.sqrt(x2_w / (sum_w * var_x))
    expected_covariance = -x_w / (sum_w * var_x)

    assert u_slope == pytest.approx(expected_u_slope)
    assert u_intercept == pytest.approx(expected_u_intercept)
    assert covariance == pytest.approx(expected_covariance)


def test_linear_fit_accepts_list_inputs():
    x = [0.0, 1.0, 2.0]
    y = [1.0, 3.0, 5.0]
    sigma_y = [0.1, 0.1, 0.1]

    result = LinearFit(x, y, sigma_y, verbose_fit=False)

    assert result[0] == pytest.approx(2.0)
    assert result[2] == pytest.approx(1.0)


def test_linear_fit_verbose_options(linear_data, capsys):
    x, y, sigma_y = linear_data

    LinearFit(
        x,
        y,
        sigma_y,
        verbose_1=True,
        verbose_2=True,
        verbose_res=True,
        verbose_fit=True,
    )

    output = capsys.readouterr().out
    assert "sum_w" in output
    assert "wVAR_x" in output
    assert "e_1" in output
    assert "E_m" in output
    assert "chi2" in output


# ---------------------------------------------------------------------
# Exponential fit
# ---------------------------------------------------------------------

def test_exponential_fit_recovers_exact_model():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 3.0 * np.exp(0.4 * x)
    sigma_y = 0.05 * y

    result = ExponentialFit(x, y, sigma_y, verbose_fit_exp=False)
    slope, u_slope, amplitude, u_amplitude, covariance, chi2, rho, residuals = result

    assert slope == pytest.approx(0.4)
    assert amplitude == pytest.approx(3.0)
    assert u_slope > 0
    assert u_amplitude > 0
    assert np.isfinite(covariance)
    assert chi2 == pytest.approx(0.0, abs=1e-12)
    assert -1.0 <= rho <= 1.0
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)


def test_exponential_fit_verbose(capsys):
    x = np.array([0.0, 1.0, 2.0])
    y = 2.0 * np.exp(0.3 * x)
    sigma_y = 0.1 * y

    ExponentialFit(x, y, sigma_y, verbose_fit_exp=True)

    assert "E_m" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------

def test_powerlaw_fit_recovers_exact_model():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    y = 5.0 * x**1.5
    sigma_y = 0.05 * y

    result = PowerLawFit(x, y, sigma_y, verbose_fit_pow=False)
    exponent, u_exponent, amplitude, u_amplitude, covariance, chi2, rho, residuals = result

    assert exponent == pytest.approx(1.5)
    assert amplitude == pytest.approx(5.0)
    assert u_exponent > 0
    assert u_amplitude > 0
    assert np.isfinite(covariance)
    assert chi2 == pytest.approx(0.0, abs=1e-12)
    assert -1.0 <= rho <= 1.0
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)


def test_powerlaw_fit_verbose(capsys):
    x = np.array([1.0, 2.0, 3.0])
    y = 4.0 * x**2
    sigma_y = 0.1 * y

    PowerLawFit(x, y, sigma_y, verbose_fit_pow=True)

    assert "E_alpha" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_x, bad_y, bad_sigma, message",
    [
        ([[0.0, 1.0, 2.0]], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], "one-dimensional"),
        ([0.0, 1.0, 2.0], [[1.0, 2.0, 3.0]], [1.0, 1.0, 1.0], "one-dimensional"),
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [[1.0, 1.0, 1.0]], "one-dimensional"),
        ([0.0, 1.0], [1.0, 2.0], [1.0, 1.0], "At least three"),
        ([0.0, 1.0, 2.0], [1.0, 2.0], [1.0, 1.0, 1.0], "same length"),
        ([0.0, 1.0, np.nan], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], "finite"),
        ([0.0, 1.0, 2.0], [1.0, np.inf, 3.0], [1.0, 1.0, 1.0], "finite"),
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 0.0, 1.0], "strictly positive"),
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1.0, -1.0, 1.0], "strictly positive"),
    ],
)
def test_linear_fit_rejects_invalid_inputs(bad_x, bad_y, bad_sigma, message):
    with pytest.raises(ValueError, match=message):
        LinearFit(bad_x, bad_y, bad_sigma, verbose_fit=False)


@pytest.mark.parametrize(
    "y, sigma_y, message",
    [
        ([1.0, 0.0, 3.0], [0.1, 0.1, 0.1], "strictly positive"),
        ([1.0, -2.0, 3.0], [0.1, 0.1, 0.1], "strictly positive"),
        ([1.0, np.nan, 3.0], [0.1, 0.1, 0.1], "finite"),
        ([1.0, 2.0, 3.0], [0.1, 0.0, 0.1], "strictly positive"),
    ],
)
def test_exponential_fit_rejects_invalid_inputs(y, sigma_y, message):
    x = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match=message):
        ExponentialFit(x, y, sigma_y, verbose_fit_exp=False)


@pytest.mark.parametrize(
    "x, y, sigma_y, message",
    [
        ([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1], "strictly positive"),
        ([-1.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.1, 0.1, 0.1], "strictly positive"),
        ([1.0, 2.0, 3.0], [1.0, 0.0, 3.0], [0.1, 0.1, 0.1], "strictly positive"),
        ([1.0, 2.0, 3.0], [1.0, -2.0, 3.0], [0.1, 0.1, 0.1], "strictly positive"),
        ([1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [0.1, 0.1, 0.1], "finite"),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.0, 0.1], "strictly positive"),
    ],
)
def test_powerlaw_fit_rejects_invalid_inputs(x, y, sigma_y, message):
    with pytest.raises(ValueError, match=message):
        PowerLawFit(x, y, sigma_y, verbose_fit_pow=False)


def test_linear_fit_rejects_constant_x():
    x = np.array([2.0, 2.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    sigma_y = np.ones(3)

    with pytest.raises(ValueError, match="variance of x"):
        LinearFit(x, y, sigma_y, verbose_fit=False)


def test_exponential_fit_rejects_constant_x():
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    sigma_y = np.ones(3)

    with pytest.raises(ValueError, match="variance of x"):
        ExponentialFit(x, y, sigma_y, verbose_fit_exp=False)


def test_powerlaw_fit_rejects_constant_x():
    x = np.array([2.0, 2.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    sigma_y = np.ones(3)

    with pytest.raises(ValueError, match="variance of x"):
        PowerLawFit(x, y, sigma_y, verbose_fit_pow=False)
