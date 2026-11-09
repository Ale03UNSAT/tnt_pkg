import numpy as np


#########################
# Weighted mean function #
#########################
def weighted_mean(x, w):
    """
    Calculate the weighted mean.

    Parameters
    ----------
    x : array_like
        Input values.

    w : array_like
        Statistical weights associated with the input values.

    Returns
    -------
    mean : float
        Weighted mean of the input values.
    """
    return np.sum(x * w) / np.sum(w)


##########################################
# Covariance function for linear fitting #
##########################################
def lin_cov(x, y, w):
    """
    Calculate the weighted covariance for linear fitting.

    Parameters
    ----------
    x : array_like
        First set of values.

    y : array_like
        Second set of values.

    w : array_like
        Statistical weights associated with the input values.

    Returns
    -------
    cov : float
        Weighted covariance of x and y.
    """
    return weighted_mean(x * y, w) - weighted_mean(x, w) * weighted_mean(y, w)


######################################
# Variance function for linear fitting #
######################################
def lin_var(x, w):
    """
    Calculate the weighted variance for linear fitting.

    Parameters
    ----------
    x : array_like
        Input values.

    w : array_like
        Statistical weights associated with the input values.

    Returns
    -------
    var : float
        Weighted variance of x.
    """
    return lin_cov(x, x, w)


############################
# Linear residual functions #
############################
def res_lin(x, y, E_m, E_c, sigma_y, verbose=False):
    """
    Calculate the normalized residuals of a linear fit.

    Parameters
    ----------
    x : array_like
        Independent variable values.

    y : array_like
        Measured dependent variable values.

    E_m : float
        Estimated slope of the fitted line.

    E_c : float
        Estimated intercept of the fitted line.

    sigma_y : array_like
        Standard uncertainties associated with y.

    verbose : bool, default=False
        If True, print each normalized residual.

    Returns
    -------
    residuals : ndarray
        Normalized residuals of the fit.
    """
    residuals = (y - E_m * x - E_c) / sigma_y

    if verbose:
        for i, residual in enumerate(residuals, start=1):
            print(f'e_{i} = {residual}')

    return residuals


#######################
# Chi-square function #
#######################
def chi2_lin(x, y, E_m, E_c, sigma_y):
    """
    Calculate the chi-square statistic for a linear fit.

    Parameters
    ----------
    x : array_like
        Independent variable values.

    y : array_like
        Measured dependent variable values.

    E_m : float
        Estimated slope of the fitted line.

    E_c : float
        Estimated intercept of the fitted line.

    sigma_y : array_like
        Standard uncertainties associated with y.

    Returns
    -------
    chi2 : float
        Chi-square statistic of the fit.
    """
    residuals = res_lin(x, y, E_m, E_c, sigma_y)
    return np.sum(residuals**2)


##################
# Line function #
##################
def _line(x, m, c):
    """
    Calculate the values of a straight line.

    Parameters
    ----------
    x : array_like
        Independent variable values.

    m : float
        Slope of the line.

    c : float
        Intercept of the line.

    Returns
    -------
    y : array_like
        Values of the straight line.
    """
    return m * x + c


##############################
# Exponential model function #
##############################
def _exponential(x, m, c):
    """
    Calculate the values of an exponential model.

    Parameters
    ----------
    x : array_like
        Independent variable values.

    m : float
        Exponential coefficient.

    c : float
        Amplitude of the exponential.

    Returns
    -------
    y : array_like
        Values of the exponential model.
    """
    return c * np.exp(m * x)


############################
# Power-law model function #
############################
def _powerlaw(x, alpha, A):
    """
    Calculate the values of a power-law model.

    Parameters
    ----------
    x : array_like
        Independent variable values.

    alpha : float
        Power-law exponent.

    A : float
        Amplitude of the power law.

    Returns
    -------
    y : array_like
        Values of the power-law model.
    """
    return A * x**alpha


##############################
# Uncertainty on the fitted line #
##############################
def y_unc(xP, u_m, u_c, cov_mc):
    """
    Calculate the uncertainty on the fitted mean line.

    Parameters
    ----------
    xP : array_like
        Independent variable values at which the uncertainty is evaluated.

    u_m : float
        Standard uncertainty of the fitted slope.

    u_c : float
        Standard uncertainty of the fitted intercept.

    cov_mc : float
        Covariance between the fitted slope and intercept.

    Returns
    -------
    u_y : ndarray or float
        Standard uncertainty on the fitted mean line, accounting for
        the covariance between the fitted parameters.

    Notes
    -----
    This function calculates parameter uncertainty only. It does not
    include the intrinsic scatter of the observations or the uncertainty
    of a future individual measurement.
    """
    variance = xP**2 * u_m**2 + u_c**2 + 2 * xP * cov_mc
    return np.sqrt(np.maximum(variance, 0))


###########################
# Linear fitting function #
###########################
def LinearFit(x, y, sigma_y,
              verbose_1=False,
              verbose_2=False,
              verbose_res=False,
              verbose_fit=True):
    """
    Perform a weighted linear least-squares fit.

    The fitted model is y = m*x + c, where the uncertainties on y
    are assumed to be known, independent, and Gaussian. Uncertainties
    on x are assumed to be negligible.

    Parameters
    ----------
    x : array_like
        Independent variable values.

    y : array_like
        Measured dependent variable values.

    sigma_y : array_like
        Standard uncertainties associated with y. All values must
        be finite and strictly positive.

    verbose_1 : bool, default=False
        If True, print the weighted means and the sum of the weights.

    verbose_2 : bool, default=False
        If True, print the weighted variance and covariance.

    verbose_res : bool, default=False
        If True, print the normalized residuals.

    verbose_fit : bool, default=True
        If True, print the fitted parameters and statistical results.

    Returns
    -------
    E_m : float
        Estimated slope.

    U_m : float
        Standard uncertainty of the slope.

    E_c : float
        Estimated intercept.

    U_c : float
        Standard uncertainty of the intercept.

    COV_mc : float
        Covariance between the slope and the intercept.

    chi2 : float
        Chi-square statistic of the fit.

    rho_mc : float
        Correlation coefficient between the slope and the intercept.

    res : ndarray
        Normalized residuals.

    Notes
    -----
    The statistical weights are defined as 1 / sigma_y**2.

    The parameter uncertainties are calculated assuming that the
    measurement uncertainties are known in absolute terms. They are
    not rescaled by the reduced chi-square.
    """

    # Convert the input data to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)

    # Validate the input data
    if x.ndim != 1 or y.ndim != 1 or sigma_y.ndim != 1:
        raise ValueError('x, y, and sigma_y must be one-dimensional arrays.')

    if not (x.size == y.size == sigma_y.size):
        raise ValueError('x, y, and sigma_y must have the same length.')

    if x.size < 3:
        raise ValueError('At least three data points are required.')

    if not (np.all(np.isfinite(x)) and
            np.all(np.isfinite(y)) and
            np.all(np.isfinite(sigma_y))):
        raise ValueError('All input values must be finite.')

    if np.any(sigma_y <= 0):
        raise ValueError('All uncertainties must be strictly positive.')

    # Calculate the statistical weights
    w = sigma_y**-2
    sum_w = np.sum(w)

    # Calculate the weighted means
    x_w = weighted_mean(x, w)
    y_w = weighted_mean(y, w)
    xy_w = weighted_mean(x * y, w)
    x2_w = weighted_mean(x**2, w)

    # Print the basic weighted quantities
    if verbose_1:
        print(f'sum_w = {sum_w}\n'
              f's_tot = {np.sqrt(1 / sum_w)}\n'
              f'x_w = {x_w}\n'
              f'y_w = {y_w}\n'
              f'xy_w = {xy_w}\n'
              f'x2_w = {x2_w}\n')

    # Calculate the weighted variance and covariance
    wvar_x = x2_w - x_w**2
    wcov_xy = xy_w - x_w * y_w

    # Print the weighted variance and covariance
    if verbose_2:
        print(f'wVAR_x = {wvar_x}\n'
              f'wCOV_xy = {wcov_xy}\n')

    if not np.isfinite(wvar_x) or wvar_x <= 0:
        raise ValueError('The weighted variance of x must be positive.')

    # Calculate the slope
    E_m = wcov_xy / wvar_x
    VAR_m = 1 / (sum_w * wvar_x)
    U_m = np.sqrt(VAR_m)

    # Calculate the intercept
    E_c = y_w - E_m * x_w
    VAR_c = x2_w / (sum_w * wvar_x)
    U_c = np.sqrt(VAR_c)

    # Calculate the covariance between the slope and intercept
    COV_mc = -x_w / (sum_w * wvar_x)

    # Calculate the correlation coefficient
    rho_mc = COV_mc / (U_m * U_c)

    # Calculate the chi-square statistic
    chi2 = chi2_lin(x, y, E_m, E_c, sigma_y)

    # Calculate the normalized residuals
    res = res_lin(x, y, E_m, E_c, sigma_y, verbose=verbose_res)

    # Print the fitting results
    if verbose_fit:
        print(f'E_m = {E_m:.4g}\n'
              f'VAR_m = {VAR_m:.4g}\n'
              f'U_m = {U_m:.4g}\n'
              f'E_c = {E_c:.4g}\n'
              f'VAR_c = {VAR_c:.4g}\n'
              f'U_c = {U_c:.4g}\n'
              f'COV_mc = {COV_mc:.4g}\n'
              f'chi2 = {chi2:.4g}\n'
              f'rho_mc = {rho_mc:.4g}\n')

    return E_m, U_m, E_c, U_c, COV_mc, chi2, rho_mc, res


################################
# Exponential fitting function #
################################
def ExponentialFit(x, y, sigma_y,
                    verbose_1_exp=False,
                    verbose_2_exp=False,
                    verbose_res_exp=False,
                    verbose_fit_exp=True):
    """
    Perform an exponential fit of the form y = c * exp(m*x).

    The fit is performed by linearizing the model:

        ln(y) = m*x + ln(c)

    Parameters
    ----------
    x : array_like
        Independent variable values.

    y : array_like
        Measured dependent variable values. All values must be positive.

    sigma_y : array_like
        Standard uncertainties associated with y. All values must
        be finite and strictly positive.

    verbose_1_exp : bool, default=False
        If True, print the basic weighted quantities.

    verbose_2_exp : bool, default=False
        If True, print the weighted variance and covariance.

    verbose_res_exp : bool, default=False
        If True, print the normalized residuals in logarithmic space.

    verbose_fit_exp : bool, default=True
        If True, print the fitting results.

    Returns
    -------
    E_m : float
        Estimated exponential coefficient.

    U_m : float
        Standard uncertainty of the exponential coefficient.

    E_c : float
        Estimated amplitude.

    U_c : float
        Standard uncertainty of the amplitude.

    COV_mc : float
        Covariance between the exponential coefficient and amplitude.

    chi2 : float
        Chi-square statistic calculated in logarithmic space.

    rho_mc : float
        Correlation coefficient between the fitted parameters.

    res : ndarray
        Normalized residuals in logarithmic space.

    Notes
    -----
    The uncertainties are propagated using first-order error propagation.

    The fit assumes negligible uncertainty on x. The logarithmic
    transformation is not, in general, an exact maximum-likelihood
    method for measurements with additive Gaussian errors in y.
    """

    # Convert the input data to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)

    # Validate the input data
    if x.ndim != 1 or y.ndim != 1 or sigma_y.ndim != 1:
        raise ValueError('x, y, and sigma_y must be one-dimensional arrays.')

    if not (x.size == y.size == sigma_y.size):
        raise ValueError('x, y, and sigma_y must have the same length.')

    if x.size < 3:
        raise ValueError('At least three data points are required.')

    if not (np.all(np.isfinite(x)) and
            np.all(np.isfinite(y)) and
            np.all(np.isfinite(sigma_y))):
        raise ValueError('All input values must be finite.')

    if np.any(y <= 0):
        raise ValueError('All y values must be strictly positive.')

    if np.any(sigma_y <= 0):
        raise ValueError('All uncertainties must be strictly positive.')

    # Transform y and propagate its uncertainties
    ln_y = np.log(y)
    sigma_ln_y = sigma_y / y

    # Perform the linear fit in logarithmic space
    E_m, U_m, ln_c, U_ln_c, COV_m_ln_c, chi2, rho_mc, res = LinearFit(
        x, ln_y, sigma_ln_y,
        verbose_1=verbose_1_exp,
        verbose_2=verbose_2_exp,
        verbose_res=verbose_res_exp,
        verbose_fit=False,
    )

    # Transform the fitted intercept back to the amplitude
    E_c = np.exp(ln_c)
    U_c = E_c * U_ln_c

    # Transform the covariance to the original parameter space
    COV_mc = E_c * COV_m_ln_c

    # Print the fitting results
    if verbose_fit_exp:
        print(f'E_m = {E_m:.4g}\n'
              f'U_m = {U_m:.4g}\n'
              f'E_c = {E_c:.4g}\n'
              f'U_c = {U_c:.4g}\n'
              f'COV_mc = {COV_mc:.4g}\n'
              f'chi2 = {chi2:.4g}\n'
              f'rho_mc = {rho_mc:.4g}\n')

    return E_m, U_m, E_c, U_c, COV_mc, chi2, rho_mc, res


############################
# Power-law fitting function #
############################
def PowerLawFit(x, y, sigma_y,
                verbose_1_pow=False,
                verbose_2_pow=False,
                verbose_res_pow=False,
                verbose_fit_pow=True):
    """
    Perform a power-law fit of the form y = A * x**alpha.

    The fit is performed by linearizing the model:

        ln(y) = alpha*ln(x) + ln(A)

    Parameters
    ----------
    x : array_like
        Independent variable values. All values must be positive.

    y : array_like
        Measured dependent variable values. All values must be positive.

    sigma_y : array_like
        Standard uncertainties associated with y. All values must
        be finite and strictly positive.

    verbose_1_pow : bool, default=False
        If True, print the basic weighted quantities.

    verbose_2_pow : bool, default=False
        If True, print the weighted variance and covariance.

    verbose_res_pow : bool, default=False
        If True, print the normalized residuals in logarithmic space.

    verbose_fit_pow : bool, default=True
        If True, print the fitting results.

    Returns
    -------
    E_alpha : float
        Estimated power-law exponent.

    U_alpha : float
        Standard uncertainty of the exponent.

    E_A : float
        Estimated amplitude.

    U_A : float
        Standard uncertainty of the amplitude.

    COV_alpha_A : float
        Covariance between the exponent and amplitude.

    chi2 : float
        Chi-square statistic calculated in logarithmic space.

    rho_alpha_A : float
        Correlation coefficient between the fitted parameters.

    res : ndarray
        Normalized residuals in logarithmic space.

    Notes
    -----
    The uncertainties are propagated using first-order error propagation.

    The fit assumes negligible uncertainty on x. The logarithmic
    transformation is not, in general, an exact maximum-likelihood
    method for measurements with additive Gaussian errors in y.
    """

    # Convert the input data to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma_y = np.asarray(sigma_y, dtype=float)

    # Validate the input data
    if x.ndim != 1 or y.ndim != 1 or sigma_y.ndim != 1:
        raise ValueError('x, y, and sigma_y must be one-dimensional arrays.')

    if not (x.size == y.size == sigma_y.size):
        raise ValueError('x, y, and sigma_y must have the same length.')

    if x.size < 3:
        raise ValueError('At least three data points are required.')

    if not (np.all(np.isfinite(x)) and
            np.all(np.isfinite(y)) and
            np.all(np.isfinite(sigma_y))):
        raise ValueError('All input values must be finite.')

    if np.any(x <= 0):
        raise ValueError('All x values must be strictly positive.')

    if np.any(y <= 0):
        raise ValueError('All y values must be strictly positive.')

    if np.any(sigma_y <= 0):
        raise ValueError('All uncertainties must be strictly positive.')

    # Transform x, y, and propagate the uncertainties
    ln_x = np.log(x)
    ln_y = np.log(y)
    sigma_ln_y = sigma_y / y

    # Perform the linear fit in logarithmic space
    E_alpha, U_alpha, ln_A, U_ln_A, COV_alpha_ln_A, chi2, rho_alpha_A, res = LinearFit(
        ln_x, ln_y, sigma_ln_y,
        verbose_1=verbose_1_pow,
        verbose_2=verbose_2_pow,
        verbose_res=verbose_res_pow,
        verbose_fit=False,
    )

    # Transform the fitted intercept back to the amplitude
    E_A = np.exp(ln_A)
    U_A = E_A * U_ln_A

    # Transform the covariance to the original parameter space
    COV_alpha_A = E_A * COV_alpha_ln_A

    # Print the fitting results
    if verbose_fit_pow:
        print(f'E_alpha = {E_alpha:.4g}\n'
              f'U_alpha = {U_alpha:.4g}\n'
              f'E_A = {E_A:.4g}\n'
              f'U_A = {U_A:.4g}\n'
              f'COV_alpha_A = {COV_alpha_A:.4g}\n'
              f'chi2 = {chi2:.4g}\n'
              f'rho_alpha_A = {rho_alpha_A:.4g}\n')

    return E_alpha, U_alpha, E_A, U_A, COV_alpha_A, chi2, rho_alpha_A, res
