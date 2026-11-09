import numpy as np
import pandas as pd


def radial_structure_factor(df: pd.DataFrame,
                            nbins: int = 100,) -> pd.DataFrame:
    """
    Compute the radial average of a structure factor.

    The first two columns of the input DataFrame must be ``kx`` and ``ky``.
    A ``k`` column containing the magnitude of the wavevector is added to
    the DataFrame. All remaining columns are then averaged over equally
    spaced radial bins.

    If the DataFrame contains ``vx`` and ``vy`` but does not contain
    ``Ekin``, the kinetic energy is constructed as

    .. math::

        E_{kin} = \\frac{1}{2}(v_x^2 + v_y^2).

    Parameters
    ----------
    df : pandas.DataFrame
        Structure-factor data. The first two columns must be ``kx`` and
        ``ky``.

    nbins : int, optional
        Number of equally spaced radial bins. Defaults to 100.

    Returns
    -------
    pandas.DataFrame
        Radially averaged structure factor with ``k`` as the first column,
        followed by the radial averages of all observables.

    Raises
    ------
    ValueError
        If the first two columns are not ``kx`` and ``ky``.
    """

    # Check that the DataFrame contains at least kx and ky.
    if df.shape[1] < 2:
        raise ValueError(
            "Structure factor must contain at least two columns: "
            "'kx' and 'ky'."
        )

    # The first two columns must be kx and ky.
    if list(df.columns[:2]) != ["kx", "ky"]:
        raise ValueError(
            "The first two columns of the structure factor must be "
            "'kx' and 'ky'."
        )

    # Work on a copy so that the input DataFrame is not modified.
    df = df.copy()

    # Compute the magnitude of the wavevector.
    df["k"] = np.sqrt(df["kx"] ** 2 + df["ky"] ** 2)

    # Construct the kinetic energy if the velocity components are available
    # but the kinetic energy is not already present.
    # Possible names (vx, vy) - (ux, uy) - (u,v)
    if "vx" in df.columns and "vy" in df.columns and "Ekin" not in df.columns:
        df["Ekin"] = 0.5 * (df["vx"] ** 2 + df["vy"] ** 2)

    if "ux" in df.columns and "uy" in df.columns and "Ekin" not in df.columns:
        df["Ekin"] = 0.5 * (df["ux"] ** 2 + df["uy"] ** 2)

    if "u" in df.columns and "v" in df.columns and "Ekin" not in df.columns:
        df["Ekin"] = 0.5 * (df["u"] ** 2 + df["v"] ** 2)

    # Create equally spaced radial bins.
    bins = np.linspace(df["k"].min(), df["k"].max(), nbins + 1)

    # Assign each point to a radial bin.
    df["bin"] = pd.cut(df["k"], bins=bins, labels=False)

    # Radially average all columns except the wavevector components
    # and the temporary bin index.
    columns = [col for col in df.columns if col not in ["kx", "ky", "bin"]]
    grouped = df.groupby("bin")

    result = grouped[columns].mean()

    # Remove empty bins and reset the index.
    return result.dropna(how="all").reset_index(drop=True)

