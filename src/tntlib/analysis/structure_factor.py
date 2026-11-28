import numpy as np
import pandas as pd



def radial_structure_factor(df: pd.DataFrame, nbins: int = 100) -> pd.DataFrame:
    """
    Compute the radial average of a structure factor.

    The first two columns must be kx and ky.
    The magnitude k is calculated and placed as the first
    column of the output.
    """

    # Check input
    if df.shape[1] < 2:
        raise ValueError(
            "Structure factor must contain at least two columns: "
            "'kx' and 'ky'."
        )

    if list(df.columns[:2]) != ["kx", "ky"]:
        raise ValueError(
            "The first two columns of the structure factor must be "
            "'kx' and 'ky'."
        )

    if not isinstance(nbins, (int, np.integer)) or isinstance(nbins, bool) or nbins < 1:
        raise ValueError("nbins must be a positive integer.")

    # Work on a copy
    df = df.copy()

    # Compute wavevector magnitude
    df["k"] = np.sqrt(df["kx"] ** 2 + df["ky"] ** 2)

    # Construct kinetic energy if not already present
    if "Ekin" not in df.columns:
        if {"vx", "vy"}.issubset(df.columns):
            df["Ekin"] = 0.5 * (df["vx"] ** 2 + df["vy"] ** 2)

        elif {"ux", "uy"}.issubset(df.columns):
            df["Ekin"] = 0.5 * (df["ux"] ** 2 + df["uy"] ** 2)

        elif {"u", "v"}.issubset(df.columns):
            df["Ekin"] = 0.5 * (df["u"] ** 2 + df["v"] ** 2)

    # Create radial bins
    bins = np.linspace(df["k"].min(), df["k"].max(), nbins + 1)

    df["bin"] = pd.cut(
        df["k"],
        bins=bins,
        labels=False,
        include_lowest=True
    )

    # Select columns to average, keeping k first
    columns = ["k"] + [
        col for col in df.columns
        if col not in ["kx", "ky", "k", "bin"]
    ]

    # Radial averaging
    result = df.groupby("bin")[columns].mean()

    # Remove empty bins and reset index
    result = result.dropna(how="all").reset_index(drop=True)

    return result


