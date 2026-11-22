from pathlib import Path

import pandas as pd

def read_ascii(
    path: str | Path,
    columns: list[str] | None = None,
    skiprows: int = 0,
    comment: str = "#",
    na_values: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Read a numerical whitespace-separated ASCII file into a DataFrame.

    The function is intended for the standard ASCII data files used in
    the analysis. Files are assumed to contain numerical data only, with
    integer values treated as floating-point values.

    Parameters
    ----------
    path : str or Path
        Path to the ASCII data file.

    columns : list of str, optional
        Names assigned to the columns of the resulting DataFrame.
        If None, pandas assigns integer column names.

    skiprows : int, optional
        Number of lines to skip at the beginning of the file.
        Defaults to 0.

    comment : str, optional
        Character used to indicate comment lines. Lines beginning with
        this character are ignored. Defaults to '#'.

    na_values : str or list of str, optional
        Additional strings to recognize as missing values.
        For example, 'NaN', '-' or ['NaN', '-'].
        Defaults to None.

    Returns
    -------
    pandas.DataFrame
        Data contained in the ASCII file.
    """

    return pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=skiprows,
        comment=comment,
        header=None,
        names=columns,
        dtype=float,
        na_values=na_values,
    )
