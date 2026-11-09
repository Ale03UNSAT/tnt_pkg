import numpy as np

def hex_to_rgb(hex_color: str,
               normalized: bool = False,) -> tuple[int, int, int] | tuple[float, float, float]:
    """
    Convert a hexadecimal color string to an RGB tuple.

    Parameters
    ----------
    hex_color : str
        Hexadecimal color in the form ``"#RRGGBB"`` or ``"RRGGBB"``.

    normalized : bool, optional
        If True, return RGB values normalized to the range [0, 1].
        Otherwise, return integer RGB values in the range [0, 255].
        Defaults to False.

    Returns
    -------
    tuple of int or float
        RGB values either in the range [0, 255] or [0, 1], depending
        on ``normalized``.

    Raises
    ------
    ValueError
        If the hexadecimal color is not a valid six-digit color.
    """

    # Remove the optional '#' prefix.
    hex_color = hex_color.lstrip("#")

    # Check that the color contains exactly six hexadecimal digits.
    if len(hex_color) != 6:
        raise ValueError(
            "Hexadecimal color must contain exactly 6 digits."
        )

    try:
        # Convert each pair of hexadecimal digits to an integer.
        rgb = tuple(
            int(hex_color[i:i + 2], 16)
            for i in (0, 2, 4)
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid hexadecimal color: {hex_color!r}"
        ) from exc

    if normalized:
        # Normalize RGB values from [0, 255] to [0, 1].
        return tuple(value / 255 for value in rgb)

    return rgb


def rgb_to_hex(rgb: tuple[int, int, int] | tuple[float, float, float],
               normalized: bool = False,) -> str:
    """
    Convert an RGB tuple to a hexadecimal color string.

    Parameters
    ----------
    rgb : tuple of int or float
        RGB values. If ``normalized=False``, values must be integers
        in the range [0, 255]. If ``normalized=True``, values must
        be floats in the range [0, 1].

    normalized : bool, optional
        If True, interpret RGB values as normalized values in [0, 1].
        Otherwise, interpret them as integer values in [0, 255].
        Defaults to False.

    Returns
    -------
    str
        Hexadecimal color in the form ``"#RRGGBB"``.
    """

    # Check that exactly three RGB values are provided.
    if len(rgb) != 3:
        raise ValueError(
            "RGB color must contain exactly three values."
        )

    if normalized:
        # Check that normalized RGB values are in the valid range.
        if not all(0 <= value <= 1 for value in rgb):
            raise ValueError(
                "Normalized RGB values must be in the range [0, 1]."
            )

        # Convert normalized values to the integer range [0, 255].
        rgb = tuple(round(value * 255) for value in rgb)

    else:
        # Check that RGB values are integers in the valid range.
        if not all(
            isinstance(value, int) and 0 <= value <= 255
            for value in rgb
        ):
            raise ValueError(
                "RGB values must be integers in the range [0, 255]."
            )

    # Convert each RGB value to a two-digit hexadecimal representation.
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def make_gradient(colors: list[str],
                  ncolors: int,
                  power: float = 1.0,) -> list[str]:
    """
    Create a power-adjusted linear color gradient between a list of colors.

    The colors are interpolated linearly in RGB space. If more than two
    colors are provided, the interpolation is performed piecewise between
    consecutive colors. The distribution of the generated colors along
    the gradient is controlled by ``power``.

    Parameters
    ----------
    colors : list of str
        List of hexadecimal colors defining the gradient. At least two
        colors must be provided.

    ncolors : int
        Number of colors in the resulting gradient.

    power : float, optional
        Power applied to the positions of the generated colors.
        The positions are computed as
        ``np.linspace(0, 1, ncolors) ** power``.
        A value of 1 gives a uniform linear gradient. Values greater
        than 1 concentrate colors towards the first color, while values
        between 0 and 1 concentrate colors towards the last color.
        Defaults to 1.

    Returns
    -------
    list of str
        List containing ``ncolors`` hexadecimal colors.

    Raises
    ------
    ValueError
        If fewer than two colors are provided, if ``ncolors`` is less
        than two, or if ``power`` is not positive.
    """

    if len(colors) < 2:
        raise ValueError(
            "At least two colors are required to create a gradient."
        )

    if ncolors < 2:
        raise ValueError(
            "The number of colors must be at least 2."
        )

    if power <= 0:
        raise ValueError(
            "The power must be greater than 0."
        )

    # Convert the input colors to normalized RGB values.
    rgb_colors = np.array([
        hex_to_rgb(color, normalized=True)
        for color in colors
    ])

    # Position of each input color along the gradient.
    positions = np.linspace(0, 1, len(colors))

    # Generate the positions of the colors to create and apply the power.
    gradient_positions = np.linspace(0, 1, ncolors) ** power

    # Interpolate each RGB component independently.
    gradient_rgb = np.column_stack([
        np.interp(gradient_positions, positions, rgb_colors[:, i])
        for i in range(3)
    ])

    # Convert the interpolated RGB values back to hexadecimal colors.
    return [
        rgb_to_hex(tuple(rgb), normalized=True)
        for rgb in gradient_rgb
    ]
