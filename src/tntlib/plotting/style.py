"""
Matplotlib style configuration for the AP scientific plotting package.

This module defines the standard Matplotlib rcParams used by the AP research
group for scientific figures and publications.
"""

import matplotlib as mpl


# Conversion factor
MM_TO_PT = 72.0 / 25.4


# ---------------------------------------------------------------------------
# AP Matplotlib style
# ---------------------------------------------------------------------------

TNT_RCPARAMS = {
    # ---------- Figure ----------
    "savefig.format": "eps",
    "savefig.bbox": "tight",

    # ---------- Font ----------
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "ps.useafm": False,
    "pdf.use14corefonts": False,
    "mathtext.fontset": "stixsans",

    # ---------- Font sizes ----------
    "font.size": 34,
    "axes.titlesize": 34,
    "axes.labelsize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 40,
    "figure.titlesize": 34,

    # ---------- Axes ----------
    "axes.linewidth": 2.0,
    "axes.grid": False,

    # ---------- Lines ----------
    "lines.linewidth": 1.0 * MM_TO_PT,
    "lines.markersize": 10,

    # ---------- EPS / PS text handling ----------
    "ps.fonttype": 42,

    # ---------- Major ticks ----------
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,

    # ---------- Minor ticks ----------
    "xtick.minor.size": 6,
    "ytick.minor.size": 6,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_style(Params=TNT_RCPARAMS):
    """
    Apply the standard AP Matplotlib style.

    This function modifies ``matplotlib.rcParams`` globally.
    """
    mpl.rcParams.update(Params)

    
def reset_style():
    """
    Reset Matplotlib to its default rcParams.
    """
    mpl.rcdefaults()


def get_style():
    """
    Return a copy of the AP Matplotlib style configuration.

    Returns
    -------
    dict
        Dictionary containing the AP rcParams.
    """
    return TNT_RCPARAMS.copy()


import numpy as np


def scale_style(PARAMs, ax_width, ax_height,
                ref_width=9, ref_height=7):
    """
    Scale numeric plotting parameters according to the axes dimensions.

    Parameters
    ----------
    PARAMs : dict
        Base Matplotlib style parameters.

    ax_width : float
        Width of the axes in inches.

    ax_height : float
        Height of the axes in inches.

    ref_width : float, default=9
        Reference axes width in inches.

    ref_height : float, default=7
        Reference axes height in inches.

    Returns
    -------
    scaled_params : dict
        Scaled plotting parameters.
    """
    scaled_params = PARAMs.copy()

    # Calculate the scaling factors
    x_ratio = ax_width / ref_width
    y_ratio = ax_height / ref_height
    scale = np.sqrt(x_ratio * y_ratio)

    # Parameters scaled with the figure dimensions
    scale_keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
        "axes.linewidth",
        "lines.linewidth",
        "lines.markersize",
        "xtick.major.size",
        "ytick.major.size",
        "xtick.major.width",
        "ytick.major.width",
        "xtick.major.pad",
        "ytick.major.pad",
        "xtick.minor.size",
        "ytick.minor.size",
        "xtick.minor.width",
        "ytick.minor.width",
    ]

    # Scale the selected numeric parameters
    for key in scale_keys:
        value = scaled_params.get(key)

        if isinstance(value, (int, float, np.number)):
            scaled_params[key] = value * scale

    return scaled_params

def print_style():
    """
    Print the AP Matplotlib style configuration.

    The values shown are the values defined by AP_RCPARAMS, not necessarily
    the values currently active in Matplotlib.
    """
    print("AP Matplotlib style")
    print("-" * 60)

    for key, value in TNT_RCPARAMS.items():
        print(f"{key:<25} = {value}")

