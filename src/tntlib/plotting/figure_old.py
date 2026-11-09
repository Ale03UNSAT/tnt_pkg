"""
Figure and axes utilities for AP scientific plotting.
"""
from collections.abc import Sequence

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, HPacker, VPacker
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullLocator, AutoMinorLocator, FixedFormatter, NullFormatter
from mpl_toolkits.axes_grid1 import Divider, Size

from pathlib import Path

# Standard AP divider region in figure coordinates
AP_DIVIDER_RECT = (0.15, 0.15, 0.8, 0.8)

def create_figure(ax_width=9,
                  ax_height=7,
                  margin=2,
                  dpi=600,):
    """
    Create a figure with a fixed-size axes.
    
    Parameters
    ----------
    ax_width : float, default=9
        Width of the axes in inches.

    ax_height : float, default=7
        Height of the axes in inches.

    margin : float, default=2
        Additional figure size in inches.

    dpi : int, default=600
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    ax : matplotlib.axes.Axes
        The created axes.
    """

    fig = plt.figure(figsize=(ax_width + margin, ax_height + margin),
                     dpi=dpi)

    h = [Size.Fixed(ax_width)]
    v = [Size.Fixed(ax_height)]

    divider = Divider(fig,
                      AP_DIVIDER_RECT,
                      h,
                      v)

    ax = fig.add_axes(divider.get_position(),
                      axes_locator=divider.new_locator(nx=0, ny=0))

    ax.set_autoscale_on(True)

    return fig, ax



def setup_axes(ax,
               xlim=None,
               ylim=None,
               xleft=None,
               xright=None,
               ybottom=None,
               ytop=None,
               xscale="linear",
               yscale="linear",
               xmajor_ticks=None,
               xminor_step=None,
               ymajor_ticks=None,
               yminor_step=None):
    """
    Configure limits, scales, and ticks for an AP-style axes.
    """

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if xscale not in ("linear", "log"):
        raise ValueError("xscale must be either 'linear' or 'log'.")

    if yscale not in ("linear", "log"):
        raise ValueError("yscale must be either 'linear' or 'log'.")

    if xmajor_ticks is not None and not isinstance(xmajor_ticks, list):
        raise TypeError("xmajor_ticks must be a list.")

    if ymajor_ticks is not None and not isinstance(ymajor_ticks, list):
        raise TypeError("ymajor_ticks must be a list.")

    # ------------------------------------------------------------------
    # Set scales
    # ------------------------------------------------------------------

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # ------------------------------------------------------------------
    # Autoscale
    # ------------------------------------------------------------------

    ax.relim()
    ax.autoscale_view()

    # ------------------------------------------------------------------
    # X limits
    # ------------------------------------------------------------------

    if xlim is not None:
        xleft, xright = xlim

    current_xleft, current_xright = ax.get_xlim()

    if xleft is None:
        xleft = current_xleft

    if xright is None:
        xright = current_xright

    ax.set_xlim(xleft, xright)

    # ------------------------------------------------------------------
    # Y limits
    # ------------------------------------------------------------------

    if ylim is not None:
        ybottom, ytop = ylim

    current_ybottom, current_ytop = ax.get_ylim()

    if ybottom is None:
        ybottom = current_ybottom

    if ytop is None:
        ytop = current_ytop

    ax.set_ylim(ybottom, ytop)

    # ------------------------------------------------------------------
    # Major ticks
    # ------------------------------------------------------------------

    if ymajor_ticks == "null":
        ax.yaxis.set_major_locator(NullLocator())
    elif xmajor_ticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(xmajor_ticks))
    #Else the default locator

    if ymajor_ticks == "null":
        ax.yaxis.set_major_locator(NullLocator())
    elif ymajor_ticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(ymajor_ticks))
    #Else the default locator
        
    # ------------------------------------------------------------------
    # Minor ticks
    # ------------------------------------------------------------------

    if xscale == "linear":
        if xminor_step is not None:
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_minor_locator(MultipleLocator(xminor_step))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
    else:
        if xminor_step is not None:
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
        else:
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
        ax.xaxis.set_minor_formatter(NullFormatter())

    if yscale == "linear":
        if yminor_step is not None:
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_locator(MultipleLocator(yminor_step))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())
    else:
        if xminor_step is not None:
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9]))            
        else:
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
        ax.yaxis.set_minor_formatter(NullFormatter())

"""
Utilities for customizing tick labels.
"""
def set_tick_labels(ax,
                    x_ticks=None,
                    x_labels=None,
                    y_ticks=None,
                    y_labels=None,):
    """
    Set custom labels for major ticks on the x and/or y axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to configure.

    x_ticks : list, optional
        Positions of the x-axis ticks.

    x_labels : list, optional
        Labels corresponding to ``x_ticks``.

    y_ticks : list, optional
        Positions of the y-axis ticks.

    y_labels : list, optional
        Labels corresponding to ``y_ticks``.
    """

    # ------------------------------------------------------------------
    # X-axis
    # ------------------------------------------------------------------

    if x_ticks is not None or x_labels is not None:

        if x_ticks is None or x_labels is None:
            raise ValueError(
                "x_ticks and x_labels must be provided together."
            )

        if not isinstance(x_ticks, list):
            raise TypeError("x_ticks must be a list.")

        if not isinstance(x_labels, list):
            raise TypeError("x_labels must be a list.")

        if len(x_ticks) != len(x_labels):
            raise ValueError(
                "x_ticks and x_labels must have the same length."
            )

        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(x_labels))

    # ------------------------------------------------------------------
    # Y-axis
    # ------------------------------------------------------------------

    if y_ticks is not None or y_labels is not None:

        if y_ticks is None or y_labels is None:
            raise ValueError(
                "y_ticks and y_labels must be provided together."
            )

        if not isinstance(y_ticks, list):
            raise TypeError("y_ticks must be a list.")

        if not isinstance(y_labels, list):
            raise TypeError("y_labels must be a list.")

        if len(y_ticks) != len(y_labels):
            raise ValueError(
                "y_ticks and y_labels must have the same length."
            )

        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(y_labels))


def make_legend_handles(handles: Sequence[Artist],
                        labels: Sequence[str],) -> list[tuple[Artist, str]]:
    """
    Create a list of legend handle-label pairs from plotted artists.

    Parameters
    ----------
    handles : sequence of Artist
        Matplotlib artists to use as legend handles.

    labels : sequence of str
        Labels corresponding to each legend handle.

    Returns
    -------
    list of tuple
        List of ``(artist, label)`` pairs suitable for passing to
        ``make_legend()``.

    Raises
    ------
    ValueError
        If the number of handles and labels is different.
    """

    # Check that every handle has a corresponding label.
    if len(handles) != len(labels):
        raise ValueError(
            "The number of handles must match the number of labels."
        )

    # Pair each artist with its corresponding label.
    return list(zip(handles, labels))

def make_legend(ax: Axes,
                handles: Sequence[Artist] | None = None,
                labels: Sequence[str] | None = None,
                title: str | None = None,
                groups: Sequence[
                    tuple[Sequence[Artist], Sequence[str], str | None]
                ] | None = None,
                frameon: bool = False,
                x_box=None,
                y_box=None,
                **kwargs,):
    """
    Create a customized legend for a Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to create the legend.

    handles : sequence of Artist, optional
        Legend handles for a standard single-section legend.

    labels : sequence of str, optional
        Labels corresponding to ``handles``.

    title : str, optional
        Title for a standard single-section legend.

    groups : sequence of tuple, optional
        Legend sections. Each entry must be a tuple of the form
        ``(handles, labels, title)``.

    frameon : bool, optional
        Whether to draw the legend frame. Defaults to False.

    x_box : float, optional
        Horizontal bounding for the box used to position the legend.

    y_box : float, optional
        Vertical bounding for the box used to position the legend.

    **kwargs
        Additional keyword arguments passed to ``ax.legend()`` for a
        standard legend.

    Returns
    -------
    matplotlib.legend.Legend or matplotlib.offsetbox.AnchoredOffsetbox
        The created legend.
    """

    if groups is not None:
        if handles is not None or labels is not None or title is not None:
            raise ValueError(
                "'groups' cannot be used together with 'handles', "
                "'labels', or 'title'."
            )

        return _make_grouped_legend(
            ax,
            groups,
            frameon=frameon,
            x_box=x_box,
            y_box=y_box,
            **kwargs,
        )
    if x_box is not None and y_box is not None:
        bbox_to_anchor=(x_box, y_box)
    elif x_box is not None and y_box is None:
        raise ValueError(
            f"'y_box' must be not None if 'x_box'={x_box}" 
        )
    elif x_box is None and y_box is not None:
        raise ValueError(
            f"'x_box' must be not None if 'y_box'={y_box}" 
        )
    else:
        bbox_to_anchor=None
    
    # Create a standard Matplotlib legend.
    leg = ax.legend(
        handles=handles,
        labels=labels,
        title=title,
        frameon=frameon,
        bbox_to_anchor=bbox_to_anchor,
        **kwargs,
    )

    # Align legend entries and title to the left.
    leg._legend_box.align = "left"

    if title is not None:
        leg.get_title().set_ha("left")

    return leg


def _make_grouped_legend(ax: Axes,
                         groups,
                         frameon=False,
                         x_box=None,
                         y_box=None,
                         **kwargs,):
    """
    Create a legend containing multiple titled sections.
    """

    rows = []

    for handles, labels, title in groups:
        section = []

        # Add the section title.
        if title is not None:
            section.append(
                TextArea(
                    title,
                    textprops={
                        "weight": "bold",
                    },
                )
            )

        leg = Legend(
            ax,
            handles,
            labels,
            frameon=False,
            ncols=1,
            **{k: v for k, v in kwargs.items() if k != "ncols"},
        )
        
        # Extract the legend's internal box.
        section.append(leg._legend_box)

        rows.append(
            VPacker(
                children=section,
                align="left",
                sep=4,
            )
        )

    # Arrange the sections horizontally.
    box = HPacker(
        children=rows,
        align="top",
        sep=16,
    )

    if x_box is	not None and y_box is not None:
        bbox_to_anchor=(x_box, y_box)
    elif x_box is not None and y_box is None:
        raise ValueError(
            f"'y_box' must be not None if 'x_box'={x_box}"
        )
    elif x_box is None and y_box is not None:
        raise ValueError(
            f"'x_box' must be not None if 'y_box'={y_box}"
        )
    else:
        bbox_to_anchor=None

    # Anchor the complete grouped legend to the Axes.
    anchored = AnchoredOffsetbox(
        loc=kwargs.get("loc", "best"),
        child=box,
        frameon=frameon,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=0.4,
    )

    ax.add_artist(anchored)

    return anchored

def save_figure(fig,
                fig_name,
                fig_dir,
                fig_ext="eps",
                transparent=True,
                close=True):
    """
    Save a figure in EPS or PNG format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.

    fig_name : str or pathlib.Path
        Name of the figure. The extension may be omitted or included.

    fig_dir : str or pathlib.Path
        Directory where the figure will be saved.

    fig_ext : {"eps", "png"}, default="eps"
        Output format.

    transparent : bool, default=True
        Whether to use a transparent figure and axes background.

    close : bool, default=True
        Whether to close the figure after saving.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """

    # ------------------------------------------------------------------
    # Validate extension
    # ------------------------------------------------------------------

    fig_ext = fig_ext.lower().lstrip(".")

    if fig_ext not in {"eps", "png"}:
        raise ValueError(
            "fig_ext must be either 'eps' or 'png'."
        )

    # ------------------------------------------------------------------
    # Prepare filename
    # ------------------------------------------------------------------

    fig_name = Path(fig_name)

    # Remove an existing extension, if present
    if fig_name.suffix.lower() in {".eps", ".png"}:
        fig_name = fig_name.with_suffix("")

    filename = fig_name.with_suffix(f".{fig_ext}")

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    filepath = fig_dir / filename

    # ------------------------------------------------------------------
    # Transparency
    # ------------------------------------------------------------------

    if transparent:
        fig.patch.set_alpha(0)

        for ax in fig.axes:
            ax.set_facecolor("none")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    fig.savefig(
        filepath,
        format=fig_ext,
        bbox_inches="tight",
    )

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    if close:
        plt.close(fig)

    return filepath
