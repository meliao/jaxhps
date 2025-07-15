import logging

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List

from jaxhps import DiscretizationNode2D, DiscretizationNode3D

FIGSIZE = 5

FONTSIZE = 16

TICKSIZE = 15


def plot_func_with_grid(
    pts: jax.Array,
    samples: jax.Array,
    leaves: List[DiscretizationNode2D | DiscretizationNode3D],
    plot_fp: str,
) -> None:
    # Make a figure with 3 panels. First col is computed u, second col is expected u, third col is the absolute error
    fig, ax = plt.subplots(figsize=(5, 5))

    #############################################################
    # First column: Computed u

    extent = [
        pts[..., 0].min(),
        pts[..., 0].max(),
        pts[..., 1].min(),
        pts[..., 1].max(),
    ]

    im_0 = ax.imshow(samples, cmap="plasma", extent=extent)
    plt.colorbar(im_0, ax=ax)
    ax.set_xlabel("$x_1$", fontsize=FONTSIZE)
    ax.set_ylabel("$x_2$", fontsize=FONTSIZE)

    #############################################################
    # Find all nodes that intersect z=0 and plot them.

    for l in leaves:
        x = [l.xmin, l.xmax, l.xmax, l.xmin, l.xmin]
        y = [l.ymin, l.ymin, l.ymax, l.ymax, l.ymin]
        ax.plot(x, y, "-", color="gray", linewidth=1)
    plt.savefig(plot_fp, bbox_inches="tight")
    plt.clf()
    plt.close(fig)


def plot_field_for_wave_scattering_experiment(
    field: jnp.array,
    target_pts: jnp.array,
    use_bwr_cmap: bool = False,
    cmap_str: str = "plasma",
    title: str = None,
    save_fp: str = None,
    ax: plt.Axes = None,
    figsize: float = FIGSIZE,
    ticksize: float = TICKSIZE,
) -> None:
    """
    Expect field to have shape (n,n) and target_pts to have shape (n, n, 2).
    """
    bool_create_ax = ax is None

    if bool_create_ax:
        fig, ax = plt.subplots(figsize=(figsize, figsize))

    extent = [
        target_pts[0, 0, 0],
        target_pts[-1, -1, 0],
        target_pts[0, 0, 1],
        target_pts[-1, -1, 1],
    ]
    logging.debug(
        "plot_field_for_wave_scattering_experiment: max_val: %s",
        jnp.max(field),
    )
    logging.debug(
        "plot_field_for_wave_scattering_experiment: min_val: %s",
        jnp.min(field),
    )

    if use_bwr_cmap:
        max_val = 3.65  # Max val of the fields we plot in the paper

        im = ax.imshow(
            field,
            cmap="bwr",
            vmin=-max_val,
            vmax=max_val,
            extent=extent,
        )
    elif cmap_str == "hot":
        im = ax.imshow(
            field,
            cmap=cmap_str,
            extent=extent,
            vmin=0.0,
            vmax=jnp.max(jnp.abs(field)),
        )
    else:
        im = ax.imshow(
            field,
            cmap=cmap_str,
            extent=extent,
            # vmin=-3.65,  # Min val of the fields we plot in the paper
            # vmax=3.65,
        )

    # Set ticks to [-1, 0, 1]
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="both", which="major", labelsize=ticksize)

    # Sizing brought to you by https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    make_scaled_colorbar(im, ax, fontsize=ticksize)

    if title is not None:
        ax.set_title(title)

    if bool_create_ax:
        if save_fp is not None:
            fig.tight_layout()
            plt.savefig(save_fp, bbox_inches="tight")
        else:
            plt.show()


CMAP_PAD = 0.1


def make_scaled_colorbar(im, ax, fontsize: float = None) -> None:
    """
    Make a colorbar that is the same size as the plot.

    Found this on StackExchange

    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=CMAP_PAD)
    cbar = plt.colorbar(im, cax=cax)
    if fontsize is not None:
        cbar.ax.tick_params(labelsize=fontsize)


def plot_field_for_multi_source_wave_scattering(
    field: jnp.array,
    target_pts: jnp.array,
    use_bwr_cmap: bool = False,
    cmap_str: str = "parula",
    title: str = None,
    save_fp: str = None,
    ax: plt.Axes = None,
    minval: float = -3.65,
    maxval: float = 3.65,
    figsize: float = 5,
    fontsize: float = 16,
    ticksize: float = 15,
    dpi: int = None,
) -> None:
    """
    Expect field to have shape (n,n) and target_pts to have shape (n, n, 2).
    """
    bool_create_ax = ax is None

    if bool_create_ax:
        fig, ax = plt.subplots(figsize=(figsize, figsize))

    extent = [
        target_pts[0, 0, 0],
        target_pts[-1, -1, 0],
        target_pts[-1, -1, 1],
        target_pts[0, 0, 1],
    ]
    # logging.debug(
    #     "plot_field_for_wave_scattering_experiment: max_val: %s", jnp.max(field)
    # )
    # logging.debug(
    #     "plot_field_for_wave_scattering_experiment: min_val: %s", jnp.min(field)
    # )

    if use_bwr_cmap:
        max_val = 3.65  # Max val of the fields we plot in the paper

        im = ax.imshow(
            field,
            cmap="bwr",
            vmin=-max_val,
            vmax=max_val,
            extent=extent,
        )
    else:
        im = ax.imshow(
            field,
            cmap=cmap_str,
            extent=extent,
            vmin=minval,  # Min val of the fields we plot in the paper
            vmax=maxval,
        )

    # Set ticks to [-1, 0, 1]
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="both", which="major", labelsize=ticksize)

    # Sizing brought to you by https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    make_scaled_colorbar(im, ax, fontsize=ticksize)

    if title is not None:
        ax.set_title(title)

    if bool_create_ax:
        if save_fp is not None:
            if save_fp.endswith(".png"):
                plt.savefig(
                    save_fp,
                    dpi=dpi,
                )
            else:
                plt.savefig(save_fp, bbox_inches="tight")

        plt.close()
