import logging


import numpy as np
from matplotlib import cm, colors
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


def get_discrete_cmap(N: int, cmap: str) -> cm.ScalarMappable:
    """
    Create an N-bin discrete colormap from the specified input map
    """
    cmap = plt.get_cmap(cmap)

    # If it's plasma, go 0 to 0.8
    if cmap.name == "plasma" or cmap.name == "parula":
        return cmap(np.linspace(0, 0.8, N))
    else:
        return cmap(np.linspace(0, 1, N))


# https://stackoverflow.com/questions/34859628/has-someone-made-the-parula-colormap-in-matplotlib
cm_data = [
    [0.2081, 0.1663, 0.5292],
    [0.2116238095, 0.1897809524, 0.5776761905],
    [0.212252381, 0.2137714286, 0.6269714286],
    [0.2081, 0.2386, 0.6770857143],
    [0.1959047619, 0.2644571429, 0.7279],
    [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286],
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571],
    [0.0165142857, 0.4266, 0.8786333333],
    [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429],
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467],
    [0.0779428571, 0.5039857143, 0.8383714286],
    [0.079347619, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429],
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381],
    [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667],
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381],
    [0.0589714286, 0.6837571429, 0.7253857143],
    [0.0843, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015, 0.6858571429],
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048],
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667],
    [0.3952571429, 0.7459, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905],
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.609852381, 0.7473142857, 0.4336857143],
    [0.6473, 0.7456, 0.4188],
    [0.6834190476, 0.7434761905, 0.4044333333],
    [0.7184095238, 0.7411333333, 0.3904761905],
    [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762],
    [0.8506571429, 0.7299, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217],
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619],
    [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.196652381],
    [0.988, 0.8066, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857],
    [0.9697, 0.8481380952, 0.147452381],
    [0.9625857143, 0.8705142857, 0.1309],
    [0.9588714286, 0.8949, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333],
    [0.9763, 0.9831, 0.0538],
]

parula_cmap = colors.LinearSegmentedColormap.from_list("parula", cm_data)
