from typing import Callable
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jnp
from matplotlib import cm, colors
from scipy.interpolate import LinearNDInterpolator

from hps.src.utils import meshgrid_to_lst_of_pts
from hps.src.quadrature.trees import Node, get_all_leaves, get_all_leaves_jitted

# plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Roman"})

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_func_with_grid(
    eval_pts: jnp.array,
    corners: jnp.array,
    plt_xlabel: str,
    plt_ylabel: str,
    plot_fp: str,
    f: Callable[[jnp.array], jnp.array],
    bwr_cmap: bool = False,
) -> None:
    """
    Plot the function with the adaptive grid overlaid.

    Expects eval_pts to be shape (n, n, 3)

    Expects corners to be shape (x, 2, 2). The first row is xmin, ymin and the second row is xmax, ymax.
    """
    LABELSIZE = 20
    LINEWIDTH = 0.5

    # evaluate perm on the eval_pts
    f_vals = f(eval_pts)

    # Plot the function
    fig, ax = plt.subplots(figsize=(5, 5))

    if corners is None:
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
        extent = [xmin, xmax, ymin, ymax]
    else:
        xmin = corners[:, 0, 0].min()
        xmax = corners[:, 1, 0].max()
        ymin = corners[:, 0, 1].min()
        ymax = corners[:, 1, 1].max()
        extent = [xmin, xmax, ymin, ymax]

    if bwr_cmap:
        min_val = f_vals.min()
        max_val = f_vals.max()
        abs_max = max(abs(min_val), abs(max_val))
        norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im = ax.imshow(f_vals, extent=extent, origin="lower", cmap="bwr", norm=norm)
        line_color = "black"
    else:
        im = ax.imshow(f_vals, extent=extent, origin="lower", cmap=parula_cmap)
        line_color = "black"
    fig.colorbar(im, ax=ax)

    # Plot the grid
    if corners is not None:
        for c in corners:
            xmin = c[0, 0]
            xmax = c[1, 0]
            ymin = c[0, 1]
            ymax = c[1, 1]
            x = [xmin, xmax, xmax, xmin, xmin]
            y = [ymin, ymin, ymax, ymax, ymin]
            ax.plot(x, y, "-", color=line_color, linewidth=LINEWIDTH)

    # Set the labels
    ax.set_xlabel(plt_xlabel, fontsize=LABELSIZE)
    ax.set_ylabel(plt_ylabel, fontsize=LABELSIZE)

    # Save the figure
    plt.savefig(plot_fp, bbox_inches="tight")
    plt.close(fig)


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


def get_discrete_cmap(N: int, cmap: str) -> cm.ScalarMappable:
    """
    Create an N-bin discrete colormap from the specified input map
    """
    if cmap == "parula":
        cmap = parula_cmap
    else:
        cmap = plt.get_cmap(cmap)

    # If it's plasma, go 0 to 0.8
    if cmap.name == "plasma" or cmap.name == "parula":
        return cmap(np.linspace(0, 0.8, N))
    else:

        return cmap(np.linspace(0, 1, N))


def autodiff_notebook_plot_1(
    cheby_nodes: jnp.ndarray,
    corners: jnp.ndarray,
    soln_1: jnp.ndarray,
    soln_2: jnp.ndarray,
    point: jnp.ndarray,
    t_1: str = None,
    t_2: str = None,
    supt: str = None,
) -> None:
    """
    Plot the PDE solution on the grid specified by cheby_nodes and corners.
    There are 2 solutions specified, soln_1 and soln_2, which correspond to different PDE coefficients.

    point specifies the (x,y) coords at which we are computing the partial derivative. We'll mark it with an x.

    """
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    # Get a list of regularly-spaced points in the domain
    n_X = 100
    x = jnp.linspace(corners[0][0], corners[1][0], n_X)
    y = jnp.linspace(corners[0][1], corners[2][1], n_X)
    X, Y = jnp.meshgrid(x, jnp.flipud(y))
    lst_of_pts = meshgrid_to_lst_of_pts(X, Y)

    # Create an interpolator for soln_1
    interp_1 = LinearNDInterpolator(cheby_nodes, soln_1)
    soln_1_reggrid = interp_1(lst_of_pts).reshape(n_X, n_X)

    # Create an interpolator for soln_2
    interp_2 = LinearNDInterpolator(cheby_nodes, soln_2)
    soln_2_reggrid = interp_2(lst_of_pts).reshape(n_X, n_X)

    # Get all solution values to set color limits
    all_soln_vals = np.concatenate([soln_1_reggrid, soln_2_reggrid])

    # Create a colormap
    cmap = cm.plasma
    norm = colors.Normalize(vmin=all_soln_vals.min(), vmax=all_soln_vals.max())
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # Loop over the list of leaf nodes, plotting the interior + boundary points in the color from the colormap.
    # for j, pt in enumerate(cheby_nodes):
    #     color_0 = cmap(norm(expected_soln[j]))
    #     ax[0].plot(pt[0], pt[1], "o", color=color_0)

    #     color_1 = cmap(norm(computed_soln[j]))
    #     ax[1].plot(pt[0], pt[1], "o", color=color_1)

    # Plot the expected and computed solutions
    im_0 = ax[0].imshow(
        soln_1_reggrid,
        cmap=cmap,
        norm=norm,
        extent=[corners[0][0], corners[1][0], corners[0][1], corners[2][1]],
    )
    im_1 = ax[1].imshow(
        soln_2_reggrid,
        cmap=cmap,
        norm=norm,
        extent=[corners[0][0], corners[1][0], corners[0][1], corners[2][1]],
    )

    # Mark the point
    ax[0].plot(point[0], point[1], "x", color="black")
    ax[1].plot(point[0], point[1], "x", color="black")

    if t_1 is not None:
        ax[0].set_title(t_1)
    if t_2 is not None:
        ax[1].set_title(t_2)
    if supt is not None:
        fig.suptitle(supt)

    # ax.legend()
    plt.colorbar(im_0, ax=ax[0])
    plt.colorbar(im_1, ax=ax[1])
    plt.show()


def plot_2D_adaptive_refinement(
    f_fn: Callable[[jnp.array], jnp.array],
    root: Node,
    p: int,
    fp: str = None,
    title: str = None,
) -> None:
    """Plot the 2D refinement.

    <corners_adaptive> specifies a list of leaf corners.
    For each leaf, draw a box with the corners specified by the leaf corners.

    Then, sample the function f_fn at a regular grid of points and plot the values.


    Args:
        corners (jnp.array): Has shape (n_leaves, 4, 2) specifying the corners of each of the leaves.
        f_fn (Callable[[jnp.array], jnp.array]): _description_
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    # Plot the function
    north_outer = root.ymax
    south_outer = root.ymin
    east_outer = root.xmax
    west_outer = root.xmin

    n = 100
    x = jnp.linspace(west_outer, east_outer, n)
    y = jnp.linspace(south_outer, north_outer, n)
    X, Y = jnp.meshgrid(x, jnp.flipud(y))
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    Z = f_fn(pts).reshape(n, n)

    # Plot the first axis, which is the adaptive mesh without
    # level restriction
    im_0 = ax.imshow(
        Z,
        extent=[west_outer, east_outer, south_outer, north_outer],
        cmap="plasma",
        alpha=0.7,
    )

    for leaf in get_all_leaves(root):
        w = leaf.xmin
        e = leaf.xmax
        s = leaf.ymin
        n = leaf.ymax
        ax.hlines(s, w, e, color="black")
        ax.hlines(n, w, e, color="black")
        ax.vlines(w, s, n, color="black")
        ax.vlines(e, s, n, color="black")

    n_leaves = len(get_all_leaves_jitted(root))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.colorbar(im_0, ax=ax)

    if fp is not None:
        plt.savefig(fp, bbox_inches="tight")
    else:
        plt.show()
    plt.clf()


def plot_adaptive_grid_histogram(root: Node, plot_fp: str, tol: float, p: int) -> None:
    """
    Given a tree with adaptive refinement, plot a histogram of the reciprocal of the leaves' side lengths.
    """
    side_lens = jnp.array(
        [leaf.xmax - leaf.xmin for leaf in get_all_leaves_jitted(root)]
    )
    one_over_side_lens = 1.0 / side_lens
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    TITLESIZE = 20

    ax.hist(one_over_side_lens, bins=20)

    ax.set_title(
        f"Domain has side length = {root.xmax - root.xmin}.\n $tol={tol}$ $p={p}$",
        fontsize=TITLESIZE,
    )
    ax.set_xlabel("1 / Side Length", fontsize=TITLESIZE)
    ax.set_ylabel("Frequency", fontsize=TITLESIZE)
    ax.grid()
    fig.tight_layout()
    plt.savefig(plot_fp, bbox_inches="tight")
    plt.close(fig)


def plot_field_for_wave_scattering_experiment(
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
        if cmap_str == "parula":
            cmap_str = parula_cmap
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


CMAP_PAD = 0.1


def make_scaled_colorbar(im, ax, fontsize: float = None) -> None:
    """
    Make a colorbar that is the same size as the plot.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=CMAP_PAD)
    cbar = plt.colorbar(im, cax=cax)
    if fontsize is not None:
        cbar.ax.tick_params(labelsize=fontsize)
