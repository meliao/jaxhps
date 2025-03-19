from typing import Dict, Any, List, Tuple
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import hashlib
import numpy as np
import os
import argparse
import logging
import sys
from timeit import default_timer
from scipy.io import loadmat, savemat

from hps.src.solver_obj import SolverObj, create_solver_obj_2D, create_solver_obj_3D
from hps.src.up_down_passes import (
    down_pass,
    local_solve_stage,
    build_stage,
    fused_pde_solve_2D,
    fused_pde_solve_2D_ItI,
    baseline_pde_solve_2D,
)
from hps.accuracy_checks.dirichlet_neumann_data import (
    nonpoly_dirichlet_data_0,
    coeff_fn_dxx_polynomial_dirichlet_data_1,
    coeff_fn_dyy_polynomial_dirichlet_data_1,
    source_polynomial_dirichlet_data_1,
)
from hps.src.logging_utils import FMT, TIMEFMT
from hps.src import config
from hps.src.quadrature.trees import Node


def hash_dict(dictionary: Dict[str, Any]) -> str:
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()


def setup_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument("-n_samples", type=int)
    parser.add_argument("-l", type=int)
    parser.add_argument("-p", type=int)
    parser.add_argument("-out_fp", type=str)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "-t", "--three_d", default=False, action="store_true", help="Time 3D problems"
    )
    parser.add_argument(
        "--fused",
        default=False,
        action="store_true",
        help="Use fused code for fewer data transfers.",
    )
    parser.add_argument(
        "--recomputation",
        default="None",
        help="Specify which recomputation strategy to use.",
        choices=["None", "Baseline", "Ours"],
    )
    parser.add_argument(
        "--ItI", "-i", default=False, action="store_true", help="Use ItI solver."
    )
    a = parser.parse_args()
    a.hash = hash_dict(vars(a))
    return a


def pde_solve(
    t: SolverObj,
    blocking: bool = False,
    fused_recomp: bool = False,
    baseline_recomp: bool = False,
    use_ItI: bool = False,
) -> Tuple[float, float, float, float]:
    """Returns solve time, merge time, down_pass time, and total time."""
    # gpu_memory_usage()
    # Setup Tree and Precompute Operators
    t0 = default_timer()

    t.reset()

    # gpu_memory_usage()
    if blocking:
        t.Q_D.block_until_ready()
    t_setup = default_timer() - t0

    ## Local Solve Stage
    t_1 = default_timer()
    # Doing non-constant coefficients
    coeffs_dxx = coeff_fn_dxx_polynomial_dirichlet_data_1(t.leaf_cheby_points)
    coeffs_dyy = coeff_fn_dyy_polynomial_dirichlet_data_1(t.leaf_cheby_points)
    source = source_polynomial_dirichlet_data_1(t.leaf_cheby_points)
    dirichlet_data = nonpoly_dirichlet_data_0(t.root_boundary_points)

    if fused_recomp:
        if use_ItI:
            fused_pde_solve_2D_ItI(
                t,
                source_term=source,
                D_xx_coeffs=coeffs_dxx,
                D_yy_coeffs=coeffs_dyy,
                boundary_data=dirichlet_data,
            )
        else:
            fused_pde_solve_2D(
                t,
                source_term=source,
                D_xx_coeffs=coeffs_dxx,
                D_yy_coeffs=coeffs_dyy,
                boundary_data=dirichlet_data,
            )
        x = t.interior_solns.block_until_ready()

        t_all = default_timer() - t0
        t_solve = np.nan
        t_merge = np.nan
        t_down = np.nan
    elif baseline_recomp:
        if use_ItI:
            raise ValueError("Baseline recomputation not implemented for ItI")
        else:

            baseline_pde_solve_2D(
                t,
                source_term=source,
                D_xx_coeffs=coeffs_dxx,
                D_yy_coeffs=coeffs_dyy,
                boundary_data=dirichlet_data,
            )
        x = t.interior_solns.block_until_ready()

        t_all = default_timer() - t0
        t_solve = np.nan
        t_merge = np.nan
        t_down = np.nan

    else:
        local_solve_stage(
            t, source_term=source, D_xx_coeffs=coeffs_dxx, D_yy_coeffs=coeffs_dyy
        )
        if blocking:
            t.leaf_node_v_vecs.block_until_ready()
        t_solve = default_timer() - t_1
        # gpu_memory_usage()
        ## Build Stage
        t_2 = default_timer()
        build_stage(t)
        if blocking:
            t.interior_node_S_maps[-1].block_until_ready()
        t_merge = default_timer() - t_2
        # gpu_memory_usage()
        ## Down Pass
        t_3 = default_timer()
        n_per_side = dirichlet_data.shape[0] // 4
        dirichlet_data_lst = [  # Splitting the data into 4 quadrants
            dirichlet_data[:n_per_side],
            dirichlet_data[n_per_side : 2 * n_per_side],
            dirichlet_data[2 * n_per_side : 3 * n_per_side],
            dirichlet_data[3 * n_per_side :],
        ]
        down_pass(
            t,
            dirichlet_data_lst,
            # D_xx_coeffs=coeffs_dxx,
            # D_yy_coeffs=coeffs_dyy,
        )
        x = t.interior_solns.block_until_ready()
        t_down = default_timer() - t_3
        t_all = default_timer() - t0

    return t_setup, t_solve, t_merge, t_down, t_all


def main(args: argparse.Namespace) -> None:
    """This function intitializes N (p,p) matrices, and then
    times the following script:
     - moving data to GPU
     - invering the matrices
     - returning data to the CPU
    """

    logging.info("Beginning main function.")
    logging.info("JAX default device is %s", jax.config.jax_default_device)
    logging.info("JAX list of devices is %s", jax.devices())
    logging.info("Host device is %s", config.HOST_DEVICE)
    north = jnp.pi / 2
    south = -jnp.pi / 2
    east = jnp.pi / 2
    west = -jnp.pi / 2
    if args.three_d:
        logging.info("Timing 3D problmes.")
        corners = jnp.array([[west, south, south], [east, north, north]])
    else:
        logging.info("Timing 2D problems.")
        corners = [(west, south), (east, south), (east, north), (west, north)]

    bool_fused = args.recomputation == "Ours"
    bool_baseline = args.recomputation == "Baseline"
    logging.info("bool_fused=%s, and bool_baseline=%s", bool_fused, bool_baseline)

    if args.three_d:
        t = create_solver_obj_3D(p=args.p, q=args.p - 2, l=args.l, corners=corners)
    else:
        xmin, ymin = corners[0]
        xmax, ymax = corners[2]
        root = Node(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        t = create_solver_obj_2D(
            p=args.p,
            q=args.p - 2,
            root=root,
            uniform_levels=args.l,
            use_ItI=args.ItI,
            eta=4.0,
            fill_tree=False,
        )

    logging.info("Running one problem instance to JIT-compile the code")
    pde_solve(
        t=t,
        blocking=False,
        fused_recomp=bool_fused,
        baseline_recomp=bool_baseline,
        use_ItI=args.ItI,
    )
    pde_solve(
        t=t,
        blocking=True,
        fused_recomp=bool_fused,
        baseline_recomp=bool_baseline,
        use_ItI=args.ItI,
    )

    logging.info(
        "Beginning to take %i time samples for full PDE solve. Parameters are p=%i; l=%i",
        args.n_samples,
        args.p,
        args.l,
    )

    time_samples_total = np.full((args.n_samples,), fill_value=np.nan, dtype=np.float64)
    for i in range(args.n_samples):
        logging.debug("Getting sample %i/%i", i + 1, args.n_samples)
        t_setup, t_solve, t_merge, t_down, t_total = pde_solve(
            t=t,
            blocking=False,
            fused_recomp=bool_fused,
            baseline_recomp=bool_baseline,
            use_ItI=args.ItI,
        )
        time_samples_total[i] = t_total

    logging.info(
        "Done taking samples for full PDE solves. Now taking samples in a blocking fashion."
    )

    time_samples_setup = np.full((args.n_samples,), fill_value=np.nan, dtype=np.float64)
    time_samples_solve = np.full((args.n_samples,), fill_value=np.nan, dtype=np.float64)
    time_samples_merge = np.full((args.n_samples,), fill_value=np.nan, dtype=np.float64)
    time_samples_down = np.full((args.n_samples,), fill_value=np.nan, dtype=np.float64)
    time_samples_total_blocking = np.full(
        (args.n_samples,), fill_value=np.nan, dtype=np.float64
    )
    if not (bool_fused or bool_baseline):
        for i in range(args.n_samples):
            logging.debug("Getting sample %i/%i", i + 1, args.n_samples)
            t_setup, t_solve, t_merge, t_down, t_total = pde_solve(
                t=t,
                blocking=True,
                fused_recomp=bool_fused,
                baseline_recomp=bool_baseline,
                use_ItI=args.ItI,
            )
            time_samples_setup[i] = t_setup
            time_samples_solve[i] = t_solve
            time_samples_merge[i] = t_merge
            time_samples_down[i] = t_down
            time_samples_total_blocking[i] = t_total

        logging.info("Done taking samples.")
    logging.info("Calculating statistics.")

    mean_setup = np.mean(time_samples_setup)
    stddev_setup = np.std(time_samples_setup)
    logging.info("Setup stage results: %f sec +/- %f", mean_setup, stddev_setup)
    mean_solve = np.mean(time_samples_solve)
    stddev_solve = np.std(time_samples_solve)
    logging.info("Solve stage results: %f sec +/- %f", mean_solve, stddev_solve)
    mean_merge = np.mean(time_samples_merge)
    stddev_merge = np.std(time_samples_merge)
    logging.info("Merge stage results: %f sec +/- %f", mean_merge, stddev_merge)
    mean_down = np.mean(time_samples_down)
    stddev_down = np.std(time_samples_down)
    logging.info("Down pass results: %f sec +/- %f", mean_down, stddev_down)
    mean_total = np.mean(time_samples_total)
    stddev_total = np.std(time_samples_total)
    logging.info("Total time results: %f sec +/- %f", mean_total, stddev_total)
    mean_total_blocking = np.mean(time_samples_total_blocking)
    stddev_total_blocking = np.std(time_samples_total_blocking)
    logging.info(
        "Total time results (blocking): %f sec +/- %f",
        mean_total_blocking,
        stddev_total_blocking,
    )

    logging.info("Saving results to %s", args.out_fp)
    savemat(
        args.out_fp,
        {
            "setup_times": time_samples_setup,
            "solve_times": time_samples_solve,
            "merge_times": time_samples_merge,
            "down_pass_times": time_samples_down,
            "times": time_samples_total,
            "times_blocking": time_samples_total_blocking,
            "default_device": str(jax.config.jax_default_device),
            "devices": [str(x) for x in jax.devices()],
        },
    )
    logging.info("Finished.")


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)

    # with jax.default_device(CPU_DEVICE):
    #     main(args)
    main(args)
