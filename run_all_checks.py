"""
This script runs a bunch of checks and makes plots inside the plots/accuracy_checks/ directory.

These plots are then embedded in the accuracy_checks.md file in the root directory, which
has some information about how to interpret the plots.
"""

import argparse
import os

import logging
from hps.src.logging_utils import FMT, TIMEFMT
from hps.accuracy_checks.single_leaf_checks import (
    single_leaf_check_0,
    single_leaf_check_1,
    single_leaf_check_2,
    single_leaf_check_3,
    single_leaf_check_4,
    single_leaf_check_5,
    single_leaf_check_6,
    single_leaf_check_7,
    single_leaf_check_8,
    single_leaf_check_9,
    single_leaf_check_10,
    single_leaf_check_11,
    single_leaf_check_12,
)
from hps.accuracy_checks.single_merge_checks import (
    single_merge_check_0,
    single_merge_check_1,
    single_merge_check_2,
    single_merge_check_3,
    single_merge_check_4,
    single_merge_check_5,
    single_merge_check_6,
    single_merge_check_7,
    single_merge_check_8,
    single_merge_check_9,
    single_merge_check_10,
)
from hps.accuracy_checks.multi_merge_checks import (
    multi_merge_check_0,
    multi_merge_check_1,
    multi_merge_check_2,
    multi_merge_check_3,
    multi_merge_check_4,
    multi_merge_check_5,
    multi_merge_check_6,
    multi_merge_check_7,
)
from hps.accuracy_checks.single_leaf_checks_3D import (
    single_leaf_check_3D_0,
    single_leaf_check_3D_1,
    single_leaf_check_3D_2,
    single_leaf_check_3D_3,
    single_leaf_check_3D_4,
)
from hps.accuracy_checks.single_merge_checks_3D import (
    single_merge_check_3D_0,
    single_merge_check_3D_1,
    single_merge_check_3D_2,
    single_merge_check_3D_3,
    single_merge_check_3D_4,
)
from hps.accuracy_checks.nonuniform_grid_checks import (
    nonuniform_grid_check_0,
    nonuniform_grid_check_1,
    nonuniform_grid_check_2,
    nonuniform_grid_check_3,
)
from hps.accuracy_checks.nonuniform_grid_checks_3D import (
    nonuniform_grid_check_3D_0,
    nonuniform_grid_check_3D_1,
    nonuniform_grid_check_3D_2,
    nonuniform_grid_check_3D_3,
)

jax_logger = logging.getLogger("jax")
jax_logger.setLevel(logging.WARNING)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)

OUTPUT_DIR = "assets/accuracy_checks"


def main():
    # For development purposes. Sometimes I want to run one check at a time.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ################################################
    ## Single Leaf Checks

    # Polynomial Data
    # single_leaf_check_0(os.path.join(OUTPUT_DIR, "single_leaf_check_0.png"))
    # # # # # # # Non-Polynomial Data
    # single_leaf_check_1(os.path.join(OUTPUT_DIR, "single_leaf_check_1.png"))

    # # # # # # Nonzero Source Terms
    # single_leaf_check_2(os.path.join(OUTPUT_DIR, "single_leaf_check_2.png"))
    # single_leaf_check_3(os.path.join(OUTPUT_DIR, "single_leaf_check_3.png"))
    # single_leaf_check_4(os.path.join(OUTPUT_DIR, "single_leaf_check_4.png"))
    # single_leaf_check_5(os.path.join(OUTPUT_DIR, "single_leaf_check_5.png"))
    # single_leaf_check_6(os.path.join(OUTPUT_DIR, "single_leaf_check_6.png"))
    # single_leaf_check_7(os.path.join(OUTPUT_DIR, "single_leaf_check_7.png"))
    # single_leaf_check_8(os.path.join(OUTPUT_DIR, "single_leaf_check_8.png"))
    # single_leaf_check_9(os.path.join(OUTPUT_DIR, "single_leaf_check_9.png"))
    # single_leaf_check_10(os.path.join(OUTPUT_DIR, "single_leaf_check_10.png"))
    single_leaf_check_11(os.path.join(OUTPUT_DIR, "single_leaf_check_11.png"))
    single_leaf_check_12(os.path.join(OUTPUT_DIR, "single_leaf_check_12.png"))

    # single_leaf_check_3D_0(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_0.png"))
    # single_leaf_check_3D_1(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_1.png"))
    # single_leaf_check_3D_2(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_2.png"))
    # single_leaf_check_3D_3(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_3.png"))
    # single_leaf_check_3D_4(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_4.png"))

    # # # ################################################
    # # # ## Single Merge Checks

    # # # # Polynomial data
    # single_merge_check_0(os.path.join(OUTPUT_DIR, "single_merge_check_0.png"))
    # single_merge_check_1(os.path.join(OUTPUT_DIR, "single_merge_check_1.png"))
    # # # # # # Non-Polynomial data
    # single_merge_check_2(os.path.join(OUTPUT_DIR, "single_merge_check_2.png"))
    # single_merge_check_3(os.path.join(OUTPUT_DIR, "single_merge_check_3.png"))
    # single_merge_check_5(os.path.join(OUTPUT_DIR, "single_merge_check_5.png"))
    # single_merge_check_6(os.path.join(OUTPUT_DIR, "single_merge_check_6.png"))
    # single_merge_check_7(os.path.join(OUTPUT_DIR, "single_merge_check_7.png"))
    # single_merge_check_8(os.path.join(OUTPUT_DIR, "single_merge_check_8.png"))
    # single_merge_check_9(os.path.join(OUTPUT_DIR, "single_merge_check_9.png"))
    # single_merge_check_10(os.path.join(OUTPUT_DIR, "single_merge_check_10.png"))
    # single_merge_check_3D_0(os.path.join(OUTPUT_DIR, "single_merge_check_3D_0.png"))
    # single_merge_check_3D_1(os.path.join(OUTPUT_DIR, "single_merge_check_3D_1.png"))
    # single_merge_check_3D_2(os.path.join(OUTPUT_DIR, "single_merge_check_3D_2.png"))
    # single_merge_check_3D_3(os.path.join(OUTPUT_DIR, "single_merge_check_3D_3.png"))
    # single_merge_check_3D_4(os.path.join(OUTPUT_DIR, "single_merge_check_3D_4.png"))

    # # #  Nonzero Source test cases
    # # # # ################################################
    # # # Multi-Merge Checks
    # multi_merge_check_0(os.path.join(OUTPUT_DIR, "multi_merge_check_0.png"))
    # multi_merge_check_1(os.path.join(OUTPUT_DIR, "multi_merge_check_1.png"))
    # multi_merge_check_2(os.path.join(OUTPUT_DIR, "multi_merge_check_2.png"))
    # multi_merge_check_3(os.path.join(OUTPUT_DIR, "multi_merge_check_3.png"))
    # multi_merge_check_4(os.path.join(OUTPUT_DIR, "multi_merge_check_4.png"))
    # multi_merge_check_5(os.path.join(OUTPUT_DIR, "multi_merge_check_5.png"))
    # multi_merge_check_6(os.path.join(OUTPUT_DIR, "multi_merge_check_6.png"))
    # multi_merge_check_7(os.path.join(OUTPUT_DIR, "multi_merge_check_7.png"))

    # # ################################################
    # Nonuniform Grid Checks
    # nonuniform_grid_check_0(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_0.png"))
    # nonuniform_grid_check_1(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_1.png"))
    # nonuniform_grid_check_2(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_2.png"))
    # nonuniform_grid_check_3(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3.png"))
    # nonuniform_grid_check_3D_0(
    #     os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_0.png")
    # )
    # nonuniform_grid_check_3D_1(
    #     os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_1.png")
    # )
    # nonuniform_grid_check_3D_2(
    #     os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_2.png")
    # )
    # nonuniform_grid_check_3D_3(
    #     os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_3.png")
    # )


def main_all():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ################################################
    ## Single Leaf Checks

    # Polynomial Data
    single_leaf_check_0(os.path.join(OUTPUT_DIR, "single_leaf_check_0.png"))
    # # # # Non-Polynomial Data
    single_leaf_check_1(os.path.join(OUTPUT_DIR, "single_leaf_check_1.png"))
    # # # Nonzero Source Terms
    single_leaf_check_2(os.path.join(OUTPUT_DIR, "single_leaf_check_2.png"))
    single_leaf_check_3(os.path.join(OUTPUT_DIR, "single_leaf_check_3.png"))
    single_leaf_check_4(os.path.join(OUTPUT_DIR, "single_leaf_check_4.png"))
    single_leaf_check_5(os.path.join(OUTPUT_DIR, "single_leaf_check_5.png"))
    single_leaf_check_6(os.path.join(OUTPUT_DIR, "single_leaf_check_6.png"))
    single_leaf_check_7(os.path.join(OUTPUT_DIR, "single_leaf_check_7.png"))
    single_leaf_check_8(os.path.join(OUTPUT_DIR, "single_leaf_check_8.png"))
    single_leaf_check_9(os.path.join(OUTPUT_DIR, "single_leaf_check_9.png"))
    single_leaf_check_10(os.path.join(OUTPUT_DIR, "single_leaf_check_10.png"))
    single_leaf_check_11(os.path.join(OUTPUT_DIR, "single_leaf_check_11.png"))
    single_leaf_check_12(os.path.join(OUTPUT_DIR, "single_leaf_check_12.png"))

    single_leaf_check_3D_0(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_0.png"))
    single_leaf_check_3D_1(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_1.png"))
    single_leaf_check_3D_2(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_2.png"))
    single_leaf_check_3D_3(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_3.png"))
    single_leaf_check_3D_4(os.path.join(OUTPUT_DIR, "single_leaf_check_3D_4.png"))

    ##################################################
    # Single Leaf DtN Checks
    # single_dtn_check_0(os.path.join(OUTPUT_DIR, "single_dtn_check_0.png"))
    # single_dtn_check_1(os.path.join(OUTPUT_DIR, "single_dtn_check_1.png"))

    # ################################################
    # ## Single Merge Checks

    # # Polynomial data; EW merge
    single_merge_check_0(os.path.join(OUTPUT_DIR, "single_merge_check_0.png"))
    # # # # Polynomial data; NS merge
    # single_merge_check_1(os.path.join(OUTPUT_DIR, "single_merge_check_1.png"))
    # # # Non-Polynomial data; EW merge
    single_merge_check_2(os.path.join(OUTPUT_DIR, "single_merge_check_2.png"))
    # # # Non-Polynomial data; NS merge
    # single_merge_check_3(os.path.join(OUTPUT_DIR, "single_merge_check_3.png"))

    # Check DtN maps
    single_merge_check_4(os.path.join(OUTPUT_DIR, "single_merge_check_4.png"))
    # ItI version of the algorithm
    # single_merge_check_5(os.path.join(OUTPUT_DIR, "single_merge_check_5.png"))
    single_merge_check_6(os.path.join(OUTPUT_DIR, "single_merge_check_6.png"))
    single_merge_check_7(os.path.join(OUTPUT_DIR, "single_merge_check_7.png"))

    # Helmholtz problem with nonzero source
    # single_merge_check_8(os.path.join(OUTPUT_DIR, "single_merge_check_8.png"))
    # single_merge_check_9(os.path.join(OUTPUT_DIR, "single_merge_check_9.png"))
    single_merge_check_10(os.path.join(OUTPUT_DIR, "single_merge_check_10.png"))

    # single_merge_check_3D_0(os.path.join(OUTPUT_DIR, "single_merge_check_3D_0.png"))
    # single_merge_check_3D_1(os.path.join(OUTPUT_DIR, "single_merge_check_3D_1.png"))
    # single_merge_check_3D_2(os.path.join(OUTPUT_DIR, "single_merge_check_3D_2.png"))
    # single_merge_check_3D_3(os.path.join(OUTPUT_DIR, "single_merge_check_3D_3.png"))
    # single_merge_check_3D_4(os.path.join(OUTPUT_DIR, "single_merge_check_3D_4.png"))

    # ################################################
    # Multi-Merge Checks
    multi_merge_check_0(os.path.join(OUTPUT_DIR, "multi_merge_check_0.png"))
    multi_merge_check_1(os.path.join(OUTPUT_DIR, "multi_merge_check_1.png"))
    # multi_merge_check_2(os.path.join(OUTPUT_DIR, "multi_merge_check_2.png"))
    # multi_merge_check_3(os.path.join(OUTPUT_DIR, "multi_merge_check_3.png"))
    multi_merge_check_4(os.path.join(OUTPUT_DIR, "multi_merge_check_4.png"))
    # multi_merge_check_5(os.path.join(OUTPUT_DIR, "multi_merge_check_5.png"))
    multi_merge_check_6(os.path.join(OUTPUT_DIR, "multi_merge_check_6.png"))
    multi_merge_check_7(os.path.join(OUTPUT_DIR, "multi_merge_check_7.png"))

    # ################################################
    # Nonuniform Grid Checks
    nonuniform_grid_check_0(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_0.png"))
    nonuniform_grid_check_1(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_1.png"))
    nonuniform_grid_check_2(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_2.png"))
    nonuniform_grid_check_3(os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3.png"))
    nonuniform_grid_check_3D_0(
        os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_0.png")
    )
    nonuniform_grid_check_3D_1(
        os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_1.png")
    )
    nonuniform_grid_check_3D_2(
        os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_2.png")
    )
    nonuniform_grid_check_3D_3(
        os.path.join(OUTPUT_DIR, "nonuniform_grid_check_3D_3.png")
    )


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    a = parser.parse_args()
    return a


if __name__ == "__main__":
    args = setup_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=level)

    if args.a:
        main_all()
    else:
        main()
