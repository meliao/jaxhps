from ._domain import Domain
from ._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
    get_all_leaves,
)
from ._pdeproblem import PDEProblem
from ._build_solver import build_solver
from ._solve import solve
from ._subtree_recomp import (
    solve_subtree,
    upward_pass_subtree,
    downward_pass_subtree,
)
from ._device_config import (
    local_solve_chunksize_2D,
    local_solve_chunksize_3D,
)

# TODO: Remove these imports.
from ._utils import plot_soln_from_cheby_nodes
from ._grid_creation_2D import rearrange_indices_ext_int

# These will appear in the module's top-level namespace when imported
__all__ = [
    "Domain",
    "DiscretizationNode2D",
    "DiscretizationNode3D",
    "get_all_leaves",
    "PDEProblem",
    "build_solver",
    "solve",
    "solve_subtree",
    "local_solve_chunksize_2D",
    "local_solve_chunksize_3D",
    "upward_pass_subtree",
    "downward_pass_subtree",
    "plot_soln_from_cheby_nodes",
    "rearrange_indices_ext_int",
]
__version__ = "0.1"
