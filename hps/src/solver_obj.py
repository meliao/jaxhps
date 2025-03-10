from typing import Tuple, List, Dict, Callable
import logging
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import LinearNDInterpolator


from hps.src.utils import meshgrid_to_lst_of_pts
from hps.src.quadrature.trees import (
    Node,
    add_uniform_levels,
    get_all_leaves,
    get_all_leaves_jitted,
    get_nodes_at_level,
)

from hps.src.quadrature.quad_2D.grid_creation import (
    get_all_boundary_gauss_legendre_points,
    get_all_leaf_2d_cheby_points,
    get_all_leaf_2d_cheby_points_uniform_refinement,
    get_all_boundary_gauss_legendre_points_uniform_refinement,
)
from hps.src.quadrature.quad_2D.differentiation import (
    precompute_diff_operators as precompute_diff_operators_2d,
    precompute_F_matrix as precompute_F_matrix_2d,
    precompute_G_matrix as precompute_G_matrix_2d,
    precompute_N_matrix as precompute_N_matrix_2d,
    precompute_N_tilde_matrix as precompute_N_tilde_matrix_2d,
)
from hps.src.quadrature.quad_2D.interpolation import (
    precompute_P_matrix as precompute_P_matrix_2d,
    precompute_Q_D_matrix as precompute_Q_D_matrix_2d,
    precompute_I_P_0_matrix as precompute_I_P_0_matrix_2d,
    precompute_Q_I_matrix as precompute_Q_I_matrix_2d,
    precompute_refining_coarsening_ops as precompute_refining_coarsening_ops_2D,
)

from hps.src.quadrature.quad_3D.grid_creation import (
    get_all_boundary_gauss_legendre_points as get_all_boundary_gauss_legendre_points_3d,
    get_all_leaf_3d_cheby_points,
)
from hps.src.quadrature.quad_3D.differentiation import (
    precompute_diff_operators as precompute_diff_operators_3d,
)
from hps.src.quadrature.quad_3D.interpolation import (
    precompute_P_matrix as precompute_P_matrix_3d,
    precompute_Q_D_matrix as precompute_Q_D_matrix_3d,
    precompute_refining_coarsening_ops as precompute_refining_coarsening_ops_3d,
)


class SolverObj:

    def __init__(
        self,
    ) -> None:

        self.root: Node = None
        self.p: int = None
        self.q: int = None
        self.l: int = None
        self.leaf_cheby_points: jnp.array = None
        self.root_boundary_points: jnp.array = None
        self.sidelens: jnp.array = None
        self.D_x: jnp.array = None
        self.D_y: jnp.array = None
        self.D_z: jnp.array = None
        self.D_xx: jnp.array = None
        self.D_yy: jnp.array = None
        self.D_xy: jnp.array = None
        self.D_xz: jnp.array = None
        self.D_yz: jnp.array = None
        self.D_zz: jnp.array = None
        self.P: jnp.array = None
        self.Q_D: jnp.array = None
        self.refinement_op: jnp.array = None
        self.coarsening_op: jnp.array = None

        # Pre-computed operators for ItI case
        self.I_P_0: jnp.array = None
        self.Q_I: jnp.array = None
        self.F: jnp.array = None
        self.G: jnp.array = None
        self.eta: float = None

        # These arrays are filled in during the upward and
        # downward passes of the HPS algorithm.
        # self.leaf_diff_operators: jnp.ndarray = None
        # self.leaf_DtN_maps: jnp.ndarray = None
        self.leaf_Y_maps: jnp.ndarray = None
        self.interior_node_DtN_maps: List[jnp.ndarray] = []
        # Initialize interior_node_S_maps with a dummy array because element i
        # of the list maps the Dirichlet data on the boundary of i-th level nodes
        # to the interior of the i-th level nodes.
        self.interior_node_S_maps: List[jnp.ndarray] = [
            None,
        ]
        self.leaf_node_v_vecs: jnp.ndarray = None
        self.leaf_node_v_prime_vecs: jnp.ndarray = None
        self.interior_node_v_int_vecs: List[jnp.ndarray] = []

        self.root_DtN: jnp.ndarray = None
        self.root_S: jnp.ndarray = None
        self.boundary_data: List[jnp.ndarray] = None
        self.interior_solns = jnp.ndarray = None

        # These are filled during the upward and downward passes of ItI versions of the algo
        self.interior_node_R_maps: List[jnp.array] = []
        self.interior_node_h_vecs: List[jnp.array] = []

        # Whether to use uniform or adaptive merge stage
        self.uniform_grid: bool = None

    def reset(self) -> None:
        """
        Resets the solver to remove the solution data and precomputed solution operators.
        The root and precomputed differentiation, interpolation operators are all kept.
        """
        # These arrays are filled in during the upward and
        # downward passes of the HPS algorithm.
        # self.leaf_diff_operators: jnp.ndarray = None
        # self.leaf_DtN_maps: jnp.ndarray = None
        self.leaf_Y_maps: jnp.ndarray = None
        self.interior_node_DtN_maps: List[jnp.ndarray] = []
        # Initialize interior_node_S_maps with a dummy array because element i
        # of the list maps the Dirichlet data on the boundary of i-th level nodes
        # to the interior of the i-th level nodes.
        self.interior_node_S_maps: List[jnp.ndarray] = [
            None,
        ]
        self.leaf_node_v_vecs: jnp.ndarray = None
        self.leaf_node_v_prime_vecs: jnp.ndarray = None
        self.interior_node_v_int_vecs: List[jnp.ndarray] = []

        self.root_DtN: jnp.ndarray = None
        self.root_S: jnp.ndarray = None
        self.boundary_data: List[jnp.ndarray] = None
        self.interior_solns = jnp.ndarray = None

        # These are filled during the upward and downward passes of ItI versions of the algo
        self.interior_node_R_maps: List[jnp.array] = []
        self.interior_node_h_vecs: List[jnp.array] = []


def create_solver_obj_2D(
    p: int,
    q: int,
    root: Node,
    uniform_levels: int = None,
    use_ItI: bool = False,
    eta: float = None,
    fill_tree: bool = True,
) -> SolverObj:
    """Assumes root is already built, unless uniform_levels is specified."""
    if use_ItI and eta is None:
        raise ValueError("Must specify eta to use ItI version.")

    t = SolverObj()
    all_leaves = get_all_leaves(root)
    if len(all_leaves) == 1 and uniform_levels is not None:
        t.uniform_grid = True
        t.l = uniform_levels
        if fill_tree:
            add_uniform_levels(root=root, l=uniform_levels)

    elif len(all_leaves) == 1 and uniform_levels is None:
        # Edge case for no refinement
        t.uniform_grid = True
        t.l = 0
    else:
        t.uniform_grid = False
        t.l = None

    t.root = root
    t.p = p
    t.q = q

    # Only initialize the Cheby points on the leaf nodes and the Gauss points on the boundary of the root node.
    if fill_tree:
        leaf_cheby_points = get_all_leaf_2d_cheby_points(p, t.root)
        root_gauss_points = get_all_boundary_gauss_legendre_points(q, t.root)
    else:
        corners = np.array(
            [
                [root.xmin, root.ymin],
                [root.xmax, root.ymin],
                [root.xmax, root.ymax],
                [root.xmin, root.ymax],
            ]
        )
        leaf_cheby_points = get_all_leaf_2d_cheby_points_uniform_refinement(
            p=p, l=uniform_levels, corners=corners
        )
        root_gauss_points = get_all_boundary_gauss_legendre_points_uniform_refinement(
            q=q, l=uniform_levels, corners=corners
        )

    t.leaf_cheby_points = leaf_cheby_points
    t.root_boundary_points = root_gauss_points
    t.sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])

    # Compute the differentiation operators.
    if use_ItI or t.uniform_grid:
        half_side_len = (root.xmax - root.xmin) / (2 ** (t.l + 1))
    else:
        # For DtN merges we rescale the differentiation operators inside the
        # local solve stage routine, because non-uniform grids may be used.
        half_side_len = 1.0
    du_dx, du_dy, du_dxx, du_dyy, du_dxy = precompute_diff_operators_2d(
        p, half_side_len=half_side_len
    )
    # P = precompute_P_matrix_2d(p, q)
    # Q_D = precompute_Q_D_matrix_2d(p, q, du_dx, du_dy)
    refinement, coarse = precompute_refining_coarsening_ops_2D(q)

    t.D_x = du_dx
    t.D_y = du_dy
    t.D_xx = du_dxx
    t.D_yy = du_dyy
    t.D_xy = du_dxy
    # t.P = P
    # t.Q_D = Q_D
    t.refinement_op = refinement
    t.coarsening_op = coarse

    if use_ItI:
        I_P_0 = precompute_I_P_0_matrix_2d(p, q)
        Q_I = precompute_Q_I_matrix_2d(p, q)
        N = precompute_N_matrix_2d(du_dx, du_dy, p)
        N_tilde = precompute_N_tilde_matrix_2d(du_dx, du_dy, p)
        F = precompute_F_matrix_2d(N_tilde, p, eta)
        G = precompute_G_matrix_2d(N, p, eta)
        t.I_P_0 = I_P_0
        t.Q_I = Q_I
        t.F = F
        t.G = G
        t.eta = eta

    else:
        P = precompute_P_matrix_2d(p, q)
        Q_D = precompute_Q_D_matrix_2d(p, q, du_dx, du_dy)
        t.P = P
        t.Q_D = Q_D

    return t


def create_solver_obj_3D(
    p: int, q: int, root: Node, uniform_levels: int = None
) -> SolverObj:
    """Creates a tree with l oct tree levels. This means there will be
    8**l leaf nodes. Each side will have 2**l leaf nodes."""
    t = SolverObj()
    all_leaves = get_all_leaves(root)
    if len(all_leaves) == 1 and uniform_levels is not None:
        t.uniform_grid = True
        add_uniform_levels(root, uniform_levels)
    else:
        t.uniform_grid = False
    t.root = root
    t.p = p
    t.q = q

    leaf_cheby_pts = get_all_leaf_3d_cheby_points(p, root)
    root_gauss_pts = get_all_boundary_gauss_legendre_points_3d(q, root)
    t.leaf_cheby_points = leaf_cheby_pts
    t.root_boundary_points = root_gauss_pts
    t.sidelens = jnp.array([l.xmax - l.xmin for l in get_all_leaves(t.root)])

    du_dx, du_dy, du_dz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz = (
        precompute_diff_operators_3d(p, half_side_len=1.0)
    )

    P = precompute_P_matrix_3d(p, q)
    Q_D = precompute_Q_D_matrix_3d(p, q, du_dx, du_dy, du_dz)
    refinement, coarse = precompute_refining_coarsening_ops_3d(q)

    t.D_x = du_dx
    t.D_y = du_dy
    t.D_z = du_dz
    t.D_xx = d_xx
    t.D_yy = d_yy
    t.D_zz = d_zz
    t.D_xy = d_xy
    t.D_xz = d_xz
    t.D_yz = d_yz
    t.P = P
    t.Q_D = Q_D
    t.refinement_op = refinement
    t.coarsening_op = coarse

    return t


def get_bdry_data_evals_lst_2D(
    solver_obj: SolverObj,
    f: Callable[[jnp.array], jnp.array],
) -> List[jnp.array]:

    side_0_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 1] == solver_obj.root.ymin
    ]
    side_1_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 0] == solver_obj.root.xmax
    ]
    side_2_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 1] == solver_obj.root.ymax
    ]
    side_3_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 0] == solver_obj.root.xmin
    ]

    bdry_data_lst = [
        f(side_0_pts),
        f(side_1_pts),
        f(side_2_pts),
        f(side_3_pts),
    ]
    return bdry_data_lst


def get_bdry_data_evals_lst_3D(
    solver_obj: SolverObj,
    f: Callable[[jnp.array], jnp.array],
) -> List[jnp.array]:
    face_1_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 0] == solver_obj.root.xmin
    ]
    face_2_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 0] == solver_obj.root.xmax
    ]
    face_3_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 1] == solver_obj.root.ymin
    ]
    face_4_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 1] == solver_obj.root.ymax
    ]
    face_5_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 2] == solver_obj.root.zmin
    ]
    face_6_pts = solver_obj.root_boundary_points[
        solver_obj.root_boundary_points[:, 2] == solver_obj.root.zmax
    ]

    bdry_data_lst = [
        f(face_1_pts),
        f(face_2_pts),
        f(face_3_pts),
        f(face_4_pts),
        f(face_5_pts),
        f(face_6_pts),
    ]
    return bdry_data_lst
