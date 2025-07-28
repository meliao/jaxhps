import jax
import jax.numpy as jnp
from ._domain import Domain
from ._discretization_tree import DiscretizationNode3D, get_all_leaves
from ._precompute_operators_2D import (
    precompute_diff_operators_2D,
    precompute_P_2D_DtN,
    precompute_Q_2D_DtN,
    precompute_P_2D_ItI,
    precompute_N_tilde_matrix_2D,
    precompute_N_matrix_2D,
    precompute_G_2D_ItI,
    precompute_QH_2D_ItI,
    precompute_QG_2D_ItI,
    precompute_projection_ops_2D,
)
from ._precompute_operators_3D import (
    precompute_diff_operators_3D,
    precompute_P_3D_DtN,
    precompute_Q_3D_DtN,
    precompute_projection_ops_3D,
)
from typing import List, Tuple


class PDEProblem:
    def __init__(
        self,
        domain: Domain,
        source: jax.Array = None,
        D_xx_coefficients: jax.Array = None,
        D_xy_coefficients: jax.Array = None,
        D_xz_coefficients: jax.Array = None,
        D_yy_coefficients: jax.Array = None,
        D_yz_coefficients: jax.Array = None,
        D_zz_coefficients: jax.Array = None,
        D_x_coefficients: jax.Array = None,
        D_y_coefficients: jax.Array = None,
        D_z_coefficients: jax.Array = None,
        I_coefficients: jax.Array = None,
        use_ItI: bool = False,
        eta: float = None,
    ):
        self.domain: Domain = domain  #: The domain, which contains information about the discretization.

        # Input validation

        # 2D problems shouldn't specify D_z_coefficients
        if isinstance(domain.root, DiscretizationNode3D):
            bool_2D = False

            # 3D code doesn't support ItI merges
            if use_ItI:
                raise NotImplementedError(
                    "ItI merges are not supported for 3D problems."
                )
        else:
            bool_2D = True
            s = "z coefficients can not be set for 2D problems."
            if D_xz_coefficients is not None:
                raise ValueError(s)
            if D_yz_coefficients is not None:
                raise ValueError(s)
            if D_zz_coefficients is not None:
                raise ValueError(s)
            if D_z_coefficients is not None:
                raise ValueError(s)
        # If ItI is being used, eta must be specified
        if use_ItI and eta is None:
            raise ValueError("eta must be specified when using ItI merges.")

        # If ItI is being used, it must be a uniform 2D problem
        if use_ItI and not domain.bool_uniform:
            raise ValueError(
                "ItI merges are only supported for uniform 2D problems."
            )

        # If it's not a uniform 2D problem, the source must be specified
        if source is None:
            if not bool_2D or not domain.bool_uniform:
                raise ValueError(
                    "Source must be specified for non-uniform or 3D problems."
                )

        # Check input shapes are OK
        check_input_shapes(
            source=source,
            use_ItI=use_ItI,
            expected_shape=domain.interior_points[..., 0].shape,
            D_xx_coefficients=D_xx_coefficients,
            D_xy_coefficients=D_xy_coefficients,
            D_xz_coefficients=D_xz_coefficients,
            D_yy_coefficients=D_yy_coefficients,
            D_yz_coefficients=D_yz_coefficients,
            D_zz_coefficients=D_zz_coefficients,
            D_x_coefficients=D_x_coefficients,
            D_y_coefficients=D_y_coefficients,
            D_z_coefficients=D_z_coefficients,
            I_coefficients=I_coefficients,
        )

        # Store coefficients

        #: Coefficient array with shape (n_leaves, p^d).
        self.D_xx_coefficients: jax.Array | None = D_xx_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_xy_coefficients: jax.Array | None = D_xy_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_xz_coefficients: jax.Array | None = D_xz_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_yy_coefficients: jax.Array | None = D_yy_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_yz_coefficients: jax.Array | None = D_yz_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_zz_coefficients: jax.Array | None = D_zz_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_x_coefficients: jax.Array | None = D_x_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_y_coefficients: jax.Array | None = D_y_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.D_z_coefficients: jax.Array | None = D_z_coefficients
        #: Coefficient array with shape (n_leaves, p^d).
        self.I_coefficients: jax.Array | None = I_coefficients
        #: Source function array with shape (n_leaves, p^d).
        self.source: jax.Array | None = source
        self.use_ItI: bool = use_ItI  #: Whether to use ItI merges.
        #: Parameter for ItI matrices and Robin boundary condition.
        self.eta: float | None = eta

        if domain.bool_uniform:
            # In this version of the code, we know the side len of each leaf is the same, so we can scale the diff
            # operators ahead of time.
            # half_side_len = (
            #     (domain.root.xmax - domain.root.xmin) / (2**domain.L) / 2
            # )

            half_side_len = (domain.root.xmax - domain.root.xmin) / (
                2 ** (domain.L + 1)
            )
        else:
            # In this version of the code, the diff operators are scaled separately by the sidelen of each leaf.
            half_side_len = 1.0

            self.sidelens: jax.Array = jnp.array(
                [l.xmax - l.xmin for l in get_all_leaves(domain.root)]
            )

        # Pre-compute spectral differentiation and interpolation matrices
        if bool_2D:
            # Differentiation operators
            # #: Spectral differentiation matrix in x direction. Has shape (p^d, p^d).
            self.D_x: jax.Array = None
            # #: Spectral differentiation matrix in y direction. Has shape (p^d, p^d).
            self.D_y: jax.Array = None
            # #: Spectral differentiation matrix in xx direction. Has shape (p^d, p^d).
            self.D_xx: jax.Array = None
            # #: Spectral differentiation matrix in xy direction. Has shape (p^d, p^d).
            self.D_xy: jax.Array = None
            # #: Spectral differentiation matrix in yy direction. Has shape (p^d, p^d).
            self.D_yy: jax.Array = None
            self.D_x, self.D_y, self.D_xx, self.D_yy, self.D_xy = (
                precompute_diff_operators_2D(domain.p, half_side_len)
            )
            if not use_ItI:
                # Interpolation / Differentiation matrices for DtN merges
                self.P = precompute_P_2D_DtN(domain.p, domain.q)
                self.Q = precompute_Q_2D_DtN(
                    domain.p, domain.q, self.D_x, self.D_y
                )
            else:
                # Interpolation / Differentiation matrices for ItI merges
                self.P = precompute_P_2D_ItI(domain.p, domain.q)

                # In the local solve stage code, F is what the paper calls G, and
                # G is what the paper calls H. The notation in this part is following
                # the paper's notation.
                N_tilde = precompute_N_tilde_matrix_2D(
                    self.D_x, self.D_y, domain.p
                )
                self.G = precompute_G_2D_ItI(N_tilde, self.eta)
                # QH always appear together so we can precompute their product.
                N = precompute_N_matrix_2D(self.D_x, self.D_y, domain.p)
                self.QH = precompute_QH_2D_ItI(N, domain.p, domain.q, self.eta)
                self.QG = precompute_QG_2D_ItI(N, domain.p, domain.q, self.eta)

            # For adaptive case, we need to precompute projection ops
            if not domain.bool_uniform:
                self.L_2f1, self.L_1f2 = precompute_projection_ops_2D(domain.q)
        else:
            # Differentiation operators
            # #: Spectral differentiation matrix in z direction. Has shape (p^d, p^d).
            self.D_z: jax.Array = None
            # #: Spectral differentiation matrix in zz direction. Has shape (p^d, p^d).
            self.D_zz: jax.Array = None
            # #: Spectral differentiation matrix in xz direction. Has shape (p^d, p^d).
            self.D_xz: jax.Array = None
            # #: Spectral differentiation matrix in yz direction. Has shape (p^d, p^d).
            self.D_yz: jax.Array = None
            (
                self.D_x,
                self.D_y,
                self.D_z,
                self.D_xx,
                self.D_yy,
                self.D_zz,
                self.D_xy,
                self.D_xz,
                self.D_yz,
            ) = precompute_diff_operators_3D(
                p=domain.p, half_side_len=half_side_len
            )
            self.P = precompute_P_3D_DtN(domain.p, domain.q)
            self.Q = precompute_Q_3D_DtN(
                domain.p, domain.q, self.D_x, self.D_y, self.D_z
            )

            # For adaptive case, we need to precompute projection ops
            if not domain.bool_uniform:
                self.L_4f1, self.L_1f4 = precompute_projection_ops_3D(domain.q)

        # Set up containers for the solution operators.
        self.Y: jax.Array = None  #: (jax.Array) Stores pre-computed interior solution operators.
        self.v: jax.Array = None  #: (jax.Array) Stores pre-computed interior particular solutions.
        self.S_lst: List[
            jax.Array
        ] = []  #: (jax.Array) Stores pre-computed propagation operators when performing uniform merges.
        self.g_tilde_lst: List[
            jax.Array
        ] = []  #: (jax.Array) Stores pre-computed incoming data along merge interfaces when performing uniform merges.
        self.D_inv_lst: List[
            jax.Array
        ] = []  #: (jax.Array) Stores pre-computed operators for the upward pass.
        self.BD_inv_lst: List[
            jax.Array
        ] = []  #: (jax.Array) Stores pre-computed BD^{-1} operators for the upward pass.
        self.Phi: jax.Array = None  #: (jax.Array) Stores pre-computed particular solution operators.

    def reset(self) -> None:
        """
        Resets the stored solution operators.
        """

        self.Y = None
        self.v = None
        self.S_lst = []
        self.g_tilde_lst = []
        self.D_inv_lst = []
        self.BD_inv_lst = []
        self.Phi = None

    def update_coefficients(
        self,
        source: jax.Array = None,
        D_xx_coefficients: jax.Array = None,
        D_xy_coefficients: jax.Array = None,
        D_xz_coefficients: jax.Array = None,
        D_yy_coefficients: jax.Array = None,
        D_yz_coefficients: jax.Array = None,
        D_zz_coefficients: jax.Array = None,
        D_x_coefficients: jax.Array = None,
        D_y_coefficients: jax.Array = None,
        D_z_coefficients: jax.Array = None,
        I_coefficients: jax.Array = None,
    ) -> None:
        """

        This function can be used to update the coefficients of the PDEProblem. When this function is called, it resets the
        stored solution operators.

        Args:
            source (jax.Array, optional): If specified, will replace the current source. Defaults to None.

            D_xx_coefficients (jax.Array, optional): If specified, will replace the current D_xx_coefficients. Defaults to None.

            D_xy_coefficients (jax.Array, optional): If specified, will replace the current D_xy_coefficients. Defaults to None.

            D_xz_coefficients (jax.Array, optional): If specified, will replace the current D_xz_coefficients. Defaults to None.

            D_yy_coefficients (jax.Array, optional): If specified, will replace the current D_yy_coefficients. Defaults to None.

            D_yz_coefficients (jax.Array, optional): If specified, will replace the current D_yz_coefficients. Defaults to None.

            D_zz_coefficients (jax.Array, optional): If specified, will replace the current D_zz_coefficients. Defaults to None.

            D_x_coefficients (jax.Array, optional): If specified, will replace the current D_x_coefficients. Defaults to None.

            D_y_coefficients (jax.Array, optional): If specified, will replace the current D_y_coefficients. Defaults to None.

            D_z_coefficients (jax.Array, optional): If specified, will replace the current D_z_coefficients. Defaults to None.

            I_coefficients (jax.Array, optional): If specified, will replace the current I_coefficients. Defaults to None.
        """
        self.reset()

        # Check input shapes are OK
        check_input_shapes(
            source=self.source,
            use_ItI=self.use_ItI,
            expected_shape=self.domain.interior_points[..., 0].shape,
            D_xx_coefficients=D_xx_coefficients,
            D_xy_coefficients=D_xy_coefficients,
            D_xz_coefficients=D_xz_coefficients,
            D_yy_coefficients=D_yy_coefficients,
            D_yz_coefficients=D_yz_coefficients,
            D_zz_coefficients=D_zz_coefficients,
            D_x_coefficients=D_x_coefficients,
            D_y_coefficients=D_y_coefficients,
            D_z_coefficients=D_z_coefficients,
            I_coefficients=I_coefficients,
        )

        # Update coefficients

        if source is not None:
            self.source = source
        if D_xx_coefficients is not None:
            self.D_xx_coefficients = D_xx_coefficients
        if D_xy_coefficients is not None:
            self.D_xy_coefficients = D_xy_coefficients
        if D_xz_coefficients is not None:
            self.D_xz_coefficients = D_xz_coefficients
        if D_yy_coefficients is not None:
            self.D_yy_coefficients = D_yy_coefficients
        if D_yz_coefficients is not None:
            self.D_yz_coefficients = D_yz_coefficients
        if D_zz_coefficients is not None:
            self.D_zz_coefficients = D_zz_coefficients
        if D_x_coefficients is not None:
            self.D_x_coefficients = D_x_coefficients
        if D_y_coefficients is not None:
            self.D_y_coefficients = D_y_coefficients
        if D_z_coefficients is not None:
            self.D_z_coefficients = D_z_coefficients
        if I_coefficients is not None:
            self.I_coefficients = I_coefficients


def _get_PDEProblem_chunk(
    pde_problem: PDEProblem, start_idx: int, end_idx: int
) -> PDEProblem:
    """
    Returns a new instance of PDEProblem, which has the same domain and precomputed differential operators.
    HOWEVER, the coefficients and source terms are sliced along the first axis (i.e., the number of leaves)
    to match the specified start and end indices.

    This is used when doing local solves in batches. It skips the precomputation of the differentiation and
    interpolation matrices; it just copies them over from the original object.

    This is a bit of a hack.

    Args:
        pde_problem (PDEProblem): Instance we want to copy data from.
        start_idx (int): Start idx along the number of leaves axis.
        end_idx (int): End idx along the number of leaves axis.

    Returns:
        PDEProblem: New instance with copies of the precomptued operators and slices of the coefficients and source.
    """

    # Get a new instance without calling __init__
    new_pde_problem = PDEProblem.__new__(PDEProblem)

    # Copy the domain and metadata
    new_pde_problem.domain = pde_problem.domain
    new_pde_problem.use_ItI = pde_problem.use_ItI
    new_pde_problem.eta = pde_problem.eta

    # Copy the differential operators
    new_pde_problem.D_x = pde_problem.D_x
    new_pde_problem.D_y = pde_problem.D_y
    new_pde_problem.D_xx = pde_problem.D_xx
    new_pde_problem.D_xy = pde_problem.D_xy
    new_pde_problem.D_yy = pde_problem.D_yy
    # 3D differential operators
    if not pde_problem.domain.bool_2D:
        new_pde_problem.D_z = pde_problem.D_z
        new_pde_problem.D_yz = pde_problem.D_yz
        new_pde_problem.D_xz = pde_problem.D_xz
        new_pde_problem.D_zz = pde_problem.D_zz

    # Copy the interpolation operators
    if pde_problem.use_ItI:
        # 2D uniform ItI case
        new_pde_problem.P = pde_problem.P
        new_pde_problem.G = pde_problem.G
        new_pde_problem.QH = pde_problem.QH

    else:
        if pde_problem.domain.bool_2D:
            # 2D DtN case
            new_pde_problem.P = pde_problem.P
            new_pde_problem.Q = pde_problem.Q

            if not pde_problem.domain.bool_uniform:
                new_pde_problem.L_2f1 = pde_problem.L_2f1
                new_pde_problem.L_1f2 = pde_problem.L_1f2
        else:
            # 3D DtN case
            new_pde_problem.P = pde_problem.P
            new_pde_problem.Q = pde_problem.Q

            if not pde_problem.domain.bool_uniform:
                new_pde_problem.L_4f1 = pde_problem.L_4f1
                new_pde_problem.L_1f4 = pde_problem.L_1f4
                new_pde_problem.sidelens = pde_problem.sidelens[
                    start_idx:end_idx
                ]

    # Copy slices of the coefficients and source terms
    new_pde_problem.D_xx_coefficients = (
        pde_problem.D_xx_coefficients[start_idx:end_idx]
        if pde_problem.D_xx_coefficients is not None
        else None
    )
    new_pde_problem.D_xy_coefficients = (
        pde_problem.D_xy_coefficients[start_idx:end_idx]
        if pde_problem.D_xy_coefficients is not None
        else None
    )
    new_pde_problem.D_xz_coefficients = (
        pde_problem.D_xz_coefficients[start_idx:end_idx]
        if pde_problem.D_xz_coefficients is not None
        else None
    )
    new_pde_problem.D_yy_coefficients = (
        pde_problem.D_yy_coefficients[start_idx:end_idx]
        if pde_problem.D_yy_coefficients is not None
        else None
    )
    new_pde_problem.D_yz_coefficients = (
        pde_problem.D_yz_coefficients[start_idx:end_idx]
        if pde_problem.D_yz_coefficients is not None
        else None
    )
    new_pde_problem.D_zz_coefficients = (
        pde_problem.D_zz_coefficients[start_idx:end_idx]
        if pde_problem.D_zz_coefficients is not None
        else None
    )
    new_pde_problem.D_x_coefficients = (
        pde_problem.D_x_coefficients[start_idx:end_idx]
        if pde_problem.D_x_coefficients is not None
        else None
    )
    new_pde_problem.D_y_coefficients = (
        pde_problem.D_y_coefficients[start_idx:end_idx]
        if pde_problem.D_y_coefficients is not None
        else None
    )
    new_pde_problem.D_z_coefficients = (
        pde_problem.D_z_coefficients[start_idx:end_idx]
        if pde_problem.D_z_coefficients is not None
        else None
    )
    new_pde_problem.I_coefficients = (
        pde_problem.I_coefficients[start_idx:end_idx]
        if pde_problem.I_coefficients is not None
        else None
    )
    # Source term
    new_pde_problem.source = (
        pde_problem.source[start_idx:end_idx]
        if pde_problem.source is not None
        else None
    )

    # Return the new instance
    return new_pde_problem


def check_input_shapes(
    source: jax.Array,
    use_ItI: bool,
    expected_shape: Tuple[int],
    D_xx_coefficients: jax.Array = None,
    D_xy_coefficients: jax.Array = None,
    D_xz_coefficients: jax.Array = None,
    D_yy_coefficients: jax.Array = None,
    D_yz_coefficients: jax.Array = None,
    D_zz_coefficients: jax.Array = None,
    D_x_coefficients: jax.Array = None,
    D_y_coefficients: jax.Array = None,
    D_z_coefficients: jax.Array = None,
    I_coefficients: jax.Array = None,
) -> None:
    # Check input shapes are OK
    check_lst = [
        (D_xx_coefficients, "D_xx_coefficients"),
        (D_xy_coefficients, "D_xy_coefficients"),
        (D_xz_coefficients, "D_xz_coefficients"),
        (D_yy_coefficients, "D_yy_coefficients"),
        (D_yz_coefficients, "D_yz_coefficients"),
        (D_zz_coefficients, "D_zz_coefficients"),
        (D_x_coefficients, "D_x_coefficients"),
        (D_y_coefficients, "D_y_coefficients"),
        (D_z_coefficients, "D_z_coefficients"),
        (I_coefficients, "I_coefficients"),
    ]
    # if not use_ItI:
    #     # Other parts of the ItI code use source terms that have shape [n_leaves, p**2, n_src]
    #     check_lst.append((source, "source"))
    for arr, name in check_lst:
        if arr is not None:
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {arr.shape} but should have shape {expected_shape} to match the Domain's interior points."
                )
