"""Define the LinearSystemComp class."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

import openmdao.api as om

from .fem import get_drdu_sparsity_pattern, get_drdK_sparsity_pattern


def _get_joint_size(joint_type):
    """
    Return the number of joint constraints for a given joint type.
    """
    if joint_type == 'ball':
        n_con = 3   # number of joint constraints: translational only
    elif joint_type == 'pin':
        n_con = 5   # translational and rotational in y and z
    elif joint_type == 'rigid':
        n_con = 6   # ranslational and rotational in x, y, and z
    else:
        raise ValueError("Joint type must be either 'pin' or 'rigid'.")
    return n_con

def get_joint_idx_from_y(surface, joint_y):
    """
    Return the joint index from the joint y (spanwise) coordinate.
    This is only good for wing or strut, but do not use for jury (or any vertical surface)
    """
    if surface["name"] == "jury":
        raise RuntimeError("Cannot use get_joint_idx_from_y for jury surface.")
    joint_idx = np.argmin((np.abs(surface["mesh"][0, :, 1]) - abs(joint_y))**2)
    return joint_idx

def _joint_rows_cols(joint_type, surface, joint_idx):
    """
    Return rows and cols indices for joint constraints.
    rows and cols are defined for the component-level indexing, but not the global wing-strut-jury system.
    """
    # partials of joint constraint residuals w.r.t. displacement
    n_con = _get_joint_size(joint_type)
    rows = np.arange(n_con)
    if joint_type in ['ball', 'rigid']:
        cols = np.arange(n_con) + 6 * joint_idx
    elif joint_type == 'pin':
        cols = np.array([0, 1, 2, 4, 5]) + 6 * joint_idx  # exclude x-rotation

    return rows, cols


class FEMStrutBraced(om.ImplicitComponent):
    """
    Component that solves an FEM linear system, K u = f.
    This component is a customization of original fem.py to solve strut-braced wing structure.

    This solve the following linear system for the wing-strut joint problem:
    
    [ K1 | 0  | C1^T ] [u1]       [f1]
    [ 0  | K2 | C2^T ] [u2]     = [f2]
    [ C1 | C2 | 0    ] [lambda]   [0 ]

    where:
        K1     = the stiffness matrix of the wing + additional entries for wing root boundary conditions
        u1     = augmented displacement of the wing (includes Lagrange multipliers for boundary conditions)
        f1     = augmented force vector of the wing (includes zeros at the end for boundary conditions)
        K2     = the stiffness matrix of the strut + additional entries for strut root boundary conditions
        u2     = augmented displacement of the strut (includes Lagrange multipliers for boundary conditions)
        f2     = augmented force vector of the strut (includes zeros at the end for boundary conditions)
        C1, C2 = constraint matrix for the wing-strut joint
        lambda = Lagrange multipliers for the joint constraints (physically these are joint reaction forces)

    Attributes
    ----------
    _lup : None or list(object)
        matrix factorizations returned from scipy.linag.lu_factor for each A matrix
    k_cols : ndarray
        Cached column indices for sparse representation of stiffness matrix.
    k_rows : ndarray
        Cached row indices for sparse representation of stiffness matrix.
    k_data : ndarray
        Cached values for sparse representation of stiffness matrix.
    """

    def __init__(self, **kwargs):
        """
        Intialize the LinearSystemComp component.

        Parameters
        ----------
        kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(FEMStrutBraced, self).__init__(**kwargs)
        self._lup = None
        self.k_cols = None
        self.k_rows = None
        self.k_data = None

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("surfaces", types=list)
        self.options.declare("vec_size", types=int, default=1, desc="Number of linear systems to solve.")

    def setup(self):
        """
        Matrix and RHS are inputs, solution vector is the output.
        """
        # --- input option check and processings ---
        vec_size = self.options["vec_size"]
        self.surfaces = self.options["surfaces"]
        self.surface_names = [surface["name"] for surface in self.surfaces]

        # surfaces must be either [wing, strut] or [wing, strut, jury] in this order
        if len(self.surfaces) == 2:
            if self.surface_names != ["wing", "strut"]:
                raise ValueError("FEMStrutBraced component requires the surfaces to be in the order of [wing, strut].")
            self.include_jury = False
            self.wing_surface = self.surfaces[0]
            self.strut_surface = self.surfaces[1]
        elif len(self.surfaces) == 3:
            if self.surface_names != ["wing", "strut", "jury"]:
                raise ValueError("FEMStrutBraced component requires the surfaces to be in the order of [wing, strut, jury].")
            self.include_jury = True
            self.wing_surface = self.surfaces[0]
            self.strut_surface = self.surfaces[1]
            self.jury_surface = self.surfaces[2]
            raise NotImplementedError("Jury strut FEM model is still under debugging")
        else:
            raise ValueError("FEMStrutBraced component requires exactly two or three surfaces (wing, strut, and/or jury).")

        # no support for asymmetric surfaces
        for surface in self.surfaces:
            if not surface["symmetry"]:
                raise NotImplementedError("Strut-braced wing joint is only implemented for symmetric surfaces.")

        # We only support the surface name to be "wing" and "strut", and "jury" for now
        for surface in self.surfaces:
            if surface["name"] not in ["wing", "strut", "jury"]:
                raise ValueError("FEMStrutBraced component does not support surface name: " + surface["name"])

        # get joint information
        wing_strut_joint_type = self.strut_surface["wing_strut_joint_type"]
        wing_strut_joint_y = self.strut_surface["wing_strut_joint_y"]

        if self.include_jury:
            wing_jury_joint_y = self.jury_surface["wing_jury_joint_y"]
            strut_jury_joint_y = self.jury_surface["strut_jury_joint_y"]

        # prepare inputs
        self.ny = []  # number of spanwise nodes
        self.size = []   # list of size (num rows) of assembled K matrix for each surface
        self.num_k_data = []  # number of non-zero entries for assembled K matrix for each surface
        self._lup = []
        k_cols = []  # column indices for stiffness matrix sparsity pattern
        k_rows = []  # row indices for stiffness matrix sparsity pattern

        for i, surface in enumerate(self.surfaces):
            name = surface["name"]

            # --- inputs, outputs, partials of FEM system for each surface ---
            ny = surface["mesh"].shape[1]
            if "root_BC_type" in surface and surface["root_BC_type"] == "ball":
                # ball boundary condition at the root. Constrain translation only.
                dof_of_boundary = 3
                root_BC = "ball"
            elif "root_BC_type" in surface and surface["root_BC_type"] == "pin":
                # pin boundary condition at the root. Constrain translation and rotation in y and z.
                dof_of_boundary = 5
                root_BC = "pin"
            elif "root_BC_type" in surface and surface["root_BC_type"] == "none":
                # no root boundary conditions for jury strut
                dof_of_boundary = 0
                root_BC = "none"
            else:
                # rigid boundary condition at the root. Constrain translation and rotation.
                dof_of_boundary = 6
                root_BC = "rigid"
            self.ny.append(ny)
            size = int(6 * ny + dof_of_boundary)
            self.size.append(size)
            
            full_size = size * vec_size
            shape = (vec_size, size) if vec_size > 1 else (size,)
            init_locK = np.tile(np.eye(12).flatten(), ny - 1).reshape(ny - 1, 12, 12)
            self.add_input(f"local_stiff_transformed_{name}", val=init_locK)
            self.add_input(f"forces_{name}", val=np.ones(shape), units="N")
            self.add_output(f"disp_aug_{name}", shape=shape, val=0.1, units="m")

            # Set up the derivatives.
            row_col = np.arange(full_size, dtype="int")
            self.declare_partials(f"disp_aug_{name}", f"forces_{name}", val=np.full(full_size, -1.0), rows=row_col, cols=row_col)

            # The derivative of residual wrt displacements is the stiffness matrix K. We can use the
            # sparsity pattern here and when constucting the sparse matrix, so save rows and cols.
            rows, cols, vec_rows, vec_cols = get_drdu_sparsity_pattern(ny, vec_size, surface["symmetry"], root_BC)
            if i == 1:
                # the second surface (strut) goes to the lower-right block of the entire stiffness matrix
                rows += self.size[0]
                cols += self.size[0]
            k_rows.append(rows)
            k_cols.append(cols)
            self.num_k_data.append(len(rows))
            self.declare_partials(of=f"disp_aug_{name}", wrt=f"disp_aug_{name}", rows=vec_rows, cols=vec_cols)

            rows, cols = get_drdK_sparsity_pattern(ny)
            self.declare_partials(f"disp_aug_{name}", f"local_stiff_transformed_{name}", rows=rows, cols=cols)

        # --- wing-strut joint constraints (coupling terms between two surfaces) ---
        # list of all joing Lagrange multipliers (will use later)
        self.joint_list = []
        # number of joint constraints
        self.n_con = 0  # total number of joint constraints (wing-strut + wing-jury + strut-jury)
        n_con_ws = self.n_con_ws = _get_joint_size(wing_strut_joint_type)
        self.n_con += n_con_ws

        # add Lagrange multipliers for joint constraints as state variables
        self.add_output("joint_Lag_wing_strut", shape=(n_con_ws,), val=0.0)
        self.joint_list.append("wing_strut")

        k_rows_joint_ws = []
        k_cols_joint_ws = []
        self.k_data_joint = []  # for all joints (wing-strut, wing-jury, strut-jury)
        for i, surface in enumerate([self.wing_surface, self.strut_surface]):
            name = surface["name"]

            # partials of joint constraint residuals w.r.t. displacement
            joint_idx = get_joint_idx_from_y(surface, wing_strut_joint_y)
            print(f'Wing-strut joint ({surface["name"]}): joint y = {surface["mesh"][0, joint_idx, 1]} and joint index = {joint_idx}')
            rows, cols = _joint_rows_cols(wing_strut_joint_type, surface, joint_idx)
            if i == 0:
                vals = np.ones(n_con_ws) * 1e9
            else:
                vals = np.ones(n_con_ws) * -1e9
            self.declare_partials("joint_Lag_wing_strut", f"disp_aug_{name}", rows=rows, cols=cols, val=vals)

            # partials of FEM residuals (r = Ku - f) w.r.t. joint Lagrange multipliers: transpose of above
            self.declare_partials(f"disp_aug_{name}", "joint_Lag_wing_strut", rows=cols, cols=rows, val=vals)

            # rows/cols in the global K matrix (bottom of the matrix)
            rows += sum(self.size)
            if name == "strut":
                cols += self.size[0]

            # append these entries to the bottom of the global K matrix and its transpose
            k_rows_joint_ws.append(rows)
            k_cols_joint_ws.append(cols)
            self.k_data_joint.append(vals)
            k_rows_joint_ws.append(cols)
            k_cols_joint_ws.append(rows)
            self.k_data_joint.append(vals)

        # row and col indices for the entire stiffness matrix (that combines wing, strut, (jury), and joint constraints)
        self.k_cols = np.concatenate(k_cols + k_cols_joint_ws)
        self.k_rows = np.concatenate(k_rows + k_rows_joint_ws)
        self.total_size = sum(self.size) + n_con_ws   # size (num rows) of total stiffness matrix (K matrices + joint constraints)

        if self.include_jury:
            # --- wing-jury joint constraints ---
            n_con_wj = self.n_con_wj = 6  # need to constrain all DOFs to avoid singularity
            self.n_con += n_con_wj

            # add Lagrange multipliers for joint constraints as state variables
            self.add_output("joint_Lag_wing_jury", shape=(n_con_wj,), val=0.0)
            self.joint_list.append("wing_jury")

            k_rows_joint_wj = []
            k_cols_joint_wj = []
            for i, surface in enumerate([self.wing_surface, self.jury_surface]):
                name = surface["name"]

                # partials of joint constraint residuals w.r.t. displacement
                if name == "wing":
                    joint_idx = get_joint_idx_from_y(surface, wing_jury_joint_y)
                elif name == "jury":
                    joint_idx = 0  # jury index starts from 0 at the wing-jury joint
                print(f'Wing-jury joint ({surface["name"]}): joint y = {surface["mesh"][0, joint_idx, 1]} and joint index = {joint_idx}')
                rows, cols = _joint_rows_cols("rigid", surface, joint_idx)
                if i == 0:
                    vals = np.ones(n_con_wj) * 1e9
                else:
                    vals = np.ones(n_con_wj) * -1e9
                self.declare_partials("joint_Lag_wing_jury", f"disp_aug_{name}", rows=rows, cols=cols, val=vals)

                # partials of FEM residuals (r = Ku - f) w.r.t. joint Lagrange multipliers: transpose of above
                self.declare_partials(f"disp_aug_{name}", "joint_Lag_wing_jury", rows=cols, cols=rows, val=vals)

                # append these entries to the global K matrix
                rows += self.total_size
                if name == "jury":
                    cols += self.size[0] + self.size[1]

                k_rows_joint_wj.append(rows)
                k_cols_joint_wj.append(cols)
                self.k_data_joint.append(vals)
                k_rows_joint_wj.append(cols)
                k_cols_joint_wj.append(rows)
                self.k_data_joint.append(vals)

            # row and col indices for the entire stiffness matrix
            self.k_cols = np.concatenate((self.k_cols, np.concatenate(k_cols_joint_wj)))
            self.k_rows = np.concatenate((self.k_rows, np.concatenate(k_rows_joint_wj)))
            self.total_size += n_con_wj

            # --- strut-jury joint constraints ---
            n_con_sj = self.n_con_sj = 6
            self.n_con += n_con_sj

            # add Lagrange multipliers for joint constraints as state variables
            self.add_output("joint_Lag_strut_jury", shape=(n_con_sj,), val=0.0)
            self.joint_list.append("strut_jury")

            k_rows_joint_sj = []
            k_cols_joint_sj = []
            for i, surface in enumerate([self.strut_surface, self.jury_surface]):
                name = surface["name"]

                # partials of joint constraint residuals w.r.t. displacement
                if name == "strut":
                    joint_idx = get_joint_idx_from_y(surface, strut_jury_joint_y)
                elif name == "jury":
                    joint_idx = surface["mesh"].shape[1] - 1  # last index of jury surface
                print(f'Strut-jury joint ({surface["name"]}): joint y = {surface["mesh"][0, joint_idx, 1]} and joint index = {joint_idx}')
                rows, cols = _joint_rows_cols("rigid", surface, joint_idx)
                if i == 0:
                    vals = np.ones(n_con_sj) * 1e9
                else:
                    vals = np.ones(n_con_sj) * -1e9
                self.declare_partials("joint_Lag_strut_jury", f"disp_aug_{name}", rows=rows, cols=cols, val=vals)

                # partials of FEM residuals (r = Ku - f) w.r.t. joint Lagrange multipliers: transpose of above
                self.declare_partials(f"disp_aug_{name}", "joint_Lag_strut_jury", rows=cols, cols=rows, val=vals)

                # append these entries to the bottom of the global K matrix (and the transpose to the right)
                rows += self.total_size
                if name == "strut":
                    cols += self.size[0]
                if name == "jury":
                    cols += self.size[0] + self.size[1]

                k_rows_joint_sj.append(rows)
                k_cols_joint_sj.append(cols)
                self.k_data_joint.append(vals)
                k_rows_joint_sj.append(cols)
                k_cols_joint_sj.append(rows)
                self.k_data_joint.append(vals)

            # row and col indices for the entire stiffness matrix
            self.k_cols = np.concatenate((self.k_cols, np.concatenate(k_cols_joint_sj)))
            self.k_rows = np.concatenate((self.k_rows, np.concatenate(k_rows_joint_sj)))
            self.total_size += n_con_sj

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        R = Ax - b.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        K = self.assemble_CSC_K(inputs)
        disp = np.concatenate([outputs[f"disp_aug_{name}"] for name in self.surface_names])
        joint_Lag = np.concatenate([outputs[f"joint_Lag_{name}"] for name in self.joint_list])
        u = np.concatenate([disp, joint_Lag])
        force = np.concatenate([inputs[f"forces_{name}"] for name in self.surface_names])
        rhs = np.concatenate([force, np.zeros(self.n_con)])

        r = K.dot(u) - rhs

        size0, size1 = self.size[0], self.size[1]  # size of each component's K matrix
        sum_size = sum(self.size)
        residuals[f"disp_aug_{self.surface_names[0]}"] = r[:size0]
        residuals[f"disp_aug_{self.surface_names[1]}"] = r[size0:size0 + size1]
        residuals["joint_Lag_wing_strut"] = r[sum_size:sum_size + self.n_con_ws]
        if self.include_jury:
            residuals[f"disp_aug_{self.surface_names[2]}"] = r[size0 + size1:sum_size]
            residuals["joint_Lag_wing_jury"] = r[sum_size + self.n_con_ws:sum_size + self.n_con_ws + self.n_con_wj]
            residuals["joint_Lag_strut_jury"] = r[sum_size + self.n_con_ws + self.n_con_wj:]
        
    def solve_nonlinear(self, inputs, outputs):
        """
        Use numpy to solve Ax=b for x.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        # lu factorization for use with solve_linear
        K = self.assemble_CSC_K(inputs)

        self._lup = splu(K)
        force = np.concatenate([inputs[f"forces_{name}"] for name in self.surface_names])
        rhs = np.concatenate([force, np.zeros(self.n_con)])

        u = self._lup.solve(rhs)

        size0, size1 = self.size[0], self.size[1]  # size of each displacement vectors
        sum_size = sum(self.size)
        outputs[f"disp_aug_{self.surface_names[0]}"] = u[:size0]
        outputs[f"disp_aug_{self.surface_names[1]}"] = u[size0:size0 + size1]
        outputs["joint_Lag_wing_strut"] = u[sum_size:sum_size + self.n_con_ws]
        if self.include_jury:
            outputs[f"disp_aug_{self.surface_names[2]}"] = u[size0 + size1:sum_size]
            outputs["joint_Lag_wing_jury"] = u[sum_size + self.n_con_ws:sum_size + self.n_con_ws + self.n_con_wj]
            outputs["joint_Lag_strut_jury"] = u[sum_size + self.n_con_ws + self.n_con_wj:]

    def linearize(self, inputs, outputs, J):
        """
        Compute the non-constant partial derivatives.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        J : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """
        if self.include_jury:
            raise RuntimeError("Not implemented for jury strut")

        vec_size = self.options["vec_size"]

        name0, name1 = self.surface_names[0], self.surface_names[1]
        ny0, ny1 = self.ny[0], self.ny[1]
        idx0 = np.tile(np.tile(np.arange(12), 12), ny0 - 1) + np.repeat(6 * np.arange(ny0 - 1), 144)
        idx1 = np.tile(np.tile(np.arange(12), 12), ny1 - 1) + np.repeat(6 * np.arange(ny1 - 1), 144)
        disp0 = outputs[f"disp_aug_{name0}"]
        disp1 = outputs[f"disp_aug_{name1}"]
        J[f"disp_aug_{name0}", f"local_stiff_transformed_{name0}"] = np.tile(disp0[idx0], vec_size)
        J[f"disp_aug_{name1}", f"local_stiff_transformed_{name1}"] = np.tile(disp1[idx1], vec_size)

        nk0, nk1 = self.num_k_data[0], self.num_k_data[1]
        k_data0 = self.k_data[:nk0]
        k_data1 = self.k_data[nk0:nk0 + nk1]
        J[f"disp_aug_{name0}", f"disp_aug_{name0}"] = np.tile(k_data0, vec_size)
        J[f"disp_aug_{name1}", f"disp_aug_{name1}"] = np.tile(k_data1, vec_size)

    def solve_linear(self, d_outputs, d_residuals, mode):
        r"""
        Back-substitution to solve the derivatives of the linear system.

        If mode is:
            'fwd': d_residuals \|-> d_outputs

            'rev': d_outputs \|-> d_residuals

        Parameters
        ----------
        d_outputs : Vector
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals : Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode : str
            either 'fwd' or 'rev'
        """
        if self.include_jury:
            raise RuntimeError("Not implemented for jury strut")

        vec_size = self.options["vec_size"]
        size0, size1 = self.size[0], self.size[1]
        name0, name1 = self.surface_names[0], self.surface_names[1]

        # TODO: verify (check_partials doesn't cover this)

        if mode == "fwd":
            if vec_size > 1:
                for j in range(vec_size):
                    rhs = np.concatenate((d_residuals[f"disp_aug_{name0}"][j], d_residuals[f"disp_aug_{name1}"][j], d_residuals["joint_Lag_wing_strut"][j]))
                    sol = self._lup.solve(rhs)
                    d_outputs[f"disp_aug_{name0}"][j] = sol[:size0]
                    d_outputs[f"disp_aug_{name1}"][j] = sol[size0:size0 + size1]
                    d_outputs["joint_Lag_wing_strut"][j] = sol[size0 + size1:]
            else:
                rhs = np.concatenate((d_residuals[f"disp_aug_{name0}"], d_residuals[f"disp_aug_{name1}"], d_residuals["joint_Lag_wing_strut"]))
                sol = self._lup.solve(rhs)
                d_outputs[f"disp_aug_{name0}"] = sol[:size0]
                d_outputs[f"disp_aug_{name1}"] = sol[size0:size0 + size1]
                d_outputs["joint_Lag_wing_strut"] = sol[size0 + size1:]
        else:
            if vec_size > 1:
                for j in range(vec_size):
                    rhs = np.concatenate((d_outputs[f"disp_aug_{name0}"][j], d_outputs[f"disp_aug_{name1}"][j], d_outputs["joint_Lag_wing_strut"][j]))
                    sol = self._lup.solve(rhs)
                    d_residuals[f"disp_aug_{name0}"][j] = sol[:size0]
                    d_residuals[f"disp_aug_{name1}"][j] = sol[size0:size0 + size1]
                    d_residuals["joint_Lag_wing_strut"][j] = sol[size0 + size1:]
            else:
                rhs = np.concatenate((d_outputs[f"disp_aug_{name0}"], d_outputs[f"disp_aug_{name1}"], d_outputs["joint_Lag_wing_strut"]))
                sol = self._lup.solve(rhs)
                d_residuals[f"disp_aug_{name0}"] = sol[:size0]
                d_residuals[f"disp_aug_{name1}"] = sol[size0:size0 + size1]
                d_residuals["joint_Lag_wing_strut"] = sol[size0 + size1:]

    def assemble_CSC_K(self, inputs):
        """
        Assemble the stiffness matrix in sparse CSC format.

        Returns
        -------
        ndarray
            Stiffness matrix as dense ndarray.
        """
        data = {}
        for surface in self.surfaces:
            name = surface["name"]
            k_loc = inputs[f"local_stiff_transformed_{name}"]

            data1 = k_loc[:, :6, 6:].flatten()
            data2 = k_loc[:, 6:, :6].flatten()
            data3 = k_loc[0, :6, :6].flatten()
            data4 = k_loc[-1, 6:, 6:].flatten()
            data5 = (k_loc[0:-1, 6:, 6:] + k_loc[1:, :6, :6]).flatten()

            # data6 corresponds to the root boundary condition
            if "root_BC_type" in surface and surface["root_BC_type"] == "ball":
                # for ball BC (constraint translation only)
                data6 = np.full((3,), 1e9)
            elif "root_BC_type" in surface and surface["root_BC_type"] == "pin":
                # for pin BC (constraint translation and rotation in y and z)
                data6 = np.full((5,), 1e9)
            elif "root_BC_type" in surface and surface["root_BC_type"] == "none":
                data6 = np.array([])
            else:
                # for rigid BC (constraint translation and rotation)
                data6 = np.full((6,), 1e9)
            
            data[name] = np.concatenate([data1, data2, data3, data4, data5, data6, data6])

        # combine matrices for wing, strut, and joint constraints
        self.k_data = np.concatenate(list(data.values()) + self.k_data_joint)

        return coo_matrix((self.k_data, (self.k_rows, self.k_cols)), shape=(self.total_size, self.total_size)).tocsc()
