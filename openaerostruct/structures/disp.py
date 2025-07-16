import numpy as np

import openmdao.api as om


class Disp(om.ExplicitComponent):
    """
    Reshape the flattened displacements from the linear system solution into
    a 2D array so we can more easily use the results.

    The solution to the linear system has meaingless entires due to the
    boundary conditions on the FEM model. The "displacements" from this portion of
    the linear system are not needed, so we select only the relevant
    portion of the displacements for further calculations.

    Parameters
    ----------
    disp_aug[6*(ny+1)] : numpy array
        Augmented displacement array with additional 6 Lagrange multipliers 
        for clamp boundary conditions at the end. Obtained by solving the system
        K * disp_aug = forces, where forces is a flattened version of loads.

    Returns
    -------
    disp[6*ny] : numpy array
        Actual displacement array formed by truncating disp_aug.

    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        surface = self.options["surface"]

        self.ny = surface["mesh"].shape[1]

        # shape of disp_aug depends on the root boundary condition type
        if "root_BC_type" in surface and surface["root_BC_type"] == "pin":
            self.root_BC_pin = True
            disp_aug_size = self.ny * 6 + 3
        else:
            self.root_BC_pin = False
            disp_aug_size = (self.ny + 1) * 6
        self.add_input("disp_aug", val=np.zeros((disp_aug_size)), units="m")

        self.add_output("disp", val=np.zeros((self.ny, 6)), units="m")

        n = self.ny * 6
        arange = np.arange((n))
        self.declare_partials("disp", "disp_aug", val=1.0, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        # Obtain the relevant portions of disp_aug and store the reshaped
        # displacements in disp
        outputs["disp"] = inputs["disp_aug"][:self.ny * 6].reshape((-1, 6))
