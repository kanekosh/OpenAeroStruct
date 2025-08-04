import numpy as np
import openmdao.api as om
from openaerostruct.structures.create_rhs import CreateRHS
from openaerostruct.structures.fem import FEM
from openaerostruct.structures.fem_strut_braced import FEMStrutBraced
from openaerostruct.structures.disp import Disp
from openaerostruct.structures.wing_weight_loads import StructureWeightLoads
from openaerostruct.structures.fuel_loads import FuelLoads
from openaerostruct.structures.total_loads import TotalLoads
from openaerostruct.structures.compute_point_mass_loads import ComputePointMassLoads
from openaerostruct.structures.compute_thrust_loads import ComputeThrustLoads


class FEMloads(om.Group):
    """
    Compute FEM right-hand side force vector

    Parameters
    ----------
    nodes : ndarray
        The coordinates of the FEM nodes.
    load_factor : float
        The load factor applied to the structure.
    element_mass : ndarray
        The mass of each element in the structure for bending relief.
    fuel_mass : ndarray
        Total fuel mass for bendling relief
    fuel_vols : ndarray
        The feul volume of each element
    point_mass_locations : ndarray
        The locations of the point masses.
    point_masses : ndarray
        Point masses as point load
    engine_thrusts : ndarray
        Engine thrusts as point load

    Returns
    -------
    forces : ndarray
        The right-hand side force vector for the FEM. This is a flattened array of loads
        plus zeros at the end for boundary condition
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        surface = self.options["surface"]

        promotes = []
        if surface["struct_weight_relief"]:
            self.add_subsystem(
                "struct_weight_loads",
                StructureWeightLoads(surface=surface),
                promotes_inputs=["element_mass", "nodes", "load_factor"],
                promotes_outputs=["struct_weight_loads"],
            )
            promotes.append("struct_weight_loads")

        if surface["distributed_fuel_weight"]:
            self.add_subsystem(
                "fuel_loads",
                FuelLoads(surface=surface),
                promotes_inputs=["nodes", "load_factor", "fuel_vols", "fuel_mass"],
                promotes_outputs=["fuel_weight_loads"],
            )
            promotes.append("fuel_weight_loads")

        if "n_point_masses" in surface.keys():
            self.add_subsystem(
                "point_masses",
                ComputePointMassLoads(surface=surface),
                promotes_inputs=["point_mass_locations", "point_masses", "nodes", "load_factor"],
                promotes_outputs=["loads_from_point_masses"],
            )
            promotes.append("loads_from_point_masses")

            self.add_subsystem(
                "thrust_loads",
                ComputeThrustLoads(surface=surface),
                promotes_inputs=["point_mass_locations", "engine_thrusts", "nodes"],
                promotes_outputs=["loads_from_thrusts"],
            )
            promotes.append("loads_from_thrusts")

        self.add_subsystem(
            "total_loads",
            TotalLoads(surface=surface),
            promotes_inputs=["loads"] + promotes,
            promotes_outputs=["total_loads"],
        )

        self.add_subsystem(
            "create_rhs", CreateRHS(surface=surface), promotes_inputs=["total_loads"], promotes_outputs=["forces"]
        )


class SpatialBeamStates(om.Group):
    """Group that contains the spatial beam states."""

    def initialize(self):
        self.options.declare("surface", types=(dict, list))
        self.options.declare("strut_braced", default=False, types=bool)

    def setup(self):
        surface = self.options["surface"]

        if not self.options["strut_braced"]:
            self.add_subsystem('forces', FEMloads(surface=surface), promotes_inputs=["*"], promotes_outputs=["forces"])
            self.add_subsystem("fem", FEM(surface=surface), promotes_inputs=["*"], promotes_outputs=["*"])
            self.add_subsystem("disp", Disp(surface=surface), promotes_inputs=["*"], promotes_outputs=["*"])

        else:
            # compute RHS force vector for wing and strut, respectively
            for surf in surface:
                name = surf["name"]
                if name in ["strut", "jury"]:
                    # no fuel and point mass loads for struts
                    surf["distributed_fuel_weight"] = False
                    surf.pop("n_point_masses", None)

                # see if we have load_factor input
                if surf["struct_weight_relief"] or surf["distributed_fuel_weight"] or "n_point_masses" in surf.keys():
                    promotes_inputs = ["load_factor"]
                else:
                    promotes_inputs = []

                if name == "jury":
                    # jury is structure-only component so no external loads
                    ny = surf["mesh"].shape[1]
                    indep = self.add_subsystem("zero", om.IndepVarComp())
                    indep.add_output("jury_loads", val=np.zeros((ny, 6)))
                    self.connect("zero.jury_loads", f"forces_{name}.loads")

                self.add_subsystem(f'forces_{name}', FEMloads(surface=surf), promotes_inputs=promotes_inputs)
                self.connect(f"forces_{name}.forces", f"fem.forces_{name}")

            # FEM of wing-strut system
            self.add_subsystem("fem", FEMStrutBraced(surfaces=surface), promotes_inputs=["local_stiff_transformed_*"])
            
            # reshape displacements and remove Lagrange multipliers from disp_aug
            for surf in surface:
                name = surf["name"]
                self.add_subsystem(f"disp_{name}", Disp(surface=surf), promotes_outputs=[("disp", f"disp_{name}")])
                self.connect(f"fem.disp_aug_{name}", f"disp_{name}.disp_aug")
