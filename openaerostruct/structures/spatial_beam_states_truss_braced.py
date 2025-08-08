import numpy as np
import openmdao.api as om
from openaerostruct.structures.create_rhs import CreateRHS
from openaerostruct.structures.fem import FEM
from openaerostruct.structures.fem_strut_braced import FEMStrutBraced, get_joint_idx_from_y
from openaerostruct.structures.disp import Disp
from openaerostruct.structures.wing_weight_loads import StructureWeightLoads
from openaerostruct.structures.fuel_loads import FuelLoads
from openaerostruct.structures.total_loads import TotalLoads
from openaerostruct.structures.compute_point_mass_loads import ComputePointMassLoads
from openaerostruct.structures.compute_thrust_loads import ComputeThrustLoads
from openaerostruct.structures.spatial_beam_states import FEMloads


class SpatialBeamStatesTrussBraced(om.Group):
    """
    Group that contains the spatial beam states for truss-braced wing model
    that combines FEM for wing and main strut + analytic rod for jury strut
    """

    def initialize(self):
        self.options.declare("surfaces", types=list)
        self.options.declare("strut_braced", default=True, types=bool)

    def setup(self):
        surfaces = self.options["surfaces"]

        if len(surfaces) != 3:
            raise ValueError("surface must have 3 surfaces: wing, strut, and jury")

        surf_wing = surfaces[0]
        surf_strut = surfaces[1]
        surf_jury = surfaces[2]

        # no fuel and point mass loads for struts and jury
        surf_strut["distributed_fuel_weight"] = False
        surf_strut.pop("n_point_masses", None)
        surf_jury["distributed_fuel_weight"] = False
        surf_jury.pop("n_point_masses", None)

        # find spanwise index of jury-wing and jury-strut joints
        jury_joint_idx = {
            "wing": get_joint_idx_from_y(surf_wing, surf_jury["wing_jury_joint_y"]),
            "strut": get_joint_idx_from_y(surf_strut, surf_jury["strut_jury_joint_y"]),
        }

        # --- Strut-braced wing FEM ---
        # compute RHS force vector for wing and strut
        for surf in [surf_wing, surf_strut]:
            name = surf["name"]

            # compute jury joint loads
            self.add_subsystem(f"external_loads_{name}", JuryJointLoads(surface=surf, joint_idx=jury_joint_idx[name]), promotes_inputs=["jury_nodes", "joint_axias_force"])

            # see if we have load_factor input
            if surf["struct_weight_relief"] or surf["distributed_fuel_weight"] or "n_point_masses" in surf.keys():
                promotes_inputs = ["load_factor"]
            else:
                promotes_inputs = []

            # RHS vector
            self.add_subsystem(f'forces_{name}', FEMloads(surface=surf), promotes_inputs=promotes_inputs)
            self.connect(f"external_loads_{name}.loads_out", f"forces_{name}.loads")
            self.connect(f"forces_{name}.forces", f"fem.forces_{name}")
    
        # FEM of wing-strut system
        self.add_subsystem("fem", FEMStrutBraced(surfaces=[surf_wing, surf_strut]), promotes_inputs=["local_stiff_transformed_*"])
        
        # reshape displacements and remove Lagrange multipliers from disp_aug
        for surf in [surf_wing, surf_strut]:
            name = surf["name"]
            self.add_subsystem(f"disp_{name}", Disp(surface=surf), promotes_outputs=[("disp", f"disp_{name}")])
            self.connect(f"fem.disp_aug_{name}", f"disp_{name}.disp_aug")

        # --- compute jury axial displacement from strut-braced wing FEM solution ---
        self.add_subsystem("jury_disp_from_SBW", JuryDispFromSBW(), promotes_inputs=["jury_nodes"], promotes_outputs=["disp_jury"])
        self.connect("disp_wing", "jury_disp_from_SBW.disp_wing", src_indices=om.slicer[jury_joint_idx["wing"], :3])
        self.connect("disp_strut", "jury_disp_from_SBW.disp_strut", src_indices=om.slicer[jury_joint_idx["strut"], :3])

        # --- Compute jury axial load from displacement ---
        self.add_subsystem("jury_load", JuryStrutRod(surface=surf_jury), promotes_inputs=["jury_nodes", "jury_A", "disp_jury"], promotes_outputs=["joint_axias_force"])

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=20, atol=1e-6, rtol=1e-6, err_on_non_converge=True, iprint=0)
        self.linear_solver = om.DirectSolver()

        
class JuryJointLoads(om.ExplicitComponent):
    """
    Compute jury joint loads
    """

    def initialize(self):
        self.options.declare("surface", types=dict)
        self.options.declare("joint_idx", desc="spanwise index of jury joint on this surface")

    def setup(self):
        surface = self.options["surface"]
        ny = surface["mesh"].shape[1]

        self.add_input("jury_nodes", val=np.zeros((2, 3)), units='m')
        self.add_input("joint_axias_force", val=0., units="N", desc="axial force on jury joint. Positive = tension")
        self.add_input("loads", val=np.ones((ny, 6)), units="N")
        self.add_output("loads_out", val=np.ones((ny, 6)), units="N")

        self.declare_partials("loads_out", ["jury_nodes", "joint_axias_force"], method="cs", step=1e-100)
        self.declare_partials("loads_out", "loads", rows=np.arange(ny * 6), cols=np.arange(ny * 6), val=np.ones(ny * 6))

    def compute(self, inputs, outputs):
        surface = self.options["surface"]
        joint_idx = self.options["joint_idx"]

        jury_nodes = inputs["jury_nodes"]

        # direction of jury struts = axial force. Jury node is defined from wing to node
        if surface["name"] == "wing":
            load_dir = jury_nodes[-1, :] - jury_nodes[0, :]
        elif surface["name"] == "strut":
            load_dir = jury_nodes[0, :] - jury_nodes[-1, :]
        else:
            raise ValueError(f"Jury joint loads are only supported for wing and strut surfaces, not {surface['name']}")
        load_dir = load_dir / np.linalg.norm(load_dir)

        # compute load vector and add to the loads
        load_vec = inputs["joint_axias_force"] * load_dir
        outputs["loads_out"] = inputs["loads"] * 1.0
        outputs["loads_out"][joint_idx, :3] += load_vec


class JuryStrutRod(om.ExplicitComponent):
    """
    Analytic rod model for jury strut
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        self.add_input("jury_nodes", val=np.zeros((2, 3)), units='m')
        self.add_input("jury_A", val=1., units="m**2", desc="cross-sectional area of jury strut")
        self.add_input("disp_jury", val=0., units='m', desc="axial displacement of jury strut (tension = positive)")

        self.add_output("joint_axias_force", val=0., units="N", desc="axial force on jury joint. Positive = tension")

        self.declare_partials("*", "*", method="cs", step=1e-100)

    def compute(self, inputs, outputs):
        surface = self.options["surface"]
        E = surface["E"]

        A = inputs["jury_A"]
        jury_nodes = inputs["jury_nodes"]
        disp = inputs["disp_jury"]

        jury_len = np.linalg.norm(jury_nodes[-1, :] - jury_nodes[0, :])
        outputs["joint_axias_force"] = E * A * disp / jury_len


class JuryDispFromSBW(om.ExplicitComponent):
    """
    Compute jury strut displacement from the strut-braced wing FEM displacements
    """
    def setup(self):
        self.add_input("disp_wing", shape=(3,), units='m')
        self.add_input("disp_strut", shape=(3,), units='m')
        self.add_input("jury_nodes", shape=(2, 3), units='m')
        
        self.add_output("disp_jury", val=0., units='m', desc="axial displacement of jury strut (tension = positive)")

        self.declare_partials("*", "*", method="cs", step=1e-100)

    def compute(self, inputs, outputs):
        jury_nodes = inputs["jury_nodes"]
        
        # jury strut direction
        jury_dir = jury_nodes[-1, :] - jury_nodes[0, :]
        jury_dir = jury_dir / np.linalg.norm(jury_dir)

        # compute jury strut tensile displacement
        disp_wing_mapped = np.dot(inputs["disp_wing"], jury_dir)
        disp_strut_mapped = np.dot(inputs["disp_strut"], jury_dir)
        outputs["disp_jury"] = disp_strut_mapped - disp_wing_mapped