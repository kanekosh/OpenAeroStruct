import openmdao.api as om

# from openaerostruct.structures.energy import Energy
# from openaerostruct.structures.weight import Weight
# from openaerostruct.structures.spar_within_wing import SparWithinWing
from openaerostruct.structures.vonmises_tube import VonMisesTube
from openaerostruct.structures.vonmises_wingbox import VonMisesWingbox
from openaerostruct.structures.non_intersecting_thickness import NonIntersectingThickness
from openaerostruct.structures.failure_exact import FailureExact
from openaerostruct.structures.failure_ks import FailureKS
from openaerostruct.structures.failure_buckling_ks import PanelLocalBucklingFailureKS, EulerColumnBucklingFailureKS


class SpatialBeamFunctionals(om.Group):
    """Group that contains the spatial beam functionals used to evaluate
    performance."""

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        surface = self.options["surface"]

        # Commented out energy for now since we haven't ever used its output
        # self.add_subsystem('energy',
        #          Energy(surface=surface),
        #          promotes=['*'])

        if surface["fem_model_type"] == "tube":
            self.add_subsystem(
                "thicknessconstraint",
                NonIntersectingThickness(surface=surface),
                promotes_inputs=["thickness", "radius"],
                promotes_outputs=["thickness_intersects"],
            )

            self.add_subsystem(
                "vonmises",
                VonMisesTube(surface=surface),
                promotes_inputs=["radius", "nodes", "disp"],
                promotes_outputs=["vonmises"],
            )
        elif surface["fem_model_type"] == "wingbox":
            self.add_subsystem(
                "vonmises",
                VonMisesWingbox(surface=surface),
                promotes_inputs=[
                    "Qz",
                    "J",
                    "A_enc",
                    "spar_thickness",
                    "htop",
                    "hbottom",
                    "hfront",
                    "hrear",
                    "nodes",
                    "disp",
                ],
                promotes_outputs=["vonmises"],
            )
        else:
            raise NameError("Please select a valid `fem_model_type` from either `tube` or `wingbox`.")

        # The following component has not been fully tested so we leave it
        # commented out for now. Use at your own risk.
        # self.add_subsystem('sparconstraint',
        #          SparWithinWing(surface=surface),
        #          promotes=['*'])

        if surface["exact_failure_constraint"]:
            self.add_subsystem(
                "failure", FailureExact(surface=surface), promotes_inputs=["vonmises"], promotes_outputs=["failure"]
            )
        else:
            self.add_subsystem(
                "failure", FailureKS(surface=surface), promotes_inputs=["vonmises"], promotes_outputs=["failure"]
            )

        # compute panel local buckling failure
        if "panel_buckling" in surface and surface["panel_buckling"]:
            # skin panel buckling and spar shear buckling
            self.add_subsystem(
                "local_buckling",
                PanelLocalBucklingFailureKS(surface=surface),
                promotes_inputs=["skin_thickness", "spar_thickness", "t_over_c", "fem_chords"],
                promotes_outputs=["failure_local_buckling"]
            )
            self.connect("vonmises.upper_skin_comp_stress", "local_buckling.upper_skin_comp_stress")
            self.connect("vonmises.lower_skin_comp_stress", "local_buckling.lower_skin_comp_stress")
            self.connect("vonmises.front_spar_shear_stress", "local_buckling.front_spar_shear_stress")
            self.connect("vonmises.rear_spar_shear_stress", "local_buckling.rear_spar_shear_stress")

        # global Euler column buckling
        if "column_buckling" in surface and surface["column_buckling"]:
            self.add_subsystem(
                "column_buckling",
                EulerColumnBucklingFailureKS(surface=surface),
                promotes_inputs=["nodes", "joint_load", "Iz"],
                promotes_outputs=["failure_column_buckling"]
            )
