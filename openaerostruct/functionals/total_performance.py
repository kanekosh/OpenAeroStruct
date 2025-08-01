import openmdao.api as om

from openaerostruct.functionals.breguet_range import BreguetRange
from openaerostruct.functionals.equilibrium import Equilibrium
from openaerostruct.functionals.center_of_gravity import CenterOfGravity
from openaerostruct.functionals.moment_coefficient import MomentCoefficient
from openaerostruct.functionals.total_lift_drag import TotalLiftDrag
from openaerostruct.functionals.sum_areas import SumAreas


class TotalPerformance(om.Group):
    """
    Group to contain the total aerostructural performance components.
    """

    def initialize(self):
        self.options.declare("surfaces", types=list)
        self.options.declare("user_specified_Sref", types=bool)
        self.options.declare("internally_connect_fuelburn", types=bool, default=True)
        self.options.declare("strut_braced", default=False, types=bool)

    def setup(self):
        surfaces = self.options["surfaces"]

        if self.options["strut_braced"]:
            if len(surfaces) == 2:
                # wing + strut. Make sure that the given surfaces are in order
                if surfaces[0]["name"] == "wing" and surfaces[1]["name"] == "strut":
                    surfaces_all = surfaces
                    surfaces_AS = surfaces
                    pass
                else:
                    raise ValueError("surfaces must be in order of [wing_surface, strut_surface, (optional) jury_surface]")
            elif len(surfaces) == 3:
                # wing + strut + jury
                if surfaces[0]["name"] == "wing" and surfaces[1]["name"] == "strut" and surfaces[2]["name"] == "jury":
                    surfaces_all = surfaces
                    surfaces_AS = [surfaces[0], surfaces[1]]   # exclude jury from VLM
                else:
                    raise ValueError("surfaces must be in order of [wing_surface, strut_surface, (optional) jury_surface]")
            else:
                raise ValueError("surfaces must be in order of [wing_surface, strut_surface, (optional) jury_surface]")

        if not self.options["user_specified_Sref"]:
            self.add_subsystem(
                "sum_areas", SumAreas(surfaces=surfaces_AS), promotes_inputs=["*S_ref"], promotes_outputs=["S_ref_total"]
            )

        if self.options["internally_connect_fuelburn"]:
            promote_fuelburn = ["fuelburn"]
        else:
            promote_fuelburn = []

        self.add_subsystem(
            "CL_CD",
            TotalLiftDrag(surfaces=surfaces_AS),
            promotes_inputs=["*CL", "*CD", "*S_ref", "S_ref_total", "rho", "v"],
            promotes_outputs=["CL", "CD", "L", "D"],
        )

        self.add_subsystem(
            "fuelburn",
            BreguetRange(surfaces=surfaces_all),
            promotes_inputs=["*structural_mass", "CL", "CD", "CT", "speed_of_sound", "R", "Mach_number", "W0"],
            promotes_outputs=["fuelburn"],
        )

        self.add_subsystem(
            "L_equals_W",
            Equilibrium(surfaces=surfaces_all),
            promotes_inputs=["CL", "*structural_mass", "S_ref_total", "W0", "load_factor", "rho", "v"]
            + promote_fuelburn,
            promotes_outputs=["L_equals_W", "total_weight"],
        )

        self.add_subsystem(
            "CG",
            CenterOfGravity(surfaces=surfaces_all),
            promotes_inputs=["*structural_mass", "*cg_location", "total_weight", "W0", "empty_cg", "load_factor"]
            + promote_fuelburn,
            promotes_outputs=["cg"],
        )

        self.add_subsystem(
            "moment",
            MomentCoefficient(surfaces=surfaces_AS),
            promotes_inputs=["v", "rho", "cg", "S_ref_total", "*b_pts", "*widths", "*chords", "*sec_forces", "*S_ref"],
            promotes_outputs=["CM"],
        )
