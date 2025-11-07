import numpy as np

import openmdao.api as om


class BreguetRange(om.ExplicitComponent):
    """
    Computes the fuel burn using the Breguet range equation using
    the computed CL, CD, weight, and provided specific fuel consumption, speed of sound,
    Mach number, initial weight, and range.

    Note that we add information from each lifting surface.

    Parameters
    ----------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    CT : float
        Specific fuel consumption for the entire aircraft.
    speed_of_sound : float
        The Mach speed, speed of sound, at the specified flight condition.
    R : float
        The total range of the aircraft, used to backcalculate the fuel mass.
    Mach_number : float
        The Mach number of the aircraft at the specified flight condition.
    W0 : float
        The operating empty weight (OEW) - wing structural mass + payload + reserve fuel
        I.e. this is the landing gross mass, ignoring the descent and landing fuel burn.
        Supplied in kg despite being a 'weight' due to convention.
    _structural_mass : float
        Weight of a single lifting surface's structural spar.

    Returns
    -------
    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.

    """

    def initialize(self):
        self.options.declare("surfaces", types=list)

    def setup(self):
        for surface in self.options["surfaces"]:
            name = surface["name"]
            self.add_input(name + "_structural_mass", val=1.0, units="kg")

        self.add_input("CT", val=0.25, units="1/s")
        self.add_input("CL", val=0.7)
        self.add_input("CD", val=0.02)
        self.add_input("speed_of_sound", val=100.0, units="m/s")
        self.add_input("R", val=3000.0, units="m")
        self.add_input("Mach_number", val=1.2)
        self.add_input("W0", val=200.0, units="kg", desc="payload + OEW - surface str. mass + reserve fuel")

        self.add_output("fuelburn", val=1.0, units="kg")
        self.add_output("W_start", val=1.0, units="kg", desc="Mass at start of this phase")

        self.declare_partials("*", "*")
        self.set_check_partial_options(wrt="*", method="cs", step=1e-30)

    def compute(self, inputs, outputs):
        CT = inputs["CT"]
        a = inputs["speed_of_sound"]
        R = inputs["R"]
        M = inputs["Mach_number"]
        W0 = inputs["W0"]

        # Loop through the surfaces and add up the structural weights
        # to get the total structural weight.
        Ws = 0.0
        for surface in self.options["surfaces"]:
            name = surface["name"]
            Ws += inputs[name + "_structural_mass"]

        CL = inputs["CL"]
        CD = inputs["CD"]

        outputs["W_start"] = (W0 + Ws) * (np.exp(R * CT / a / M * CD / CL))
        outputs["fuelburn"] = outputs["W_start"] - (W0 + Ws)

    def compute_partials(self, inputs, partials):
        CT = inputs["CT"]
        a = inputs["speed_of_sound"]
        R = inputs["R"]
        M = inputs["Mach_number"]
        W0 = inputs["W0"]

        Ws = 0.0
        for surface in self.options["surfaces"]:
            name = surface["name"]
            Ws += inputs[name + "_structural_mass"]

        CL = inputs["CL"]
        CD = inputs["CD"]

        dW_start_dCL = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a / M * CD / CL**2
        dW_start_dCD = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a / M / CL
        dW_start_dCT = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R / a / M / CL * CD
        dW_start_dR = (W0 + Ws) * np.exp(R * CT / a / M * CD / CL) / a / M / CL * CD * CT
        dW_start_da = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a**2 / M * CD / CL
        dW_start_dM = -(W0 + Ws) * np.exp(R * CT / a / M * CD / CL) * R * CT / a / M**2 * CD / CL

        dW_start_dW = np.exp(R * CT / a / M * CD / CL)

        partials["W_start", "CL"] = dW_start_dCL
        partials["W_start", "CD"] = dW_start_dCD
        partials["W_start", "CT"] = dW_start_dCT
        partials["W_start", "speed_of_sound"] = dW_start_da
        partials["W_start", "R"] = dW_start_dR
        partials["W_start", "Mach_number"] = dW_start_dM
        partials["W_start", "W0"] = dW_start_dW

        partials["fuelburn", "CL"] = dW_start_dCL
        partials["fuelburn", "CD"] = dW_start_dCD
        partials["fuelburn", "CT"] = dW_start_dCT
        partials["fuelburn", "speed_of_sound"] = dW_start_da
        partials["fuelburn", "R"] = dW_start_dR
        partials["fuelburn", "Mach_number"] = dW_start_dM
        partials["fuelburn", "W0"] = dW_start_dW - 1

        for surface in self.options["surfaces"]:
            name = surface["name"]
            inp_name = name + "_structural_mass"
            partials["W_start", inp_name] = dW_start_dW
            partials["fuelburn", inp_name] = dW_start_dW - 1


class BreguetRangeClimbAndCruise(om.Group):
    """
    Breguet range equation for climb and cruise.
    Computes the fuel burn using the Breguet range equation using
    the computed CL, CD, weight, and provided specific fuel consumption, speed of sound,
    Mach number, initial weight, and range.

    Note that we add information from each lifting surface.

    Parameters
    ----------
    CL : float
        Total coefficient of lift (CL) for the lifting surface.
    CD : float
        Total coefficient of drag (CD) for the lifting surface.
    CT : float
        Specific fuel consumption for the entire aircraft.
    speed_of_sound : float
        The speed of sound at cruise.
    speed_of_sound_climb : float
        The speed of sound at climb.
    R : float
        Cruise range, used to backcalculate the cruise fuel burn.
    R_climb : float
        Climb range, used to backcalculate the climb fuel burn.
    Mach_number : float
        Cruise Mach number.
    Mach_number_climb : float
        Climb Mach number.
    gamma_climb : float
        Climb flight path angle
    The operating empty weight (OEW) - wing structural mass + payload + reserve fuel
        I.e. this is the landing gross mass, ignoring the descent and landing fuel burn.
        Supplied in kg despite being a 'weight' due to convention.
    _structural_mass : float
        Weight of a single lifting surface's structural spar.

    Returns
    -------
    fuelburn : float
        Computed fuel burn in kg based on the Breguet range equation.

    """

    def initialize(self):
        self.options.declare("surfaces", types=list)

    def setup(self):
        # cruise fuel burn
        self.add_subsystem("cruise", BreguetRange(surfaces=self.options["surfaces"]), promotes_inputs=["*"])

        # climb fuel burn.
        # First compute equivalent "CD": We replace (D/L) -> cos(gamma) / (L/D) + sin(gamma) = (D cos(gamma) + L sin(gamma) / L
        self.add_subsystem(
            "climb_CD_equiv",
            om.ExecComp(
                "CD_equiv = (CD * cos(gamma_climb) + CL * sin(gamma_climb))",
                CD_equiv={'units': None},
                CD={'units': None},
                CL={'units': None},
                gamma_climb={'units': 'rad'},
            ),
            promotes_inputs=["gamma_climb", "CD", "CL"],
        )
        # By not supplying `surface` and connecting start-of-cruise mass to W0, we can reuse the Breguet component for climb.
        self.add_subsystem(
            "climb",
            BreguetRange(surfaces=[]),
            promotes_inputs=[
                "CT",  # this sets same TSFC as cruise
                ("speed_of_sound", "speed_of_sound_climb"),
                ("R", "R_climb"),
                ("Mach_number", "Mach_number_climb"),
                "CL",
            ],
        )
        self.connect("cruise.W_start", "climb.W0")
        self.connect("climb_CD_equiv.CD_equiv", "climb.CD")

        # sum fuel burn
        self.add_subsystem("total_fuelburn", om.AddSubtractComp(output_name="fuelburn", input_names=["cruise_fb", "climb_fb"], units="kg"), promotes_outputs=["*"])
        self.connect("cruise.fuelburn", "total_fuelburn.cruise_fb")
        self.connect("climb.fuelburn", "total_fuelburn.climb_fb")
