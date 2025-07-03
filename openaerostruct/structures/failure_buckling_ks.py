import numpy as np
import openmdao.api as om


class FailureBucklingKS(om.ExplicitComponent):
    """
    Buckling stress constraints for skin compression and spar shear buckling.
    Returns KS-aggregated failure metric that should be constrained to be <= 0.

    Source: Michael Niu, Airframe Stress Analysis and Sizing, Hong Kong Conmilit Press Ltd., 1997

    The rho inputeter controls how conservatively the KS function aggregates
    the failure constraints. A lower value is more conservative while a greater
    value is more aggressive (closer approximation to the max() function).

    parameters
    ----------
    upper_skin_comp_stress[ny-1] : numpy array
        Compression stress in the upper skin for each FEM element.
    lower_skin_comp_stress[ny-1] : numpy array
        Compression stress in the lower skin for each FEM element.
    front_spar_shear_stress[ny-1] : numpy array
        Shear stress in the front spar for each FEM element.
    rear_spar_shear_stress[ny-1] : numpy array
        Shear stress in the rear spar for each FEM element.

    Returns
    -------
    failure_buckling : float
        KS aggregation quantity by combining the failure constraints from upper skin buckling,
        front spar shear buckling, and rear spar shear buckling for all FEM elements.
    upper_skin_buckling_margin[ny-1] : numpy array
        Margin for upper skin buckling for each FEM element, should be >= 0
        Margin = (buckling_limit - stress) / buckling_limit
    front_spar_buckling_margin[ny-1] : numpy array
        Margin for front spar shear buckling for each FEM element, should be >= 0
    rear_spar_buckling_margin[ny-1] : numpy array
        Margin for rear spar shear buckling for each FEM element, should be >= 0

    """

    def initialize(self):
        self.options.declare("surface", types=dict)
        self.options.declare("rho", types=float, default=100.0)

    def setup(self):
        surface = self.options["surface"]
        rho = self.options["rho"]

        if surface["fem_model_type"] == "tube":
            raise NotImplementedError("FailureBucklingKS is not implemented for tube FEM model type")

        self.ny = surface["mesh"].shape[1]

        # stress of each element
        self.add_input("upper_skin_comp_stress", val=np.zeros(self.ny - 1), units="N/m**2")
        self.add_input("lower_skin_comp_stress", val=np.zeros(self.ny - 1), units="N/m**2")
        self.add_input("front_spar_shear_stress", val=np.zeros(self.ny - 1), units="N/m**2")
        self.add_input("rear_spar_shear_stress", val=np.zeros(self.ny - 1), units="N/m**2")
        # wingbox geometry
        self.add_input("skin_thickness", val=np.zeros(self.ny - 1), units="m")
        self.add_input("spar_thickness", val=np.zeros(self.ny - 1), units="m")
        self.add_input("t_over_c", val=np.ones((self.ny - 1)))
        self.add_input("fem_chords", val=np.zeros(self.ny - 1), units="m")

        self.add_output("failure_buckling", val=0.0)
        # also output margins for each element/buckling for post-processing
        # but don't declare partials for these outputs because we only use aggregated failure
        self.add_output("upper_skin_buckling_margin", val=np.zeros(self.ny - 1), desc='(buckling_limit - stress) / buckling_limit >= 0?')
        self.add_output("lower_skin_buckling_margin", val=np.zeros(self.ny - 1), desc='(buckling_limit - stress) / buckling_limit >= 0?')
        self.add_output("front_spar_buckling_margin", val=np.zeros(self.ny - 1), desc='(buckling_limit - stress) / buckling_limit >= 0?')
        self.add_output("rear_spar_buckling_margin", val=np.zeros(self.ny - 1), desc='(buckling_limit - stress) / buckling_limit >= 0?')

        self.rho = rho

        self.declare_partials("failure_buckling", "*", method="cs")

    def compute(self, inputs, outputs):
        surface = self.options["surface"]
        E = self.options["surface"]["E"]
        rib_pitch = self.options["surface"]["rib_pitch"]
        rho = self.options["rho"]  # KS aggregation parameter

        safety_factor = 1.5

        # --- compute buckling stress ---
        # buckling model parameters
        eff_factor = 0.88  # stiffened panel efficiency for skin compression buckling. From Fig. 14.4.4 in Niu 1997
        Ks = 10.0  # model parameter for spar shear buckling. From Eq. 11.3.4 in Niu 1997

        # buckling stress for skin compression. Eq. 14.4.2, Page 618 from Niu 1997
        # Positive stress = compression.
        load_intensity_upper = inputs["skin_thickness"] * inputs["upper_skin_comp_stress"]  # Pa*m
        load_positive_upper = np.maximum(load_intensity_upper, 1.)  # NOTE: this is C1 discontinuous so may negatively affect convergence
        sigma_max_upper = (eff_factor * (load_positive_upper * E / rib_pitch)**0.5) / safety_factor
        load_intensity_lower = inputs["skin_thickness"] * inputs["lower_skin_comp_stress"]
        load_positive_lower = np.maximum(load_intensity_lower, 1.)
        sigma_max_lower = (eff_factor * (load_positive_lower * E / rib_pitch)**0.5) / safety_factor

        # shear buckling stress of spars. Eq. 14.4.3, Page 618 from Niu 1997
        t_over_c_orig = surface["original_wingbox_airfoil_t_over_c"]
        spar_height_front_nondim = (surface["data_y_upper"][0] - surface["data_y_lower"][0])  # nondim height from wingbox cross-section
        spar_height_rear_nondim = (surface["data_y_upper"][-1] - surface["data_y_lower"][-1])
        spar_height_front = spar_height_front_nondim * inputs["fem_chords"] * (inputs["t_over_c"] / t_over_c_orig)
        spar_height_rear = spar_height_rear_nondim * inputs["fem_chords"] * (inputs["t_over_c"] / t_over_c_orig)
        tau_max_front = (Ks * E * (inputs["spar_thickness"] / spar_height_front)**2) / safety_factor
        tau_max_rear = (Ks * E * (inputs["spar_thickness"] / spar_height_rear)**2) / safety_factor

        # --- compute failure and stress margins ---
        outputs["upper_skin_buckling_margin"] = (sigma_max_upper - inputs["upper_skin_comp_stress"]) / sigma_max_upper
        outputs["lower_skin_buckling_margin"] = (sigma_max_lower - inputs["lower_skin_comp_stress"]) / sigma_max_lower
        # apply 40% to shear stress based on Elham 2014, Effect of wing-box structure on the optimum wing outer shape, The Aeronautical Journal
        outputs["front_spar_buckling_margin"] = (tau_max_front - inputs["front_spar_shear_stress"] * 0.4) / tau_max_front
        outputs["rear_spar_buckling_margin"] = (tau_max_rear - inputs["rear_spar_shear_stress"] * 0.4) / tau_max_rear

        # Failure criteria (must be <= 0)
        failure_all = np.concatenate(
            [
                inputs["upper_skin_comp_stress"] / sigma_max_upper - 1,
                inputs["lower_skin_comp_stress"] / sigma_max_lower - 1,
                inputs["front_spar_shear_stress"] * 0.4 / tau_max_front - 1,
                inputs["rear_spar_shear_stress"] * 0.4 / tau_max_rear - 1,
            ]
        )

        fmax = np.max(failure_all)

        nlog, nsum, nexp = np.log, np.sum, np.exp
        ks = 1 / rho * nlog(nsum(nexp(rho * (failure_all - fmax))))
        outputs["failure_buckling"] = fmax + ks
