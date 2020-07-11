import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.utils.testing import run_test, get_default_surfaces
from openaerostruct.geometry.convert_mesh import ConvertMesh

class Test(unittest.TestCase):

    """ !!!commenting out all tests including partials!!!
    def test(self):
        surfaces = get_default_surfaces()

        group = om.Group()

        comp = VLMGeometry(surface=surfaces[0])

        indep_var_comp = om.IndepVarComp()

        indep_var_comp.add_output('def_mesh_orig', val=surfaces[0]['mesh'], units='m')

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('convertmesh', ConvertMesh(mesh_shape=surfaces[0]['mesh'].shape))
        group.add_subsystem('geom', comp, promotes=['*'])
        group.connect('def_mesh_orig', 'convertmesh.mesh_orig')
        group.connect('convertmesh.mesh', 'def_mesh')

        run_test(self, group)

    def test_derivs_wetted(self):
        # This is a much richer test with the following attributes:
        # - 5x7 mesh so that each dimension of all variables is unique.
        # - Random values given as inputs.
        # - The mesh has been given a random z height at all points.
        # - S_ref_type option is "wetted"
        # Create a dictionary to store options about the mesh
        mesh_dict = {'num_y' : 7,
                     'num_x' : 5,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        # Generate the aerodynamic mesh based on the previous dictionary
        mesh, twist_cp = generate_mesh(mesh_dict)

        mesh[:, :, 2] = np.random.random(mesh[:, :, 2].shape)

        # Create a dictionary with info and options about the aerodynamic
        # lifting surface
        surface = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'S_ref_type' : 'wetted', # how we compute the wing area,
                                             # can be 'wetted' or 'projected'
                    'fem_model_type' : 'tube',

                    'twist_cp' : twist_cp,
                    'mesh' : mesh,

                    # Aerodynamic performance of the lifting surface at
                    # an angle of attack of 0 (alpha=0).
                    # These CL0 and CD0 values are added to the CL and CD
                    # obtained from aerodynamic analysis of the surface to get
                    # the total CL and CD.
                    # These CL0 and CD0 values do not vary wrt alpha.
                    'CL0' : 0.0,            # CL of the surface at alpha=0
                    'CD0' : 0.015,            # CD of the surface at alpha=0

                    # Airfoil properties for viscous drag calculation
                    'k_lam' : 0.05,         # percentage of chord with laminar
                                            # flow, used for viscous drag
                    't_over_c_cp' : np.array([0.15]),      # thickness over chord ratio (NACA0015)
                    'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                            # thickness
                    'with_viscous' : True,  # if true, compute viscous drag
                    'with_wave' : False,     # if true, compute wave drag
                    }

        #surfaces = get_default_surfaces()
        surfaces = [surface]

        prob = om.Problem()
        group = prob.model

        comp = VLMGeometry(surface=surfaces[0])

        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output('def_mesh_orig', val=surfaces[0]['mesh'], units='m')

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('convertmesh', ConvertMesh(mesh_shape=surfaces[0]['mesh'].shape))
        group.add_subsystem('geom', comp, promotes=['*'])
        group.connect('def_mesh_orig', 'convertmesh.mesh_orig')
        group.connect('convertmesh.mesh', 'def_mesh')

        prob.setup()

        prob['geom.def_mesh'] = np.random.random(prob['geom.def_mesh'].shape)

        prob.run_model()

        check = prob.check_partials(compact_print=True)
        assert_check_partials(check, atol=3e-5, rtol=1e-5)

    def test_derivs_projected(self):
        # This is a much richer test with the following attributes:
        # - 5x7 mesh so that each dimension of all variables is unique.
        # - Random values given as inputs.
        # - The mesh has been given a random z height at all points.
        # - S_ref_type option is "projected"

        # Create a dictionary to store options about the mesh
        mesh_dict = {'num_y' : 7,
                     'num_x' : 5,
                     'wing_type' : 'CRM',
                     'symmetry' : True,
                     'num_twist_cp' : 5}

        # Generate the aerodynamic mesh based on the previous dictionary
        mesh, twist_cp = generate_mesh(mesh_dict)

        mesh[:, :, 2] = np.random.random(mesh[:, :, 2].shape)

        # Create a dictionary with info and options about the aerodynamic
        # lifting surface
        surface = {
                    # Wing definition
                    'name' : 'wing',        # name of the surface
                    'symmetry' : True,     # if true, model one half of wing
                                            # reflected across the plane y = 0
                    'S_ref_type' : 'projected', # how we compute the wing area,
                                             # can be 'wetted' or 'projected'
                    'fem_model_type' : 'tube',

                    'twist_cp' : twist_cp,
                    'mesh' : mesh,

                    # Aerodynamic performance of the lifting surface at
                    # an angle of attack of 0 (alpha=0).
                    # These CL0 and CD0 values are added to the CL and CD
                    # obtained from aerodynamic analysis of the surface to get
                    # the total CL and CD.
                    # These CL0 and CD0 values do not vary wrt alpha.
                    'CL0' : 0.0,            # CL of the surface at alpha=0
                    'CD0' : 0.015,            # CD of the surface at alpha=0

                    # Airfoil properties for viscous drag calculation
                    'k_lam' : 0.05,         # percentage of chord with laminar
                                            # flow, used for viscous drag
                    't_over_c_cp' : np.array([0.15]),      # thickness over chord ratio (NACA0015)
                    'c_max_t' : .303,       # chordwise location of maximum (NACA0015)
                                            # thickness
                    'with_viscous' : True,  # if true, compute viscous drag
                    'with_wave' : False,     # if true, compute wave drag
                    }

        #surfaces = get_default_surfaces()
        surfaces = [surface]

        prob = om.Problem()
        group = prob.model

        comp = VLMGeometry(surface=surfaces[0])

        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output('def_mesh_orig', val=surfaces[0]['mesh'], units='m')

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('convertmesh', ConvertMesh(mesh_shape=surfaces[0]['mesh'].shape))
        group.add_subsystem('geom', comp, promotes=['*'])
        group.connect('def_mesh_orig', 'convertmesh.mesh_orig')
        group.connect('convertmesh.mesh', 'def_mesh')

        prob.setup()

        prob['geom.def_mesh'] = np.random.random(prob['geom.def_mesh'].shape)

        prob.run_model()

        check = prob.check_partials(compact_print=True)

        assert_check_partials(check, atol=3e-5, rtol=1e-5)
    """

    def test_outputs(self):
        surfaces = get_default_surfaces()

        group = om.Group()

        comp = VLMGeometry(surface=surfaces[0])

        indep_var_comp = om.IndepVarComp()

        indep_var_comp.add_output('def_mesh_orig', val=surfaces[0]['mesh'], units='m')

        group.add_subsystem('indep_var_comp', indep_var_comp, promotes=['*'])
        group.add_subsystem('convertmesh', ConvertMesh(mesh_shape=surfaces[0]['mesh'].shape))
        group.add_subsystem('geom', comp, promotes=['*'])
        group.connect('def_mesh_orig', 'convertmesh.mesh_orig')
        group.connect('convertmesh.mesh', 'def_mesh')

        prob = om.Problem()
        prob.model.add_subsystem('group', group, promotes=['*'])
        prob.setup()
        # om.n2(prob)
        prob.run_model()

        assert_rel_error(self, prob['widths'] , np.array([11.95624787, 11.90425878, 11.44086572]), 1e-6)
        assert_rel_error(self, prob['cos_sweep'] , np.array([9.7938336, 9.79384207, 9.79385053]), 1e-6)
        assert_rel_error(self, prob['S_ref'] , np.array([415.02211208]), 1e-6)
        # assert_rel_error(self, prob['chords'] , np.array([ 2.72796,    5.1252628,  7.8891638, 13.6189974]), 1e-6)
        # assert_rel_error(self, prob['lengths'] , np.array([ 2.72796,    5.1252628,  7.8891638, 13.6189974]), 1e-6)
        chords_lengths = np.array([[2.72796, 5.1252628], [5.1252628, 7.8891638], [7.8891638, 13.6189974]])
        assert_rel_error(self, prob['chords'] , chords_lengths, 1e-6)
        assert_rel_error(self, prob['lengths'] , chords_lengths, 1e-6)

        normals = np.array([[[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]]])
        assert_rel_error(self, prob['normals'] , normals, 1e-6)
        
        b_pts = np.zeros((1, 3, 2, 3))
        b_pts[:, 0, 0, :] = np.array([45.9127098, -29.3815262, 0.])
        b_pts[:, 0, 1, :] = np.array([39.0546717, -19.5876926, 0.])
        b_pts[:, 1, 0, :] = np.array([39.0546717, -19.5876926, 0.])
        b_pts[:, 1, 1, :] = np.array([32.28769048, -9.79385053, 0.])
        b_pts[:, 2, 0, :] = np.array([32.28769048, -9.79385053, 0.])
        b_pts[:, 2, 1, :] = np.array([26.3738169, 0, 0])
        assert_rel_error(self, prob['b_pts'] , b_pts, 1e-6)

if __name__ == '__main__':
    unittest.main()
