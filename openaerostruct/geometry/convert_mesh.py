from __future__ import division, print_function
import numpy as np
import openmdao.api as om

class ConvertMesh(om.ExplicitComponent):
    """
    Converts the mesh array from the original 3D form to the 4D form that accomodates multi-valued vertices.

    Parameters
    ----------
    mesh_orig [nx, ny, 3] : numpy array
        Mesh in the origial form that is generatedmanipulated by geometry components.

    Returns
    -------
    mesh [nx-1, ny-1, 4, 3] : numpy array
        Mesh in the multiple-valued format. Mesh vertices would have multiple values of
        locations in aerostructural cases. This is passed to the aero or AS point group.
        First two indices denote cells, the third index for the four vertices of the cell
        (in the order of 11, 12, 21, 22), and the last one for xyz locations. 
    """

    def initialize(self):
        self.options.declare('mesh_shape', desc='Tuple containing mesh shape (nx, ny, 3).')

    def setup(self):
        mesh_shape = self.options['mesh_shape']  # original mesh shape
        self.add_input('mesh_orig', shape=mesh_shape, units='m')

        mesh_shape_out = (mesh_shape[0]-1, mesh_shape[1]-1, 4, 3)
        self.add_output('mesh', shape=mesh_shape_out, units='m')

        self.declare_partials(of='*', wrt='*', method='fd')
        
    def compute(self, inputs, outputs):
        mesh_orig = inputs['mesh_orig']
        mesh_shape = self.options['mesh_shape'] 

        mesh_shape_out = (mesh_shape[0]-1, mesh_shape[1]-1, 4, 3)
        mesh_out = np.zeros(mesh_shape_out)

        # vertices 11
        mesh_out[:, :, 0, :] = mesh_orig[:-1, :-1, :]
        # vertices 12
        mesh_out[:, :, 1, :] = mesh_orig[1:, :-1, :]
        # vertices 21
        mesh_out[:, :, 2, :] = mesh_orig[:-1, 1:, :]
        # vertices 22
        mesh_out[:, :, 3, :] = mesh_orig[1:, 1:, :]
        
        outputs['mesh'] = mesh_out


if __name__ == '__main__':
    # TODO: add this as an unit test
    nx = 3
    ny = 5

    mesh_orig = np.zeros((nx, ny, 3))
    for i in range(ny):
        mesh_orig[:, i, 0] = np.arange(0, nx)
    for i in range(nx):
        mesh_orig[i, :, 1] = np.arange(0, ny)
    
    print('mesh_x', mesh_orig[:, :, 0])
    print('mesh_y', mesh_orig[:, :, 1])

    prob = om.Problem()
    indep = prob.model.add_subsystem('indep', om.ExplicitComponent(), promotes_outputs=['*'])
    indep.add_output('mesh_orig', val=mesh_orig)

    mesh_shape = (nx, ny, 3)
    prob.model.add_subsystem('convertmesh', ConvertMesh(mesh_shape=mesh_shape), promotes=['*'])

    prob.setup(check=True)

    prob.run_model()

    mesh_out = prob['mesh']
    print('converted mesh at cell 0. 0', mesh_out[0, 0, :, :] )
    print('converted mesh at cell -1, 0', mesh_out[-1, 0, :, :] )
    print('converted mesh at cell 0, -1', mesh_out[0, -1, :, :] )
    print('converted mesh at cell -1, -1', mesh_out[-1, -1, :, :] )
        
