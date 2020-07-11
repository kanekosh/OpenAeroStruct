from __future__ import print_function
import numpy as np

import openmdao.api as om

class VortexMesh(om.ExplicitComponent):
    """
    Compute the vortex mesh based on the deformed aerodynamic mesh.

    Parameters
    ----------
    def_mesh[nx-1, ny-1, 4, 3] : numpy array
        We have a mesh for each lifting surface in the problem.
        That is, if we have both a wing and a tail surface, we will have both
        `wing_def_mesh` and `tail_def_mesh` as inputs.

    Returns
    -------
    vortex_mesh[nx-1, ny-1, 4, 3] : numpy array
        The actual aerodynamic mesh used in VLM calculations, where we look
        at the rings of the panels instead of the panels themselves. That is,
        this mesh coincides with the quarter-chord panel line, except for the
        final row, where it lines up with the trailing edge.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        # Because the vortex_mesh always comes from the deformed mesh in the
        # same way, the Jacobian is fully linear and can be set here instead
        # of doing compute_partials.
        # We do have to account for symmetry here to create a ghost mesh
        # by mirroring the symmetric mesh.
        for surface in surfaces:
            mesh=surface['mesh']  # this is the original 3D mesh
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            name = surface['name']

            mesh_name = '{}_def_mesh'.format(name)
            vortex_mesh_name = '{}_vortex_mesh'.format(name)

            self.add_input(mesh_name, shape=(nx-1, ny-1, 4, 3), units='m')

            if surface['symmetry']:
                ny_full = ny*2-1
                self.add_output(vortex_mesh_name, shape=(nx-1, ny_full-1, 4, 3), units='m')

                """
                mesh_indices = np.arange(nx * ny * 3).reshape((nx, ny, 3))
                vor_indices = np.arange(nx * (2*ny-1) * 3).reshape((nx, (2*ny-1), 3))

                rows = np.tile(vor_indices[:(nx-1), :ny, :].flatten(), 2)
                rows = np.hstack((rows, vor_indices[-1  , :ny, :].flatten()))

                rows = np.hstack((rows, np.tile(vor_indices[:(nx-1), ny:, [0, 2]][:, ::-1, :].flatten(), 2)))
                rows = np.hstack((rows, vor_indices[-1, ny:, [0, 2]].flatten()[::-1]))

                rows = np.hstack((rows, np.tile(vor_indices[:(nx-1), ny:, 1][:, ::-1].flatten(), 2)))
                rows = np.hstack((rows, vor_indices[-1, ny:, 1].flatten()))

                cols = np.concatenate([
                    mesh_indices[:-1, :, :].flatten(),
                    mesh_indices[1:  , :, :].flatten(),
                    mesh_indices[-1  , :, :].flatten(),

                    mesh_indices[:-1, :-1, [0, 2]].flatten(),
                    mesh_indices[1:  , :-1, [0, 2]].flatten(),
                    mesh_indices[-1  , :-1, [0, 2]][::-1, :].flatten(),

                    mesh_indices[:-1, :-1, 1].flatten(),
                    mesh_indices[1:  , :-1, 1].flatten(),
                    mesh_indices[-1  , :-1, 1][::-1].flatten(),
                ])

                data = np.concatenate([
                    0.75 * np.ones((nx-1) * ny * 3),
                    0.25 * np.ones((nx-1) * ny * 3),
                    np.ones(ny * 3),  # back row

                    0.75 * np.ones((nx-1) * (ny-1) * 2),
                    0.25 * np.ones((nx-1) * (ny-1) * 2),
                    np.ones((ny-1) * 2),  # back row

                    -0.75 * np.ones((nx-1) * (ny-1)),
                    -.25  * np.ones((nx-1) * (ny-1)),
                    -np.ones((ny-1)),  # back row
                ])

                self.declare_partials(vortex_mesh_name, mesh_name, val=data, rows=rows, cols=cols)
                """
            else:
                self.add_output(vortex_mesh_name, shape=(nx-1, ny-1, 4, 3), units='m')

                """
                mesh_indices = np.arange(nx * ny * 3).reshape(
                    (nx, ny, 3))

                rows = np.tile(mesh_indices[:(nx-1), :, :].flatten(), 2)
                rows = np.hstack((rows, mesh_indices[-1  , :, :].flatten()))
                cols = np.concatenate([
                    mesh_indices[:-1, :, :].flatten(),
                    mesh_indices[1:  , :, :].flatten(),
                    mesh_indices[-1  , :, :].flatten(),
                ])

                data = np.concatenate([
                    0.75 * np.ones((nx-1) * ny * 3),
                    0.25 * np.ones((nx-1) * ny * 3),
                    np.ones(ny * 3),  # back row
                ])

                self.declare_partials(vortex_mesh_name, mesh_name, val=data, rows=rows, cols=cols)
                """
            self.declare_partials(vortex_mesh_name, mesh_name, method='fd')

    def compute(self, inputs, outputs):
        surfaces = self.options['surfaces']

        for surface in surfaces:
            nx = surface['mesh'].shape[0]
            ny = surface['mesh'].shape[1]
            name = surface['name']

            mesh_name = '{}_def_mesh'.format(name)
            vortex_mesh_name = '{}_vortex_mesh'.format(name)

            if surface['symmetry']:
                # mesh = np.zeros((nx, ny*2-1, 3), dtype=type(inputs[mesh_name][0, 0, 0]))
                # mesh[:, :ny, :] = inputs[mesh_name]
                # mesh[:, ny:, :] = inputs[mesh_name][:, :-1, :][:, ::-1, :]
                # mesh[:, ny:, 1] *= -1.
                mesh = np.zeros((nx-1, ny*2-1-1, 4, 3), dtype=type(inputs[mesh_name][0, 0, 0, 0]))
                mesh[:, :ny-1, :, :] = inputs[mesh_name]
                mesh[:, ny-1:, :, :] = inputs[mesh_name][:, ::-1, [2,3,0,1], :]  # flip (0,1) <-> (2,3)
                mesh[:, ny-1:, :, 1] *= -1.  # flip y sign
            else:
                mesh = inputs[mesh_name]

            # outputs[vortex_mesh_name][:-1, :, :] = 0.75 * mesh[:-1, :, :] + 0.25 * mesh[1:, :, :]
            # outputs[vortex_mesh_name][-1, :, :] = mesh[-1, :, :]
            # panels except for the last (trailing edge) row
            outputs[vortex_mesh_name][:-1, :, :, :] = 0.75 * mesh[:-1, :, :, :] + 0.25 * mesh[1:, :, :, :]
            # panels along the trailing edge
            outputs[vortex_mesh_name][-1, :, [0,2], :] = 0.75 * mesh[-1, :, [0,2], :] + 0.25 * mesh[-1, :, [1,3], :]  # quarter-chord line
            outputs[vortex_mesh_name][-1, :, [1,3], :] = mesh[-1, :, [1,3], :]   # trailing edge


if __name__ == '__main__':
    # TODO: add this as an unit test
    nx = 3
    ny = 5

    mesh_orig = np.zeros((nx, ny, 3))
    for i in range(ny):
        mesh_orig[:, i, 0] = np.arange(0, nx)
    for i in range(nx):
        mesh_orig[i, :, 1] = np.arange(ny-1, -1, -1)
    
    print('mesh_x', mesh_orig[:, :, 0])
    print('mesh_y', mesh_orig[:, :, 1])

    prob = om.Problem()
    indep = prob.model.add_subsystem('indep', om.ExplicitComponent(), promotes_outputs=['*'])
    indep.add_output('mesh_orig', val=mesh_orig)

    mesh_shape = (nx, ny, 3)
    from openaerostruct.geometry.convert_mesh import ConvertMesh
    prob.model.add_subsystem('convertmesh', ConvertMesh(mesh_shape=mesh_shape), promotes=['*'])
    surfaces = [{'mesh' : mesh_orig, 'name' : 'wing', 'symmetry' : True}]
    prob.model.add_subsystem('vortexmesh', VortexMesh(surfaces=surfaces), promotes_outputs=['*'])
    prob.model.connect('mesh', 'vortexmesh.wing_def_mesh')

    prob.setup(check=True)

    # om.n2(prob)

    prob.run_model()

    mesh_out = prob['mesh']
    print('converted mesh at cell 0. 0', mesh_out[0, 0, :, :] )
    print('converted mesh at cell -1, 0', mesh_out[-1, 0, :, :] )
    print('converted mesh at cell 0, -1', mesh_out[0, -1, :, :] )
    print('converted mesh at cell -1, -1', mesh_out[-1, -1, :, :] )
    print('')

    mesh_vortex = prob['wing_vortex_mesh']
    print('vertex mesh shape', mesh_vortex.shape)
    print('vertex mesh at cell 0. 0', mesh_vortex[0, 0, :, :] )
    print('vertex mesh at cell -1, 0', mesh_vortex[-1, 0, :, :] )
    print('vertex mesh at cell 0, -1', mesh_vortex[0, -1, :, :] )
    print('vertex mesh at cell -1, -1', mesh_vortex[-1, -1, :, :] )

    # print('vertex mesh at cell 0. 3', mesh_vortex[0, 3, :, :] )
    # print('vertex mesh at cell -1, 3', mesh_vortex[-1, 3, :, :] )
    # print('vertex mesh at cell 0. 4', mesh_vortex[0, 4, :, :] )
    # print('vertex mesh at cell -1, 4', mesh_vortex[-1, 4, :, :] )
        
