import numpy as np
import openmdao.api as om

# TODO: unify all these for any input/output dimensions
# TODO: when I declare the partial sparsity here, total coloring computation takes forever. What's happining?

class Scalars2Vector(om.ExplicitComponent):
    """
    Concatenates scalars into a vector
    Inputs: scalar0, scalar1, scalar2, ..., scalar<nn>
    Output: vector = [scalar0, scalar1, scalar2, ..., scalar<nn>]
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='length of the output vector')
        self.options.declare('units', types=str, default=None)

    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']

        for i in range(nn):
            self.add_input('scalar' + str(i), shape=(1,), units=units)
        # END FOR

        self.add_output('vector', shape=(nn,), units=units)

        # partials
        eyemat = np.eye(nn)
        for i in range(nn):
            self.declare_partials('vector', 'scalar' + str(i), val=eyemat[:, i])

        # partial coloring
        # self.declare_coloring(wrt='*', method='fd', show_summary=True, show_sparsity=False)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        vector = np.zeros(nn)
        for i in range(nn):
            vector[i] = inputs['scalar' + str(i)]
        # END FOR

        outputs['vector'] = vector

    # def compute_partials(self, inputs, partials):
    #     pass


class Vectors2Matrix(om.ExplicitComponent):
    """
    Concatenates vectors into a matrix
    Inputs: vector0, vector1, vector2, ..., vector<nn> (shape n_vec)
    Output: matrix = [vector0, vector1, ..., vector<nn>] (shape (nn, len_vector))
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='number of input vectors')
        self.options.declare('len_vector', types=int, desc='length of each vector (e.g. number of panels)')
        self.options.declare('units', types=str, default=None)

    def setup(self):
        nn = self.options['num_nodes']
        nvec = self.options['len_vector']
        units = self.options['units']

        for i in range(nn):
            self.add_input('vector' + str(i), shape=(nvec), units=units)
        # END FOR

        self.add_output('matrix', shape=(nn, nvec), units=units)
        
        # self.declare_partials('*', '*', method='exact')
        # # partial coloring
        # self.declare_coloring(wrt='*', method='fd', show_summary=True, show_sparsity=False)
        
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        nvec = self.options['len_vector']
        matrix = np.zeros((nn, nvec))
        for i in range(nn):
            matrix[i, :] = inputs['vector' + str(i)]
        # END FOR

        outputs['matrix'] = matrix

    # def compute_partials(self, inputs, partials):
    #     # TODO: implement partials if this output matrix is used for optimization. Currently, this code is only used to save timehistory, so partials here is not necessary
    #     pass


class Arrays2Dto3D(om.ExplicitComponent):
    """
    Concatenates 2D arrays into a 3D array
    Inputs: arr0, arr1, arr2, ..., arr<nn> (shape (nx, ny))
    Output: array = [arr0, arr1, arr2, ..., arr<nn>] (shape (nn, nx, ny))
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='number of input vectors')
        self.options.declare('input_shape', types=tuple, desc='(n1, n2) of the input array')
        self.options.declare('units', types=str, default=None)

    def setup(self):
        nn = self.options['num_nodes']
        input_shape = self.options['input_shape']
        units = self.options['units']

        for i in range(nn):
            self.add_input('array' + str(i), shape=input_shape, units=units)
        # END FOR

        self.add_output('array_out', shape=(nn, input_shape[0], input_shape[1]), units=units)

        # self.declare_partials('*', '*', method='exact')
        # # partial coloring
        # self.declare_coloring(wrt='*', method='fd', show_summary=True, show_sparsity=False)
        
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        input_shape = self.options['input_shape']
        array_out = np.zeros((nn, input_shape[0], input_shape[1]))
        for i in range(nn):
            array_out[i, :, :] = inputs['array' + str(i)]
        # END FOR

        outputs['array_out'] = array_out

    # def compute_partials(self, inputs, partials):
    #     # TODO: implement partials if this output matrix is used for optimization. Currently, this code is only used to save timehistory, so partials here is not necessary
    #     pass


class Arrays3Dto4D(om.ExplicitComponent):
    """
    Concatenates 3D arrays (mesh) into a 4D matrix (time history of mesh)
    Inputs: mesh0, mesh1, mesh2, ..., mesh<nn> (shape (nx, ny, 3))
    Output: array = [mesh0, mesh1, mesh2, ..., mesh<nn>] (shape (nn, nx, ny, 3))
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='number of input vectors')
        self.options.declare('input_shape', types=tuple, desc='(nx, ny, 3)')
        self.options.declare('units', types=str, default=None)

    def setup(self):
        nn = self.options['num_nodes']
        input_shape = self.options['input_shape']
        units = self.options['units']

        for i in range(nn):
            self.add_input('array' + str(i), shape=input_shape, units=units)
        # END FOR

        self.add_output('array_out', shape=(nn, input_shape[0], input_shape[1], input_shape[2]), units=units)
        
        # self.declare_partials('*', '*', method='exact')
        # # partial coloring
        # self.declare_coloring(wrt='*', method='fd', show_summary=True, show_sparsity=False)
        
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        input_shape = self.options['input_shape']
        array_out = np.zeros((nn, input_shape[0], input_shape[1], input_shape[2]))
        for i in range(nn):
            array_out[i, :, :, :] = inputs['array' + str(i)]
        # END FOR

        outputs['array_out'] = array_out

    # def compute_partials(self, inputs, partials):
    #     # TODO: implement partials if this output matrix is used for optimization. Currently, this code is only used to save timehistory, so partials here is not necessary
    #     pass
