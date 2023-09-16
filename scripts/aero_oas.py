# docs checkpoint 0
import numpy as np

import openmdao.api as om

from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

from utils import Scalars2Vector, Vectors2Matrix, Arrays2Dto3D, Arrays3Dto4D

from oas_subproblem import AeroStructPoint_SubProblemWrapper
from oas_subproblem_eff import AeroStructPoint_SubProblemWrapper as AeroStructPoint_SubProblemWrapper_NoGeometry


class AeroForceOAS(om.Group):
    """
    Computes the aerodynamic force in the wind frame.
    It calls OpenAeroStruct aerostructural analysis at each node to compute CL and CD.
    The OAS analyses can be parallelized. The `ParallelGroup` is added under `OASAnalyses` group.
    
    Parameters
    ----------
    v : ndarray, shape (num_nodes,)
        air velocity at each node
    alpha : ndarray, shape (num_nodes,)
        angle of attack
    theta : ndarray, shape (num_nodes,)
        bank angle
    m : ndarray, shape (num_nodes,)
        mass (kg)

    Returns
    -------
    f_lift : ndarray, shape (num_nodes,)
        lift at each node, N
    f_drag : ndarray, shape (num_nodes,)
        drag at each node, N
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated')
        self.options.declare('OAS_surface', types=dict, desc='Surface dict for OAS')

    def setup(self):
        nn = self.options['num_nodes']

        # air density - hardcoded for now
        indep = self.add_subsystem('indep', om.IndepVarComp(), promotes_outputs=['*'])
        indep.add_output('rho', val=1.2 * np.ones(nn), units="kg/m**3")

        # compute CL and CD by OpenAeroStruct
        self.add_subsystem(name='OAS',
                           subsys=OASAnalyses(num_nodes=nn, OAS_surface=self.options['OAS_surface']),
                           promotes_inputs=['rho', 'v', 'alpha', 'm', 'theta'],
                           promotes_outputs=[('Lift', 'f_lift'), ('Drag', 'f_drag')])


class OASAnalyses(om.Group):
    """
    Computes CL and CD of the wing using OAS aerostructural analysis.
    
    Inputs: v, alpha, theta, rho, m (vectors)
    Outputs: Lift, Drag (vectors)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, desc='Number of nodes to be evaluated')
        self.options.declare('OAS_surface', types=dict, desc='Surface dict for OAS')
        self.options.declare('use_subproblem', types=bool, default=True, desc='If True, use a OAS subproblem and wrapper instead of directly adding OAS as a subsystem')
        self.options.declare('compact_subproblem_coupling', types=bool, default=False)
        # If compact_subproblem_coupling = True, the wing geometry is within the subproblem (i.e., we'll have one wing geometry component per node).
        #     This is not efficient (for non-morping wing), but this simplified the coupling a lot. From the top-level problem into subprobelm, we'll only need to give the flight conditions (v, alpha, etc) and design variables (twist_cp, thickness_cp, etc).
        # If True, then the wing geometry is at the top-level, and subproblem only includes the analysis point. This should be more efficient.
        #     However, we have to pass many variables to subproblem, such as mesh.

    def setup(self):
        nn = self.options['num_nodes']
        surface = self.options['OAS_surface']

        # --- setup OAS model ---
        # Time-independent settings. A lot of things are hardcoded for now
        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output("Mach_number", val=0.044)   # TODO: compute Mach and Reynolds number based on v
        indep_var_comp.add_output("re", val=1.0e6, units="1/m")   # TODO: see above
        indep_var_comp.add_output("beta", val=0, units="deg")   # sideslip angle
        # following settings are not used, but we set them here to clean up the N2 diagram AutoIVCs
        indep_var_comp.add_output("empty_cg", val=np.array([0.13, 0.0, 0.0]), units="m")
        indep_var_comp.add_output("CT", val=9.81 * 8.6e-6, units="1/s")
        indep_var_comp.add_output("speed_of_sound", val=340, units="m/s")
        indep_var_comp.add_output("R", val=10e3, units="m")
        indep_var_comp.add_output("load_factor", val=1.0, units=None)
        
        # Add this IndepVarComp to the problem model
        self.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

        # this component does nothing, but this hack is necessary to connect the vector inputs (v, alpha, rho, m) to each OAS points.
        vec_comp = om.ExecComp(['v_vector = v', 'alpha_vector = alpha', 'rho_vector = rho', 'm_vector = m'],
                               v_vector={'shape': (nn), 'units': 'm/s'},
                               v={'shape': (nn), 'units': 'm/s'},
                               alpha_vector={'shape': (nn), 'units': 'deg'},
                               alpha={'shape': (nn), 'units': 'deg'},
                               rho_vector={'shape': (nn), 'units': 'kg/m**3'},
                               rho={'shape': (nn), 'units': 'kg/m**3'},
                               m_vector={'shape': (nn), 'units': 'kg'},
                               m={'shape': (nn), 'units': 'kg'},
                               has_diag_partials=True)
        self.add_subsystem('vector_in', vec_comp, promotes_inputs=['v', 'alpha', 'rho', 'm'])
    
        # load factor (used to compute the structural load; not relevant for aerodynamic analysis)
        load_factor_comp = om.ExecComp('load_factor = cos(theta)',
                                       load_factor={'shape': (nn), 'units': None},
                                       theta={'shape': (nn), 'units': 'rad'},
                                       has_diag_partials=True)
        self.add_subsystem('str_load_factor', load_factor_comp, promotes_inputs=['theta'])

        # OAS coupling type
        if not self.options['use_subproblem']:
            oas_type = 'mono'
        elif self.options['compact_subproblem_coupling']:
            oas_type = 'sub-cpt'
        else:
            oas_type = 'sub-eff'
        # END IF
        print('\n\n', 'OAS coupling type =', oas_type, '\n\n')

        # list of OAS variable names that changes depending on if we use subproblem or not
        oas_vars = {}
        oas_vars['load_factor'] = {'mono': '.coupled.load_factor', 'sub-cpt': '.load_factor', 'sub-eff': '.load_factor'}
        oas_vars['CL'] = {'mono': '.wing_perf.CL', 'sub-cpt': '.CL', 'sub-eff': '.CL'}
        oas_vars['CD'] = {'mono': '.wing_perf.CD', 'sub-cpt': '.CD', 'sub-eff': '.CD'}
        oas_vars['Lift'] = {'mono': '.total_perf.L', 'sub-cpt': '.Lift', 'sub-eff': '.Lift'}
        oas_vars['Drag'] = {'mono': '.total_perf.D', 'sub-cpt': '.Drag', 'sub-eff': '.Drag'}
        oas_vars['sec_Cl'] = {'mono': '.wing_perf.Cl', 'sub-cpt': '.Cl', 'sub-eff': '.Cl'}
        oas_vars['sec_forces'] = {'mono': '.coupled.aero_states.wing_sec_forces', 'sub-cpt': '.sec_forces', 'sub-eff': '.sec_forces'}
        oas_vars['def_mesh'] = {'mono': '.coupled.wing.def_mesh', 'sub-cpt': '.def_mesh', 'sub-eff': '.def_mesh'}
        oas_vars['vonmises'] = {'mono': '.wing_perf.vonmises', 'sub-cpt': '.vonmises', 'sub-eff': '.vonmises'}
        oas_vars['failure'] = {'mono': '.wing_perf.failure', 'sub-cpt': '.failure', 'sub-eff': '.failure'}

        # -----------------------------
        # OAS aerostructural analyses
        # -----------------------------
        # --- wing geometry ---
        if oas_type == 'sub-cpt':
            # --- compact subproblem ---
            # a dummy wing geometry component to connect twist/thickness cp. Inputs of this component will be connected from Dymos traj parameters, and output will be connected to each OAS wrappers.
            n_cp_twist = len(surface['twist_cp'])
            n_cp_thickness = len(surface['thickness_cp'])
            wing_desvar_comp = om.ExecComp(['twist_cp_out = twist_cp', 'thickness_cp_out = thickness_cp'],
                                           twist_cp_out={'shape': (n_cp_twist), 'units': 'deg'},
                                           twist_cp={'shape': (n_cp_twist), 'units': 'deg'},
                                           thickness_cp_out={'shape': (n_cp_thickness), 'units': 'm'},
                                           thickness_cp={'shape': (n_cp_thickness), 'units': 'm'},
                                           has_diag_partials=True)
            self.add_subsystem('wing', wing_desvar_comp)
        else:
            # --- monolithic coupling or involved subproblem ---
            name = surface["name"]
            aerostruct_group = AerostructGeometry(surface=surface)
            self.add_subsystem(name, aerostruct_group)
        # END IF

        # parallelize OAS analysis at each node
        parallel_group = self.add_subsystem('parallel', om.ParallelGroup(), promotes=['*'])

        # --- add aerostruct analysis points ---
        for i in range(nn):
            point_name = "node_" + str(i)
            
            if oas_type == 'mono':
                # monolihic OM problem
                promotes_inputs = ["Mach_number", "re", "empty_cg", "load_factor", "beta", "CT", "speed_of_sound", "R"]
                parallel_group.add_subsystem(point_name, AerostructPoint(surfaces=[surface]), promotes_inputs=promotes_inputs)
            elif oas_type == 'sub-cpt':
                # subproblem
                promotes_inputs = ["Mach_number", "re"]
                parallel_group.add_subsystem(point_name, AeroStructPoint_SubProblemWrapper(surface=surface), promotes_inputs=promotes_inputs)
            else:
                # subproblem with no wing geometry
                promotes_inputs = ["Mach_number", "re"]
                parallel_group.add_subsystem(point_name, AeroStructPoint_SubProblemWrapper_NoGeometry(surface=surface), promotes_inputs=promotes_inputs)
            # END IF

            # --- connect inputs to OAS ---
            # connect load factor for structural loading. We ignore in-plane structural loading, only considers out-of-plane component of the gravity
            self.connect('str_load_factor.load_factor', point_name + oas_vars['load_factor'][oas_type], src_indices=i)
            
            # connect m, rho, v and alpha (vector) to each point
            self.connect('vector_in.rho_vector', point_name + '.rho', src_indices=i)
            self.connect('vector_in.v_vector', point_name + '.v', src_indices=i)
            self.connect('vector_in.alpha_vector', point_name + '.alpha', src_indices=i)
            self.connect('vector_in.m_vector', point_name + '.W0', src_indices=i)

            if oas_type == 'mono':
                # connection from wing geometry group to analysis point
                self.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
                self.connect(name + ".element_mass", point_name + "." + "coupled." + name + ".element_mass")
                self.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")
                self.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
                com_name = point_name + "." + name + "_perf"
                self.connect(name + ".radius", com_name + ".radius")
                self.connect(name + ".thickness", com_name + ".thickness")
                self.connect(name + ".nodes", com_name + ".nodes")
                self.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
                self.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
                self.connect(name + ".t_over_c", com_name + ".t_over_c")
            elif oas_type == 'sub-eff':
                self.connect(name + ".local_stiff_transformed", point_name + ".local_stiff_transformed")
                self.connect(name + ".element_mass", point_name + ".element_mass")
                self.connect(name + ".nodes", point_name + ".nodes")
                self.connect(name + ".mesh", point_name + ".mesh")
                self.connect(name + ".radius", point_name + ".radius")
                self.connect(name + ".thickness", point_name + ".thickness")
                self.connect(name + ".cg_location", point_name + ".cg_location")
                self.connect(name + ".structural_mass", point_name + ".structural_mass")
                self.connect(name + ".t_over_c", point_name + ".t_over_c")
            else:
                # connect wing design variables for subproblem
                self.connect('wing.twist_cp_out', point_name + '.twist_cp')
                self.connect('wing.thickness_cp_out', point_name + '.thickness_cp')
            # END IF
        # END FOR (add points)

        # --- connect outputs from OAS ---
        # Output Lift, Drag, CL, and CD vectors
        self.add_subsystem('CL_vector', Scalars2Vector(num_nodes=nn, units=None), promotes_outputs=[('vector', 'CL')])
        self.add_subsystem('CD_vector', Scalars2Vector(num_nodes=nn, units=None), promotes_outputs=[('vector', 'CD')])
        self.add_subsystem('Lift_vector', Scalars2Vector(num_nodes=nn, units='N'), promotes_outputs=[('vector', 'Lift')])
        self.add_subsystem('Drag_vector', Scalars2Vector(num_nodes=nn, units='N'), promotes_outputs=[('vector', 'Drag')])
        # connect outputs of each aero point into the vectors
        for i in range(nn):
            point_name = "node_" + str(i)
            self.connect(point_name + oas_vars['CL'][oas_type], 'CL_vector.scalar' + str(i))
            self.connect(point_name + oas_vars['CD'][oas_type], 'CD_vector.scalar' + str(i))
            self.connect(point_name + oas_vars['Lift'][oas_type], 'Lift_vector.scalar' + str(i))
            self.connect(point_name + oas_vars['Drag'][oas_type], 'Drag_vector.scalar' + str(i))
        # END FOR

        # --- log time histories of other variables ---
        # TODO: do these off-line after optimization to save runtime, as these are not used in optimization but only used for post-processing and plotting

        # Cl distribution
        ny = surface['mesh'].shape[1] - 1
        self.add_subsystem('Cl_history', Vectors2Matrix(num_nodes=nn, len_vector=ny, units=None), promotes_outputs=[('matrix', 'Cl_dist_his')])   # output matrix: (nn, ny)
        for i in range(nn):
            point_name = "node_" + str(i)
            self.connect(point_name + oas_vars['sec_Cl'][oas_type], 'Cl_history.vector' + str(i))

        # sectional force
        mesh_shape = surface['mesh'].shape
        sec_force_shape = (mesh_shape[0] - 1, mesh_shape[1] - 1, 3)
        self.add_subsystem('sec_forces_history', Arrays3Dto4D(num_nodes=nn, input_shape=sec_force_shape, units='N'), promotes_outputs=[('array_out', 'sec_forces_his')])   # output matrix: (nn, nx-1, ny-1, 3)
        for i in range(nn):
            point_name = "node_" + str(i)
            self.connect(point_name + oas_vars['sec_forces'][oas_type], 'sec_forces_history.array' + str(i))

        # deformed mesh (aerostructural only)
        mesh_shape = surface['mesh'].shape
        self.add_subsystem('mesh_history', Arrays3Dto4D(num_nodes=nn, input_shape=mesh_shape, units='m'), promotes_outputs=[('array_out', 'mesh_his')])   # output matrix: (nn, nx, ny, 3)
        for i in range(nn):
            point_name = "node_" + str(i)
            self.connect(point_name + oas_vars['def_mesh'][oas_type], 'mesh_history.array' + str(i))

        # von-mises stress (aerostructural only)
        ny = mesh_shape[1]
        self.add_subsystem('stress_history', Arrays2Dto3D(num_nodes=nn, input_shape=(ny - 1, 2), units='N/m**2'), promotes_outputs=[('array_out', 'stress_his')])   # output matrix: (nn, ny - 1, 2)
        for i in range(nn):
            point_name = "node_" + str(i)
            self.connect(point_name + oas_vars['vonmises'][oas_type], 'stress_history.array' + str(i))

        # failure metric (KS-aggregated, should be <=0)
        self.add_subsystem('failure_history', Scalars2Vector(num_nodes=nn, units=None), promotes_outputs=[('vector', 'failure_his')])
        for i in range(nn):
            point_name = "node_" + str(i)
            self.connect(point_name + oas_vars['failure'][oas_type], 'failure_history.scalar' + str(i))
            # END FOR


class DynamicPressureComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(name='rho', val=0.5 * np.ones(nn), desc='atmospheric density', units='kg/m**3')
        self.add_input(name='v', shape=(nn,), desc='air-relative velocity', units='m/s')
        self.add_output(name='q', shape=(nn,), desc='dynamic pressure', units='N/m**2')
        ar = np.arange(nn)
        self.declare_partials(of='q', wrt='rho', rows=ar, cols=ar)
        self.declare_partials(of='q', wrt='v', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['q'] = 0.5 * inputs['rho'] * inputs['v'] ** 2

    def compute_partials(self, inputs, partials):
        partials['q', 'rho'] = 0.5 * inputs['v'] ** 2
        partials['q', 'v'] = inputs['rho'] * inputs['v']


class LiftDragForceComp(om.ExplicitComponent):
    """
    Compute the aerodynamic forces (lift, drag) on the vehicle in the wind axis frame. Side force is assumed 0.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(name='CL', val=np.zeros(nn,), desc='lift coefficient', units=None)
        self.add_input(name='CD', val=np.zeros(nn,), desc='drag coefficient', units=None)
        self.add_input(name='q', val=np.zeros(nn,), desc='dynamic pressure', units='N/m**2')
        self.add_input(name='S', shape=(1,), desc='aerodynamic reference area', units='m**2')
        self.add_output(name='f_lift', shape=(nn,), desc='aerodynamic lift force', units='N')
        self.add_output(name='f_drag', shape=(nn,), desc='aerodynamic drag force', units='N')

        ar = np.arange(nn)
        self.declare_partials(of='f_lift', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='f_lift', wrt='S', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='f_lift', wrt='CL', rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='S', rows=ar, cols=np.zeros(nn))
        self.declare_partials(of='f_drag', wrt='CD', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S
        outputs['f_lift'] = qS * CL
        outputs['f_drag'] = qS * CD

    def compute_partials(self, inputs, partials):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S
        partials['f_lift', 'q'] = S * CL
        partials['f_lift', 'S'] = q * CL
        partials['f_lift', 'CL'] = qS
        partials['f_drag', 'q'] = S * CD
        partials['f_drag', 'S'] = q * CD
        partials['f_drag', 'CD'] = qS


if __name__ == '__main__':

    # Create a dictionary to store options about the mesh
    mesh_dict = {
        "num_y": 7,
        "num_x": 2,
        "wing_type": "rect",
        "symmetry": True,
        "span": 10.0,
        "chord": 1,
        "span_cos_spacing": 1.0,
        "chord_cos_spacing": 1.0,
    }

    # Generate half-wing mesh of rectangular wing
    mesh = generate_mesh(mesh_dict)

    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing
        # reflected across the plane y = 0
        "S_ref_type": "wetted",  # how we compute the wing area,
        # can be 'wetted' or 'projected'
        "fem_model_type": "tube",
        "thickness_cp": np.array([0.001]),
        "radius_cp": np.array([0.007]),
        "twist_cp": np.zeros(5),
        "mesh": mesh,
        # Aerodynamic performance of the lifting surface at
        # an angle of attack of 0 (alpha=0).
        # These CL0 and CD0 values are added to the CL and CD
        # obtained from aerodynamic analysis of the surface to get
        # the total CL and CD.
        # These CL0 and CD0 values do not vary wrt alpha.
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.015,  # CD of the surface at alpha=0
        # Airfoil properties for viscous drag calculation
        "k_lam": 0.9,  # percentage of chord with laminar
        # flow, used for viscous drag
        "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
        "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
        # thickness
        "with_viscous": True,
        "with_wave": False,  # if true, compute wave drag
        # Structural values are based on aluminum 7075
        "E": 72.6e9,  # [Pa] Young's modulus of the spar
        "G": 26.0e9,  # [Pa] shear modulus of the spar
        "yield": 250.0e6 / 2.5,  # [Pa] yield stress divided by 2.5 for limiting case
        "mrho": 2.79e3,  # [kg/m^3] material density
        "fem_origin": 0.35,  # normalized chordwise location of the spar
        "wing_weight_ratio": 5.0,
        "struct_weight_relief": True,  # True to add the weight of the structure to the loads on the structure
        "distributed_fuel_weight": False,
        # Constraints
        "exact_failure_constraint": False,  # if false, use KS function
    }

    p = om.Problem()
    p.model.add_subsystem('OAS', AeroForceOAS(num_nodes=3, OAS_surface=surface), promotes_inputs=['v', 'alpha'], promotes_outputs=['CL', 'CD'])

    p.setup(check=True)
    p.set_val('alpha', np.array([1, 2, 3]), units='deg')
    p.set_val('v', np.array([50, 60, 70]), units='m/s')
    
    # p.run_model()
    ### p.check_partials(compact_print=True)
    om.n2(p)
