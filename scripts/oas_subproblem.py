import numpy as np
import openmdao.api as om

from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint

# NOTE / TODO: currently, the wing geometry group (AerostructGeometry is inside the OAS subproblem, therefore we instantiate one wing geom group per node.
# Alternatively, we can move that to the top-level problem, and pass the wing geometry as an input to the subproblem. (Not sure if that's good or bad)

# WARNING: when you add additional design variables, states, or controls at the top-level, you'll need to change the inputs, outputs, and *partials* here.

# TODO: use om.SubmodelComp and refactor. But can I set mode ('rev' or 'fwd') for the submodelcomp?


class AeroStructPoint_SubProblemWrapper(om.ExplicitComponent):
    """
    Wrapper of a (single-point) OAS analysis.

    Currently, in only declares the partials of [CL, CD] w.r.t. [v, alpha, m].
    Other outputs and variables are assumed to be not part of the optimization problem.

    Parameters
    ----------
    v : float
        freestream air velocity, m/s
    alpha : float
        angle of attack, deg
    rho : float
        air density, kg/m**3
    W0 : float
        aircraft mass, kg
    load_factor : float
        load factor
    Mach_number : float
        freestream Mach number
    re : float
        freestream Reynolds number, 1/m
    twist_cp : numpy array
        twist control points, deg
    thickness_cp : numpy array
        thickness control points, m
    
    Returns
    -------
    Lift : float
        lift, N
    Drag : float
        drag, N
    CL : float
        lift coefficient
    CD : float
        drag coefficient
    S_ref : float
        wing reference area, m**2
    Cl : numpy array
        sectional lift coefficient
    sec_forces : numpy array
        sectional forces, N
    def_mesh : numpy array
        deformed mesh, m
    vonmises : numpy array
        von-mises stress, N/m**2
    failure : float
        failure metric (KS-aggregated, should be <=0)
    """

    def initialize(self):
        self.options.declare('surface', types=dict)
        self.options.declare('optimize_design', types=bool, default=True)   # set True to compute derivatives w.r.t. twist and thickenss

    def setup(self):
        surface = self.options['surface']

        # --- inputs ---
        # flight conditions
        self.add_input('v', val=10., units='m/s')
        self.add_input('alpha', val=5., units='deg')
        self.add_input('rho', val=1.225, units='kg/m**3')
        self.add_input('W0', val=10., units='kg')
        self.add_input('load_factor', val=1.)
        self.add_input("Mach_number", val=0.044)
        self.add_input("re", val=1.0e6, units="1/m")
        # wing design variables
        self.add_input('twist_cp', val=surface["twist_cp"], units='deg')
        self.add_input('thickness_cp', val=surface["thickness_cp"], units='m')

        # --- outputs ---
        # Lift and drag are used by dynamics model
        self.add_output('Lift', shape=(1,), units='N')
        self.add_output('Drag', shape=(1,), units='N')

        # the following outputs are for logging only (hence no derivatives)
        self.add_output('CL', shape=(1,))
        self.add_output('CD', shape=(1,))
        # wing reference area
        self.add_output('S_ref', val=1., units='m**2')
        # sectional lift coeff
        ny = surface['mesh'].shape[1] - 1
        self.add_output('Cl', shape=(ny,))
        # sectional force
        mesh_shape = surface['mesh'].shape
        sec_force_shape = (mesh_shape[0] - 1, mesh_shape[1] - 1, 3)
        self.add_output('sec_forces', shape=sec_force_shape, units='N')
        # deformed mesh
        mesh_shape = surface['mesh'].shape
        self.add_output('def_mesh', shape=mesh_shape, units='m')
        # von-mises stress
        ny = mesh_shape[1]
        self.add_output('vonmises', shape=(ny - 1, 2), units='N/m**2')
        # failure metric (KS-aggregated, should be <=0)
        self.add_output('failure', shape=(1,))

        # --- partials ---
        # TODO: if I change problem, declare the additional partials here.
        # - declare the partial of failure to include failure constraint
        # - declare the partials w.r.t. load factor to include bank angle as control
        outputs = ['Lift', 'Drag']
        inputs = ['v', 'alpha', 'W0']
        if self.options['optimize_design']:
            inputs += ['twist_cp', 'thickness_cp']
        self.declare_partials(outputs, inputs)

        # ------------------------------------
        # setup OAS subproblem
        # ------------------------------------
        # Create the problem and assign the model group
        prob = om.Problem(reports=False)

        # Add problem information as an independent variables component
        indep_var_comp = om.IndepVarComp()
        indep_var_comp.add_output("v", val=29., units="m/s")
        indep_var_comp.add_output("alpha", val=5.0, units="deg")
        indep_var_comp.add_output("W0", val=10., units="kg")   # = W0
        indep_var_comp.add_output("Mach_number", val=0.044)
        indep_var_comp.add_output("re", val=1.0e6, units="1/m")
        indep_var_comp.add_output("rho", val=1.225, units="kg/m**3")
        indep_var_comp.add_output("load_factor", val=1.0, units=None)
        # following settings are not used, but we set them here to clean up the N2 diagram AutoIVCs
        indep_var_comp.add_output("empty_cg", val=np.array([0.13, 0.0, 0.0]), units="m")
        indep_var_comp.add_output("CT", val=9.81 * 8.6e-6, units="1/s")
        indep_var_comp.add_output("speed_of_sound", val=340, units="m/s")
        indep_var_comp.add_output("R", val=10e3, units="m")
        indep_var_comp.add_output("element_mass", val=0., units="kg")
        prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

        # design variables
        design_var_comp = om.IndepVarComp()
        design_var_comp.add_output("twist_cp", val=surface["twist_cp"], units='deg')
        design_var_comp.add_output("thickness_cp", val=surface["thickness_cp"], units='m')
        prob.model.add_subsystem('wing_design_vars', design_var_comp)

        aerostruct_group = AerostructGeometry(surface=surface)

        name = "wing"

        # Add tmp_group to the problem with the name of the surface.
        prob.model.add_subsystem(name, aerostruct_group)
        # Connect design variables
        prob.model.connect('wing_design_vars.twist_cp', name + '.twist_cp')
        prob.model.connect('wing_design_vars.thickness_cp', name + '.thickness_cp')

        point_name = "AS_point"

        # Create the aero point group and add it to the model
        AS_point = AerostructPoint(surfaces=[surface])

        prob.model.add_subsystem(
            point_name,
            AS_point,
            promotes_inputs=["Mach_number", "re", "empty_cg", "load_factor", "beta", "CT", "speed_of_sound", "R"],
        )

        # connect inputs to AS_point
        prob.model.connect('v', point_name + ".v")
        prob.model.connect('alpha', point_name + ".alpha")
        prob.model.connect('rho', point_name + ".rho")
        prob.model.connect('W0', point_name + ".W0")

        # Establish connections not taken care of internally
        prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
        prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

        # Connect aerodyamic mesh to coupled group mesh
        prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")

        # Connect performance calculation variables
        com_name = point_name + "." + name + "_perf"
        prob.model.connect(name + ".radius", com_name + ".radius")
        prob.model.connect(name + ".thickness", com_name + ".thickness")
        prob.model.connect(name + ".nodes", com_name + ".nodes")
        prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
        prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
        prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")

        # Set up the problem
        self._point_name = point_name
        self._prob_OAS = prob
        self._prob_OAS.setup(check=False, mode='rev')   # set 'rev' to perform adjoint in compute_totals. Need to maually specify here, because I don't declare desvar/functions, thus OM never knows.

        # solver settings
        prob.model.AS_point.coupled.nonlinear_solver.options['iprint'] = 0   # turn off solver print
        # PETSc linear solver for derivatives
        prob.model.AS_point.coupled.linear_solver = om.PETScKrylov(assemble_jac=False, iprint=0, err_on_non_converge=True)
        prob.model.AS_point.coupled.linear_solver.precon = om.LinearRunOnce(iprint=-1)

        # call final_setup here to eliminate it from run_model timing
        self._prob_OAS.final_setup()

    def compute(self, inputs, outputs):
        point_name = self._point_name

        # set flight conditions
        self._prob_OAS.set_val('v', inputs['v'], units='m/s')
        self._prob_OAS.set_val('alpha', inputs['alpha'], units='deg')
        self._prob_OAS.set_val('rho', inputs['rho'], units='kg/m**3')
        self._prob_OAS.set_val('W0', inputs['W0'], units='kg')
        self._prob_OAS.set_val('load_factor', inputs['load_factor'], units=None)
        self._prob_OAS.set_val('Mach_number', inputs['Mach_number'])
        self._prob_OAS.set_val('re', inputs['re'], units='1/m')

        # set design variables
        self._prob_OAS.set_val('wing_design_vars.twist_cp', inputs['twist_cp'], units='deg')
        self._prob_OAS.set_val('wing_design_vars.thickness_cp', inputs['thickness_cp'], units='m')

        # run OAS analysis
        self._prob_OAS.run_model()

        ### om.n2(self._prob_OAS, outfile='n2_OAS_subproblem.html', show_browser=False)

        # get outputs from the model
        outputs['Lift'] = self._prob_OAS.get_val(point_name + ".total_perf.L", units='N')
        outputs['Drag'] = self._prob_OAS.get_val(point_name + ".total_perf.D", units='N')
        outputs['CL'] = self._prob_OAS.get_val(point_name + ".wing_perf.CL")
        outputs['CD'] = self._prob_OAS.get_val(point_name + ".wing_perf.CD")

        outputs['S_ref'] = self._prob_OAS.get_val(point_name + ".coupled.wing.S_ref", units='m**2')
        outputs['Cl'] = self._prob_OAS.get_val(point_name + ".wing_perf.Cl")
        outputs['sec_forces'] = self._prob_OAS.get_val(point_name + ".coupled.aero_states.wing_sec_forces", units='N')
        outputs['def_mesh'] = self._prob_OAS.get_val(point_name + ".coupled.wing.def_mesh", units='m')
        outputs['vonmises'] = self._prob_OAS.get_val(point_name + ".wing_perf.vonmises", units='N/m**2')
        outputs['failure'] = self._prob_OAS.get_val(point_name + ".wing_perf.failure")
    
    def compute_partials(self, inputs, partials):
        """ Partials of this wrapper = total derivatives of OAS subproblem"""
        point_name = self._point_name

        # TODO: use cache here to avoid re-calling compute() for the same inputs.
        # (analysis restarts from the previously-converged point and converges in 1 iter, so it's probably not a big deal)
        self.compute(inputs, {})

        of = [point_name + ".total_perf.L", point_name + ".total_perf.D"]
        wrt = ['v', 'alpha', 'W0']
        if self.options['optimize_design']:
            wrt += ['wing_design_vars.twist_cp', 'wing_design_vars.thickness_cp']
        derivs = self._prob_OAS.compute_totals(of, wrt)

        partials['Lift', 'v'] = derivs[(point_name + ".total_perf.L", 'v')]
        partials['Lift', 'alpha'] = derivs[(point_name + ".total_perf.L", 'alpha')]
        partials['Lift', 'W0'] = derivs[(point_name + ".total_perf.L", 'W0')]
        partials['Drag', 'v'] = derivs[(point_name + ".total_perf.D", 'v')]
        partials['Drag', 'alpha'] = derivs[(point_name + ".total_perf.D", 'alpha')]
        partials['Drag', 'W0'] = derivs[(point_name + ".total_perf.D", 'W0')]
        if self.options['optimize_design']:
            partials['Lift', 'twist_cp'] = derivs[(point_name + ".total_perf.L", 'wing_design_vars.twist_cp')]
            partials['Lift', 'thickness_cp'] = derivs[(point_name + ".total_perf.L", 'wing_design_vars.thickness_cp')]
            partials['Drag', 'twist_cp'] = derivs[(point_name + ".total_perf.D", 'wing_design_vars.twist_cp')]
            partials['Drag', 'thickness_cp'] = derivs[(point_name + ".total_perf.D", 'wing_design_vars.thickness_cp')]


if __name__ == '__main__':

    # get OAS surface
    from get_oas_surface import get_OAS_surface
    mass = 7.2
    v_cruise = 30
    Sref = 2 * mass * 9.81 / (1.225 * v_cruise**2 * 0.5)  # wing ref area. CL = 0.5 at reference cruise speed
    span = 1.6  # m

    surface = get_OAS_surface(Sref, span, num_y=21, num_x=5)

    # create problem
    prob = om.Problem()
    prob.model.add_subsystem('AS_wrap', AeroStructPoint_SubProblemWrapper(surface=surface), promotes=['*'])

    # run problem
    prob.setup(check=True)

    # set flight conditions
    prob.set_val('v', 30, units='m/s')
    prob.set_val('alpha', 5, units='deg')
    prob.set_val('rho', 1.225, units='kg/m**3')
    prob.set_val('W0', mass, units='kg')
    prob.set_val('load_factor', 1.0, units=None)
    prob.set_val('Mach_number', 0.044)
    prob.set_val('re', 1.0e6, units='1/m')

    # set design variables
    prob.set_val('twist_cp', surface['twist_cp'], units='deg')
    prob.set_val('thickness_cp', surface['thickness_cp'], units='m')

    prob.run_model()

    # prob.check_partials(compact_print=True)

    derivs = prob.compute_totals(['CL', 'CD'], ['v', 'alpha', 'W0'])
    print('\n--- Total derivatives ---')
    print(derivs)

    om.n2(prob, outfile='n2_OAS_wrapper.html', show_browser=False)
