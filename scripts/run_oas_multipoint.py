"""
Massive multi-point OAS optimization.
Similar problem structure to trajectory optimization, but this does not use Dymos

This problem is just for testing memory usage and scalability. Not a real design problem.

"""

# avoid numpy multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from mpi4py import MPI
import numpy as np
import openmdao.api as om

import time as time_package

from dynamics import Aircraft2DODE
from get_oas_surface import get_OAS_surface

if __name__ == '__main__':
    # set problem size here.
    num_segments = 100   # number of segments in dymos trajectory
    num_nodes = 4 * num_segments   # 4 nodes per 1 Radau 3rd-order segment

    # --- mission/aircraft settings ---
    # reference cruise speed
    v_cruise = 30.  # m/s, ref cruise speed
    vmin = 25.
    vmax = 35.

    # setup aircraft design
    mass = 7.2
    Sref = 2 * mass * 9.81 / (1.225 * v_cruise**2 * 0.5)  # wing ref area. CL = 0.5 at reference cruise speed
    span = 1.6  # m

    # thrust bounds
    thrust_ref = mass * 9.81 * 0.1  # N
    thrust_LB = -mass * 9.81 * 0.05  # negative thrust = brake, N
    thrust_UB = mass * 9.81 * 0.15

    # --- setup OAS surface ---
    surface = get_OAS_surface(Sref, span, num_y=41, num_x=9)
    ### surface = get_OAS_surface(Sref, span, num_y=21, num_x=5)
    ### surface = get_OAS_surface(Sref, span, num_y=5, num_x=2)   # use coarse mesh when computing total coloring. Also, for this mesh size, DirectSolver at top-level is much faster than PETSc linear solver.

    # --------------------------
    # Setup OpenMDAO problem
    # --------------------------
    prob = om.Problem(reports=True)

    # wing design parameter
    design_var_comp = om.IndepVarComp()
    design_var_comp.add_output("twist_cp", val=surface["twist_cp"], units='deg')
    design_var_comp.add_output("thickness_cp", val=surface["thickness_cp"], units='m')
    prob.model.add_subsystem('wing_design_vars', design_var_comp)
    # optimize wing design
    # twist0 = surface['twist_cp']
    # prob.model.add_design_var('wing_design_vars.twist_cp', lower=twist0 - 5, upper=twist0 + 5, ref=10, units='deg')
    # prob.model.add_design_var('wing_design_vars.thickness_cp', lower=0.001, upper=0.004, ref=0.001, units='m')

    # states and controls
    states_cons_comp = om.IndepVarComp()
    states_cons_comp.add_output('v', val=np.ones(num_nodes) * v_cruise, units='m/s')
    states_cons_comp.add_output('alpha', val=np.ones(num_nodes), units='deg')
    states_cons_comp.add_output('thrust', val=np.ones(num_nodes) * thrust_ref, units='N')
    prob.model.add_subsystem('states_controls', states_cons_comp, promotes_outputs=['*'])
    # optimize states and controls
    prob.model.add_design_var('v', lower=vmin, upper=vmax, ref=v_cruise, units='m/s')
    prob.model.add_design_var('alpha', lower=-10, upper=10, ref=10, units='deg')
    prob.model.add_design_var('thrust', lower=thrust_LB, upper=thrust_UB, ref=thrust_ref, units='N')

    # add ODE (which includes parallelized OAS multipoint analyses inside)
    prob.model.add_subsystem('ode', Aircraft2DODE(num_nodes=num_nodes, OAS_surface=surface), promotes=['*'])

    # connect wing design to dynamics model
    prob.model.connect('wing_design_vars.twist_cp', 'aero.OAS.wing.twist_cp')
    prob.model.connect('wing_design_vars.thickness_cp', 'aero.OAS.wing.thickness_cp')
    # fixed inputs to dynamics model
    prob.model.set_input_defaults('S_ref', Sref, units='m**2')
    prob.model.set_input_defaults('m', mass * np.ones(num_nodes), units='kg')

    # some non-important objective and constraints
    prob.model.add_objective('aero.f_drag', index=0)
    prob.model.add_constraint('fd.v_dot', equals=0.)
    ### prob.model.add_constraint('ode.fd.z_dot', equals=0.)
    prob.model.add_constraint('fd.gam_dot', equals=0.)
    ### prob.model.add_constraint('ode.fd.chi_dot', equals=0.)

    # --- optimizer ---
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.options['print_results'] = True
    prob.driver.opt_settings['Iterations limit'] = 30000
    prob.driver.opt_settings['Major iterations limit'] = 0
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-5
    prob.driver.opt_settings['Verify level'] = -1   # do not check gradient
    prob.driver.opt_settings['Function precision'] = 1e-10
    prob.driver.opt_settings['Hessian full memory'] = 1
    prob.driver.opt_settings['Hessian frequency'] = 100
    
    # --- setup ---
    t0 = time_package.time()
    prob.setup(check=False, mode='fwd')
    t_setup = time_package.time() - t0
    print('\n\n     Finished setup \n\n')

    # om.n2(prob)

    """
    # --- update solver settings for each OAS system ---
    for subsystem in prob.model.system_iter():
        if subsystem.name == 'coupled':
            subsystem.nonlinear_solver.options['iprint'] = 0  # turn off solver print

            # linear solver for derivatives
            subsystem.linear_solver = om.PETScKrylov(assemble_jac=False, iprint=0, err_on_non_converge=True)
            subsystem.linear_solver.precon = om.LinearRunOnce(iprint=-1)


    # manually call final_setup so that we can only timing the analysis
    t0 = time_package.time()
    prob.final_setup()
    t_final_setup = time_package.time() - t0
    print('\n\n     Finished final_setup \n\n')

    t0 = time_package.time()
    prob.run_model()
    t_run_model = time_package.time() - t0
    print('\n\n     Finished run_model \n\n')

    t0 = time_package.time()
    prob.compute_totals()
    t_compute_totals = time_package.time() - t0
    print('\n\n     Finished compute_totals \n\n')

    if MPI.COMM_WORLD.rank == 0:
        print('num procs:', MPI.COMM_WORLD.size)
        print('runtimes [s]')
        print('   `setup` :', t_setup)
        print('   `final_setup` :', t_final_setup)
        print('   `run_model` :', t_run_model)
        print('   `compute_totals` :', t_compute_totals)
    """



