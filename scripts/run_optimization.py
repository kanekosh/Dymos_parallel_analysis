"""
Trajectory optimization with OpenAeroStruct aerodynamic model

Here, we setup a very simple trajectory optimization problem where we minimize the energy consumption for straight and level flight.
At each time discretization node, we call OpenAeroStruct aerostructural analysis to compute the aerodynamic force.
OAS analyses can be performed in parallel (one processor for each node).

For this problem setup, the total Jacobian structure is independent of the OAS mesh size.
Therefore, to compute total coloring, I recommend to use a coarse OAS mesh.
"""

import matplotlib.pyplot as plt
import time as time_package

from mpi4py import MPI
import openmdao.api as om
import dymos as dm

from dynamics import Aircraft2DODE
from get_oas_surface import get_OAS_surface


if __name__ == '__main__':

    # --- total coloring file ---
    # coloring_file = 'coloring_files/total_coloring_Radau10.pkl'
    # coloring_file = 'coloring_files/total_coloring_Radau20.pkl'
    coloring_file = 'coloring_files/total_coloring_Radau40.pkl'
    # coloring_file = None
    
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
    surface, Sref = get_OAS_surface(Sref, span, num_y=21, num_x=5)
    ### surface, Sref = get_OAS_surface(Sref, span, num_y=7, num_x=2)   # use coarse mesh when computing total coloring

    # --------------------------
    # Setup OpenMDAO problem
    # --------------------------
    prob = om.Problem(reports=True)

    # wing design parameter
    design_var_comp = om.IndepVarComp()
    design_var_comp.add_output("twist_cp", val=surface["twist_cp"], units='deg')
    design_var_comp.add_output("thickness_cp", val=surface["thickness_cp"], units='m')
    prob.model.add_subsystem('wing_design_vars', design_var_comp)
    # simultaneously optimize wing design and trajectory
    ### twist0 = surface['twist_cp']
    ### prob.model.add_design_var('wing_design_vars.twist_cp', lower=twist0 - 5, upper=twist0 + 5, ref=10, units='deg')
    ### prob.model.add_design_var('wing_design_vars.thickness_cp', lower=0.001, upper=0.004, ref=0.001, units='m')

    # --- setup trajectory ---
    traj = dm.Trajectory()
    prob.model.add_subsystem('traj', traj)

    tx = dm.Radau(num_segments=40, order=3, solve_segments=False, compressed=True)
    ### tx = dm.Radau(num_segments=20, order=3, solve_segments=False, compressed=True)
    nn = tx.grid_data.num_nodes

    phase1 = dm.Phase(transcription=tx, ode_class=Aircraft2DODE, ode_init_kwargs={'OAS_surface': surface})
    traj.add_phase('phase1', phase1)

    phase1.set_time_options(fix_initial=True, duration_bounds=(1, 50), duration_ref=30)
    phase1.add_state('x', fix_initial=True, fix_final=True, rate_source='fd.x_dot', units='m', ref=1000.)
    phase1.add_state('y', fix_initial=True, fix_final=True, rate_source='fd.y_dot', units='m', ref=1000.)
    phase1.add_state('v', fix_initial=True, fix_final=True, rate_source='fd.v_dot', targets='v', lower=vmin, upper=vmax, units='m/s', ref=30.)
    phase1.add_state('energy', fix_initial=True, fix_final=False, rate_source='power', units='W*s', ref=100.)
    phase1.add_control('alpha', targets='alpha', units='deg', lower=0.1, upper=10., ref=10., rate_continuity=True)  # angle of attack
    phase1.add_control('thrust', targets='thrust', units='N', lower=thrust_LB, upper=thrust_UB, ref=thrust_ref, rate_continuity=True)
    phase1.add_parameter('m', val=mass, units='kg', static_target=False)
    phase1.add_parameter('Sref', val=Sref, units='m**2', static_target=True)
    phase1.add_parameter('wing_twist_cp', val=surface['twist_cp'], units='deg', static_target=True, targets=['aero.OAS.wing.twist_cp'])
    phase1.add_parameter('wing_thickness_cp', val=surface['thickness_cp'], units='m', static_target=True, targets=['aero.OAS.wing.thickness_cp'])

    # add design parameters to traj
    traj.add_parameter('m', val=mass, units='kg', static_target=True)   # ignore structural weight in dynamics model for simplicity
    traj.add_parameter('Sref', val=Sref, units='m**2', static_target=True)
    traj.add_parameter('wing_twist_cp', val=surface['twist_cp'], units='deg', static_target=True)
    traj.add_parameter('wing_thickness_cp', val=surface['thickness_cp'], units='m', static_target=True)

    # connect design variables to traj
    prob.model.connect('wing_design_vars.twist_cp', 'traj.parameters:wing_twist_cp')
    prob.model.connect('wing_design_vars.thickness_cp', 'traj.parameters:wing_thickness_cp')
      
    # objective: minimize energy consumption
    phase1.add_objective('energy', loc='final', units='W*s', ref=100.)

    # impose vertical trim (L=W)
    phase1.add_path_constraint('fd.gam_dot', lower=-1e-3, upper=1e-3, units='rad/s')

    # log time series of some variables
    vars_log = ['aero.CL', 'aero.f_drag', 'aero.f_lift', 'aero.OAS.Cl_dist_his', 'aero.OAS.mesh_his', 'aero.OAS.sec_forces_his', 'aero.OAS.stress_his', 'aero.OAS.failure_his']
    phase1.add_timeseries_output(vars_log)

    # --- optimizer ---
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.options['print_results'] = False
    prob.driver.opt_settings['Iterations limit'] = 30000
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.opt_settings['Major optimality tolerance'] = 1e-5
    prob.driver.opt_settings['Verify level'] = -1  # do not check gradient
    prob.driver.opt_settings['Function precision'] = 1e-10
    prob.driver.opt_settings['Hessian full memory'] = 1
    prob.driver.opt_settings['Hessian frequency'] = 100

    prob.driver.declare_coloring(num_full_jacs=3)
    if coloring_file is not None:
        print('Using coloring file:', coloring_file)
        prob.driver.use_fixed_coloring(coloring_file)

    ### prob.model.linear_solver = om.DirectSolver()   # only for serial

    t0 = time_package.time()
    prob.setup(check=False)
    t_setup = time_package.time() - t0

    # --- update solver settings for each OAS system ---
    for subsystem in prob.model.system_iter():
        if subsystem.name == 'coupled':
            subsystem.nonlinear_solver.options['iprint'] = 0  # turn off solver print

            # linear solver for derivatives
            subsystem.linear_solver = om.PETScKrylov(assemble_jac=False, iprint=0, err_on_non_converge=True)
            subsystem.linear_solver.precon = om.LinearRunOnce(iprint=-1)

    # -------------------------
    # set initial guess
    # -------------------------
    # straight flight from (0, 0) to (0, 1000)
    prob.set_val('traj.phase1.t_initial', 0.)
    prob.set_val('traj.phase1.t_duration', 1000 / v_cruise)
    prob.set_val('traj.phase1.states:x', phase1.interpolate(ys=[0, 0], nodes='state_input'), units='m')
    prob.set_val('traj.phase1.states:y', phase1.interpolate(ys=[0, 1000], nodes='state_input'), units='m')
    prob.set_val('traj.phase1.states:v', phase1.interpolate(ys=[v_cruise, v_cruise], nodes='state_input'), units='m/s')
    prob.set_val('traj.phase1.states:energy', phase1.interpolate(ys=[0, 100], nodes='state_input'), units='W*s')
    prob.set_val('traj.phase1.controls:alpha', phase1.interpolate(ys=[3., 3.], nodes='control_input'), units='deg')
    prob.set_val('traj.phase1.controls:thrust', phase1.interpolate(ys=[thrust_ref, thrust_ref], nodes='control_input'), units='N')

    # ------------------------------
    # run model and compute totals
    # ------------------------------
    # manually call final_setup so that we can only timing the analysis
    t0 = time_package.time()
    prob.final_setup()
    t_final_setup = time_package.time() - t0

    t0 = time_package.time()
    prob.run_model()
    t_run_model = time_package.time() - t0

    t0 = time_package.time()
    prob.compute_totals()
    t_compute_totals = time_package.time() - t0

    om.n2(prob, show_browser=False)

    if MPI.COMM_WORLD.rank == 0:
        print('num procs:', MPI.COMM_WORLD.size)
        print('runtimes [s]')
        print('   `setup` :', t_setup)
        print('   `final_setup` :', t_final_setup)
        print('   `run_model` :', t_run_model)
        print('   `compute_totals` :', t_compute_totals)
        print('\nenergy consumption', prob.get_val('traj.phase1.timeseries.states:energy', units='W*s')[-1], 'Ws')
        print('total flight time:', prob.get_val('traj.phase1.timeseries.time', units='s')[-1], 's')
        print('wing design (twist):', prob.get_val('wing_design_vars.twist_cp', units='deg'), 'deg')
        print('wing design (thickness):', prob.get_val('wing_design_vars.thickness_cp', units='m'), 'm')

        # -----
        # plot
        # -----
        t = prob.get_val('traj.phase1.timeseries.time', units='s')
        v = prob.get_val('traj.phase1.timeseries.states:v', units='m/s')
        alpha = prob.get_val('traj.phase1.timeseries.controls:alpha', units='deg')
        thrust = prob.get_val('traj.phase1.timeseries.controls:thrust', units='N')
        energy = prob.get_val('traj.phase1.timeseries.states:energy', units='W*s')

        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        ax[0].plot(t, v)
        ax[0].set_ylabel('v (m/s)')
        ax[1].plot(t, energy)
        ax[1].set_ylabel('energy (Ws)')
        ax[2].plot(t, alpha)
        ax[2].set_ylabel('alpha (deg)')
        ax[3].plot(t, thrust)
        ax[3].set_ylabel('thrust (N)')
        ax[3].set_xlabel('time (s)')
        plt.show()
