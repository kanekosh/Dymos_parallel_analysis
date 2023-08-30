import numpy as np
import openmdao.api as om
from aero_oas import AeroForceOAS

"""
Aircraft ODE model
"""

class Aircraft2DODE(om.Group):
    """
    ODE model of a 2D point-mass aircraft

    Inputs:
        time
        states = (m, x, y, v)
            m   = aircraft mass (assumed constant for now)
            v   = speed
        controls = (alpha, theta, thrust)
            alpha  = angle of attack
            thrust = thrust
        Sref = wing ref area (fixed parameter). Not necessary if using OAS

    Outputs:
        state rate of change (fd.x_dot, fd.y_dot, fd.v_dot, fd.z_dot, fd.gam_dot, fd.chi_dot)
        power (prop power required at each node)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('OAS_surface', types=dict, default={}, desc='Surface dict for OAS')

    def setup(self):
        nn = self.options['num_nodes']

        # --- aerodynamic model ---
        # input: m, v, alpha, theta, S_ref
        # outputs: f_lift, f_drag
        self.add_subsystem(name='aero', subsys=AeroForceOAS(num_nodes=nn, OAS_surface=self.options['OAS_surface']), promotes_inputs=['m', 'v', 'alpha', 'S_ref'])
        self.set_input_defaults('aero.theta', np.zeros(nn), units='deg')   # set 0 deg bank angle

        # --- flight dynamics ---
        # input: m, v, gam, chi, alpha, theta, L, D, thrust
        # output: m_dot, x_dot, y_dot, z_dot, v_dot, gam_dot, chi_dot
        self.add_subsystem(name='fd',
                           subsys=AircraftDynamics(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'alpha', ('T', 'thrust')])
        self.connect('aero.f_drag', 'fd.D')
        self.connect('aero.f_lift', 'fd.L')
        self.set_input_defaults('fd.gam', np.zeros(nn), units='deg')   # 0 deg flight path angle (level flight)
        self.set_input_defaults('fd.chi', 90 * np.ones(nn), units='deg')   # 90 deg heading angle (south-to-north straight flight)
        self.set_input_defaults('fd.theta', np.zeros(nn), units='deg')   # 0 deg bank angle

        # --- propulsion model ---
        # compute power required to generate the thrust based on actuator-disk momentum theory
        # first, overwrite negative thrust (= brake) with zero thrust
        self.add_subsystem('thrust_positive_comp', SoftMaximum(num_nodes=nn, alpha=30, units='N'), promotes_inputs=[('x1', 'thrust')])
        self.set_input_defaults('thrust_positive_comp.x2', np.ones(nn) * 0.0001, units='N')
         
        rotor_radius = 0.15
        rotor_disk_area = np.pi * rotor_radius**2
        power_comp = om.ExecComp('power_req = (thrust**3 / (2 * rho * area))**0.5 / power_eff',
                                 power_req={'shape': (nn,), 'units': 'W'},
                                 thrust={'shape': (nn,), 'units': 'N'},
                                 rho={'val': 1.225 * np.ones(nn,), 'units': 'kg/m**3'},
                                 area={'val': rotor_disk_area, 'units': 'm**2'},
                                 power_eff={'val': 0.8, 'units': None},
                                 has_diag_partials=True)
        self.add_subsystem('power_comp', power_comp, promotes=[('power_req', 'power')])
        self.connect('thrust_positive_comp.softmax', 'power_comp.thrust')


class AircraftDynamics(om.ExplicitComponent):
    """
    2D point-mass equation of motion for aircraft.
    Flight pass angle gam is computed, which should be driven to zero by optimizer. z is not computed.
    Based on (but modified): Dai and Cochran 2009, Journal of Aircraft, Vol. 46, No. 2
    """
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # control inputs
        self.add_input(name='alpha', val=np.zeros(nn), units='rad', desc='(geometric) angle of attack')
        self.add_input(name='theta', val=np.zeros(nn), units='rad', desc='bank angle')
        self.add_input(name='T', val=np.zeros(nn), units='N', desc='thrust aligned in body axis')
        # state inputs
        self.add_input(name='m', val=np.ones(nn), units='kg', desc='aircraft mass')
        self.add_input(name='v', val=np.ones(nn), units='m/s', desc='flight speed (Earth-fixed frame)')
        self.add_input(name='gam', val=np.zeros(nn), units='rad', desc='flight path angle')
        self.add_input(name='chi', val=np.zeros(nn), units='rad', desc='heading angle')
        # forces computed given the controls and states
        self.add_input(name='L', val=np.zeros(nn), units='N', desc='lift')
        self.add_input(name='D', val=np.zeros(nn), units='N', desc='drag')
        
        # outputs: state rate of change
        self.add_output(name='m_dot', val=np.zeros(nn), units='kg/s', desc='rate of change of mass')
        self.add_output(name='x_dot', val=np.zeros(nn), units='m/s', desc='rate of change of x location')
        self.add_output(name='y_dot', val=np.zeros(nn), units='m/s', desc='rate of change of y location')
        self.add_output(name='z_dot', val=np.zeros(nn), units='m/s', desc='rate of change of altitude')
        self.add_output(name='v_dot', val=np.zeros(nn), units='m/s**2', desc='rate of change of flight speed',)
        self.add_output(name='gam_dot', val=np.zeros(nn), units='rad/s', desc='rate of change of flight path angle')
        self.add_output(name='chi_dot', val=np.zeros(nn), units='rad/s', desc='rate of change of heading angle')
        
        ar = np.arange(nn)
        self.declare_partials('x_dot', ['v', 'gam', 'chi'], rows=ar, cols=ar)
        self.declare_partials('y_dot', ['v', 'gam', 'chi'], rows=ar, cols=ar)
        self.declare_partials('z_dot', ['v', 'gam'], rows=ar, cols=ar)
        self.declare_partials('v_dot', ['T', 'D', 'm', 'alpha', 'gam'], rows=ar, cols=ar)
        self.declare_partials('gam_dot', ['T', 'L', 'm', 'alpha', 'gam', 'theta', 'v'], rows=ar, cols=ar)
        self.declare_partials('chi_dot', ['L', 'm', 'gam', 'theta', 'v'], rows=ar, cols=ar)
        
    def compute(self, inputs, outputs):
        g = 9.80665
        alpha = inputs['alpha']
        theta = inputs['theta']
        m = inputs['m']
        v = inputs['v']
        gam = inputs['gam']
        chi = inputs['chi']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']

        # constant mass for glider
        outputs['m_dot'] = 0.
        outputs['v_dot'] = (T * np.cos(alpha) - D) / m - g * np.sin(gam)
        outputs['gam_dot'] = (T * np.sin(alpha) + L * np.cos(theta)) / (m * v) - (g / v) * np.cos(gam)
        outputs['chi_dot'] = L * np.sin(theta) / (m * v * np.cos(gam))
        outputs['x_dot'] = v * np.cos(gam) * np.cos(chi)
        outputs['y_dot'] = v * np.cos(gam) * np.sin(chi)
        outputs['z_dot'] = v * np.sin(gam)
        
    def compute_partials(self, inputs, J):
        g = 9.80665
        alpha = inputs['alpha']
        theta = inputs['theta']
        m = inputs['m']
        v = inputs['v']
        gam = inputs['gam']
        chi = inputs['chi']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']

        J['x_dot', 'v'] = np.cos(gam) * np.cos(chi)
        J['x_dot', 'gam'] = -v * np.sin(gam) * np.cos(chi)
        J['x_dot', 'chi'] = -v * np.cos(gam) * np.sin(chi)
        J['y_dot', 'v'] = np.cos(gam) * np.sin(chi)
        J['y_dot', 'gam'] = -v * np.sin(gam) * np.sin(chi)
        J['y_dot', 'chi'] = v * np.cos(gam) * np.cos(chi)
        J['z_dot', 'v'] = np.sin(gam)
        J['z_dot', 'gam'] = v * np.cos(gam)

        J['v_dot', 'T'] = np.cos(alpha) / m
        J['v_dot', 'D'] = -1. / m
        J['v_dot', 'm'] = (D - T * np.cos(alpha)) / (m**2)
        J['v_dot', 'gam'] = -g * np.cos(gam)
        J['v_dot', 'alpha'] = -T * np.sin(alpha) / m

        J['gam_dot', 'T'] = np.sin(alpha) / (m * v)
        J['gam_dot', 'L'] = np.cos(theta) / (m * v)
        J['gam_dot', 'm'] = -(L * np.cos(theta) + T * np.sin(alpha)) / (m * m * v)
        J['gam_dot', 'gam'] = g * np.sin(gam) / v
        J['gam_dot', 'alpha'] = T * np.cos(alpha) / (m * v)
        J['gam_dot', 'theta'] = -L * np.sin(theta) / (m * v)
        J['gam_dot', 'v'] = g * np.cos(gam) / v**2 - (L * np.cos(theta) + T * np.sin(alpha)) / (v * m * v)
        
        J['chi_dot', 'L'] = np.sin(theta) / (m * v * np.cos(gam))
        J['chi_dot', 'm'] = -L * np.sin(theta) / (m**2 * v * np.cos(gam))
        J['chi_dot', 'gam'] = L * np.sin(theta) / (m * v) * np.tan(gam) / np.cos(gam)
        J['chi_dot', 'theta'] = L * np.cos(theta) / (m * v * np.cos(gam))
        J['chi_dot', 'v'] = -L * np.sin(theta) / (m * v**2 * np.cos(gam))


class SoftMaximum(om.ExplicitComponent):
    # soft maximum between vector a and b
    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('alpha', default=100.)   # nonlinearness factor, If this is too high, optimization fails.
        self.options.declare('units', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']
        self.add_input('x1', shape=(nn,), units=units)   # original BEMT output
        self.add_input('x2', shape=(nn,), units=units)   # linear model
        self.add_output('softmax', shape=(nn,), units=units)
        self.declare_partials('softmax', ['*'], rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        alp = self.options['alpha']
        x1 = inputs['x1']
        x2 = inputs['x2']

        # soft minimum
        outputs['softmax'] = np.log(np.exp(alp * x1) + np.exp(alp * x2)) / alp

    def compute_partials(self, inputs, partials):
        alp = self.options['alpha']
        x1 = inputs['x1']
        x2 = inputs['x2']

        # softmin
        exp1 = np.exp(alp * x1)
        exp2 = np.exp(alp * x2)
        partials['softmax', 'x1'] = exp1 / (exp1 + exp2)
        partials['softmax', 'x2'] = exp2 / (exp1 + exp2)
