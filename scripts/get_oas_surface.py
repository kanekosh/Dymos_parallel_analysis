import numpy as np
from openaerostruct.geometry.utils import generate_mesh, taper, sweep


def get_OAS_surface(wing_area, span, num_y=15, num_x=5):
    """
    creates an OAS surface dict for a tapered rectangle wing

    Parameters
    ----------
    wing_area : float
        wing reference area, m**2
    span : float
        wing span, m
    num_y : int
        number of spanwise vortices
    num_x : int
        number of chordwise vortices

    Returns
    -------
    surface : dict
        OAS surface dict
    """

    taper_ratio = 0.5
    sweep_angle = 5.   # deg
    root_chord = 2 * wing_area / span / (1 + taper_ratio)

    n_twist_points = 5

    # Create a dictionary to store options about the surface
    mesh_dict = {"num_y": num_y, "num_x": num_x, "wing_type": "rect", "symmetry": True, "span": span, "root_chord": root_chord}

    mesh = generate_mesh(mesh_dict)
    # apply taper and sweep
    taper(mesh, taper_ratio, symmetry=True)
    sweep(mesh, sweep_angle, symmetry=True)

    surface = {
        # Wing definition
        "name": "wing",  # name of the surface
        "symmetry": True,  # if true, model one half of wing
        # reflected across the plane y = 0
        "S_ref_type": "wetted",  # how we compute the wing area,
        # can be 'wetted' or 'projected'
        "fem_model_type": "tube",
        "thickness_cp": np.array([0.003]),
        "radius_cp": np.array([0.03]),
        "twist_cp": np.zeros(n_twist_points),
        "mesh": mesh,
        # "taper" : taper_ratio,
        "sweep" : sweep_angle,
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

    return surface, wing_area
