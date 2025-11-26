import numpy as np
import scipy as sp
import sympy


def symbolic_r_tilde():
    ## system params
    m = sympy.symbols('m')
    g = sympy.symbols('g')
    dt = sympy.symbols('dt')

    ## state and input vector
    vx, vy, vz = sympy.symbols('v_x, v_y, v_z')
    phi, theta = sympy.symbols('phi, theta')
    psi = 0
    T = sympy.symbols('T')

    v = sympy.Matrix([vx, vy, vz])
    Re3 = sympy.Matrix([
        sympy.cos(psi)*sympy.sin(theta)*sympy.cos(phi) + sympy.sin(psi)*sympy.sin(phi),
        sympy.sin(psi)*sympy.sin(theta)*sympy.cos(phi) - sympy.cos(psi)*sympy.sin(phi),
        sympy.cos(theta)*sympy.cos(phi),
    ])

    U = sympy.Matrix([T - g, phi, theta])

    ## cost matrices
    r_1, r_2, r_3 = sympy.symbols('r_1, r_2, r_3')
    R_ = sympy.Matrix(np.diag([r_1, r_2, r_3]))
    r_tilde = sympy.symbols('r_tilde')
    R_tilde = r_tilde * sympy.Matrix(np.eye(3))

    ## cost function
    input_cost = U.T @ R_ @ U
    U_tilde  = dt * (T * Re3 - sympy.Matrix([0, 0, g]))
    input_cost_bound = U_tilde.T @ R_tilde @ U_tilde

    ## find expression of smallest r_tilde
    inequality = input_cost_bound - input_cost
    solution = sympy.solve(inequality[0], r_tilde)
    return min(solution)


def get_r_tilde_max(cfg):
    ## get symbol expression
    r_tilde = symbolic_r_tilde()

    ## system values
    m = cfg.robot.mass
    g = 9.81
    dt = cfg.mpc.T / cfg.mpc.N
    r_1, r_2, r_3 = cfg.mpc.weights.acc, cfg.mpc.weights.att[0], cfg.mpc.weights.att[1]
    T_range = [0, cfg.robot.limits.gamma]
    phi_range = [-cfg.robot.limits.roll, cfg.robot.limits.roll]
    theta_range = [-cfg.robot.limits.pitch, cfg.robot.limits.pitch]

    ## solve for max r_tilde over u
    def objective(x):
        T, phi, theta = x
        return - r_tilde.subs({'phi': phi, 'theta': theta, 'T': T, 'g': g, 'm': m, 'dt': dt, 'r_1': r_1, 'r_2': r_2, 'r_3': r_3}).evalf()

    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] - T_range[0]},  # T >= T_min
        {'type': 'ineq', 'fun': lambda x: T_range[1] - x[0]},   # T <= T_max
        {'type': 'ineq', 'fun': lambda x: x[1] - phi_range[0]},  # phi >= phi_min
        {'type': 'ineq', 'fun': lambda x: phi_range[1] - x[1]},  # phi <= phi_max
        {'type': 'ineq', 'fun': lambda x: x[2] - theta_range[0]},  # theta >= theta_min
        {'type': 'ineq', 'fun': lambda x: theta_range[1] - x[2]},  # theta <= theta_max
    ]

    # x0 = [(T_range[0] + T_range[1]) / 2, (phi_range[0] + phi_range[1]) / 2, (theta_range[0] + theta_range[1]) / 2]
    x0 = [np.random.uniform(*T_range), np.random.uniform(*phi_range), np.random.uniform(*theta_range)]
    sol = sp.optimize.minimize(objective, x0, constraints=constraints, method='SLSQP')

    return -sol.fun
