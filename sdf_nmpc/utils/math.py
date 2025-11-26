from math import pi, factorial
import numpy as np
import casadi as cs


## rotations (quat, euler, mat, ~angle-axis~)
def quat2rot(q):
    """Compute the rotation matrix associated to a quaternion.
    q       -- quaternion with scalar as first element [qw qx qy qz]
    """
    r11 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    r21 = 2 * (q[1] * q[2] + q[0] * q[3])
    r31 = 2 * (q[1] * q[3] - q[0] * q[2])
    r12 = 2 * (q[1] * q[2] - q[0] * q[3])
    r22 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    r32 = 2 * (q[2] * q[3] + q[0] * q[1])
    r13 = 2 * (q[1] * q[3] + q[0] * q[2])
    r23 = 2 * (q[2] * q[3] - q[0] * q[1])
    r33 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2
    if type(q).__module__ == 'casadi.casadi':
        return cs.vertcat(cs.horzcat(r11, r12, r13), cs.horzcat(r21, r22, r23), cs.horzcat(r31, r32, r33))
    else:
        return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def euler2rot(euler):
    """Compute the rotation matrix associated to euler angles.
    euler   -- euler angles [roll pitch yaw] (Z1Y2X3 convention, see https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
    """
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    if type(euler).__module__ == 'casadi.casadi':
        r11 = cs.cos(pitch) * cs.cos(yaw)
        r21 = cs.cos(pitch) * np.sin(yaw)
        r31 = -cs.sin(pitch)
        r12 = cs.sin(roll) * cs.sin(pitch) * cs.cos(yaw) - cs.cos(roll) * cs.sin(yaw)
        r22 = cs.sin(roll) * cs.sin(pitch) * cs.sin(yaw) + cs.cos(roll) * cs.cos(yaw)
        r32 = cs.sin(roll) * cs.cos(pitch)
        r13 = cs.cos(roll) * cs.sin(pitch) * cs.cos(yaw) + cs.sin(roll) * cs.sin(yaw)
        r23 = cs.cos(roll) * cs.sin(pitch) * cs.sin(yaw) - cs.sin(roll) * cs.cos(yaw)
        r33 = cs.cos(roll) * cs.cos(pitch)
        return cs.vertcat(cs.horzcat(r11, r12, r13), cs.horzcat(r21, r22, r23), cs.horzcat(r31, r32, r33))
    else:
        r11 = np.cos(pitch) * np.cos(yaw)
        r21 = np.cos(pitch) * np.sin(yaw)
        r31 = -np.sin(pitch)
        r12 = np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw)
        r22 = np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw)
        r32 = np.sin(roll) * np.cos(pitch)
        r13 = np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)
        r23 = np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)
        r33 = np.cos(roll) * np.cos(pitch)
        return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])


def quat2euler(q):
    """Compute the euler angles associated to a quaternion.
    q       -- quaternion with scalar as first element [qw qx qy qz]
    """
    if type(q).__module__ == 'casadi.casadi':
        roll = cs.atan2(2 * (q[0]*q[1] + q[2]*q[3]), 1 - 2 * (q[1]*q[1] + q[2]*q[2]))
        pitch = cs.asin(2 * (q[0]*q[2] - q[3]*q[1]))
        yaw = cs.atan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]*q[2] + q[3]*q[3]))
        return cs.vertcat(roll, pitch, yaw)
    else:
        roll = np.arctan2(2 * (q[0]*q[1] + q[2]*q[3]), 1 - 2 * (q[1]*q[1] + q[2]*q[2]))
        pitch = np.arcsin(2 * (q[0]*q[2] - q[3]*q[1]))
        yaw = np.arctan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]*q[2] + q[3]*q[3]))
        return np.array([roll, pitch, yaw])


def quat2yaw(q):
    """Compute the yaw angle associated to a quaternion.
    q       -- quaternion with scalar as first element [qw qx qy qz]
    """
    if type(q).__module__ == 'casadi.casadi':
        yaw = cs.atan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]*q[2] + q[3]*q[3]))
        return yaw
    else:
        yaw = np.arctan2(2 * (q[0]*q[3] + q[1]*q[2]), 1 - 2 * (q[2]*q[2] + q[3]*q[3]))
        return yaw


def rot2euler(R):
    """Compute the euler angles associated to a rotation matrix.
    R       -- rotation matrix
    """
    if type(R).__module__ == 'casadi.casadi':
        roll = cs.atan2(R[2,1], R[2,2])
        pitch = cs.asin(-R[2,0])
        yaw = cs.atan2(R[1,0], R[0,0])
        return cs.vertcat(roll, pitch, yaw)
    else:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arcsin(-R[2,0])
        yaw = np.arctan2(R[1,0], R[0,0])
        return np.array([roll, pitch, yaw])


def rot2quat(R):
    """Compute the euler angles associated to a rotation matrix.
    This is disgusting, clean implementation TODO using https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
    see https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf for reference
    R       -- rotation matrix
    """
    return euler2quat(rot2euler(R))


def euler2quat(euler):
    """Compute the quaternion associated to euler angles.
    euler   -- euler angles [roll pitch yaw] (Z1Y2X3 convention, see https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
    """
    if type(euler).__module__ == 'casadi.casadi':
        cr = cs.cos(euler[0] * 0.5)
        sr = cs.sin(euler[0] * 0.5)
        cp = cs.cos(euler[1] * 0.5)
        sp = cs.sin(euler[1] * 0.5)
        cy = cs.cos(euler[2] * 0.5)
        sy = cs.sin(euler[2] * 0.5)
        return cs.vertcat(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )
    else:
        cr = np.cos(euler[0] * 0.5)
        sr = np.sin(euler[0] * 0.5)
        cp = np.cos(euler[1] * 0.5)
        sp = np.sin(euler[1] * 0.5)
        cy = np.cos(euler[2] * 0.5)
        sy = np.sin(euler[2] * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])


def yaw2quat(yaw):
    """Compute the quaternion associated to a yaw angle.
    yaw     -- yaw (Z1Y2X3 convention, see https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix)
    """
    cr = cp = 1
    sr = sp = 0
    hyaw = yaw * 0.5
    if type(yaw).__module__ == 'casadi.casadi':
        cy = cs.cos(hyaw)
        sy = cs.sin(hyaw)
        return cs.vertcat(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )
    else:
        cy = np.cos(hyaw)
        sy = np.sin(hyaw)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])


def invert(q):
    """Rerturn the inverse quaternion of q."""
    if type(q).__module__ == 'casadi.casadi':
        return cs.vertcat(q[0], -q[1], -q[2], -q[3]) / cs.norm_2(q)
    else:
        return np.array([q[0], -q[1], -q[2], -q[3]]) / np.linalg.norm(q, 2)


def hamilton_prod(q1, q2):
    """Return the Hamilton product of 2 quaternions q1*q2."""
    if type(q1).__module__ == 'casadi.casadi' or type(q2).__module__ == 'casadi.casadi':
        return cs.vertcat(
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        )
    else:
        return np.array([
            q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
            q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
            q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
            q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
        ])


def dist_geo_quat(q1, q2):
    """Computes the squared geodesic distance between two quaternions."""
    q2i = [q2[0], -q2[1], -q2[2], -q2[3]]  # conjugate (invert) quaternion
    q1q2_inv = hamilton_prod(q1,q2i)
    normv = cs.norm_2(q1q2_inv[1:4])
    return cs.if_else(normv < 1e-6, 0, cs.norm_2(2 * q1q2_inv[1:4] * cs.atan2(normv, q1q2_inv[0]) / normv)**2)


def dist_quat(q1, q2):
    """Computes the angular distance between two quaternions."""
    q1n = q1/cs.norm_2(q1)
    q2n = q2/cs.norm_2(q2)
    return 1 - cs.fabs(cs.dot(q1n,q2n))


def deuler_avel_map(euler):
    """Computes the map between euler angles derivatives and angular rates."""
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    if type(euler).__module__ == 'casadi.casadi':
        return cs.vertcat(
            cs.horzcat(1, cs.sin(pitch)*cs.sin(roll)/cs.cos(pitch), cs.sin(pitch)*cs.cos(roll)),
            cs.horzcat(0, cs.cos(roll), -cs.sin(pitch)),
            cs.horzcat(0, cs.sin(roll)/cs.cos(pitch), cs.cos(roll)/cs.cos(pitch))
        )
    else:
        return np.array([
            [1, np.sin(pitch)*np.sin(roll)/np.cos(pitch), np.sin(pitch)*np.cos(roll)],
            [0, np.cos(roll), -np.sin(pitch)],
            [0, np.sin(roll)/np.cos(pitch), np.cos(roll)/np.cos(pitch)],
        ])



## coordinates
def euclidean2spherical(p):
    """Converts a 3D vector from Euclidean to spherical coordinates.
    Uses the 'mathematical' convention for spherical coords (range, azimuth, elevation).
    """
    if type(p).__module__ == 'casadi.casadi':
        range = cs.norm_2(p)
        azimuth = cs.atan2(p[1], p[0])
        elevation = cs.atan2(cs.norm_2(p[:2]), p[2])
        return cs.vertcat(range, azimuth, elevation)
    else:
        range = np.linalg.norm(p, 2)
        azimuth = np.arctan2(p[1], p[0])
        elevation = np.arctan2(np.linalg.norm(p[:2], 2), p[2])
        return np.array([range, azimuth, elevation])


def spherical2euclidean(p):
    """Converts a 3D vector from spherical to Euclidean coordinates.
    Uses the 'mathematical' convention for spherical coords (range, azimuth, elevation).
    """
    if type(p).__module__ == 'casadi.casadi':
        x = p[0] * cs.cos(p[2]) * cs.cos(p[1])
        y = p[0] * cs.cos(p[2]) * cs.sin(p[1])
        z = p[0] * cs.sin(p[2])
        return cs.vertcat(x, y, z)
    else:
        x = p[0] * np.cos(p[2]) * np.cos(p[1])
        y = p[0] * np.cos(p[2]) * np.sin(p[1])
        z = p[0] * np.sin(p[2])
        return np.array([x, y, z])



## misc
def skew_mat(v):
    """Computes the skew matrix of a vector."""
    if type(v).__module__ == 'casadi.casadi':
        return cs.vertcat(
            cs.horzcat(    0, -v[2],  v[1]),
            cs.horzcat( v[2],     0, -v[0]),
            cs.horzcat(-v[1],  v[0],     0),
        )
    else:
        return np.array([
            [    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],     0],
        ])


def rad(angle):
    """Convert an angle from degrees to radians."""
    return angle * pi / 180


def get_vfov(hfov, aspect_ratio, is_spherical):
    """Compute the half vertical fov from the half horizontal fov and aspect ratio."""
    if is_spherical:
        return hfov / aspect_ratio
    else:
        return np.arctan(np.tan(hfov) / aspect_ratio)


def polynomial_3variate(deg, coeffs=None):
    """Construct a 3-variate polynomial casadi expression of arbitrary degree.
    Returns a casadi function evaluating the polynomial parametrized the the coefficients c in x: poly_c(x), and the vector of coefficients.
    If coeffs is None, symbolic coefficients are computed. The returned function is f(x, c) = poly_c(x).
    Else, the returned function is f(x) = poly_c(x).
    """
    nb_coeffs = int(round(factorial(deg + 3) / 6 / factorial(deg)))
    if coeffs is None:
        coeffs = cs.SX.sym('coeffs', nb_coeffs, 1)
    x = cs.SX.sym('x', 3, 1)
    terms = []

    ## iterate over degrees
    for total_degree in range(deg + 1):
        ## generate all combinations of powers (a, b, c) such that a + b + c == total_degree
        for a in range(total_degree + 1):
            for b in range(total_degree + 1 - a):
                c = total_degree - a - b
                term = (x[0]**a) * (x[1]**b) * (x[2]**c)
                terms = cs.vertcat(terms, term)
    poly = cs.sum1(terms * coeffs)

    if type(coeffs).__module__ == 'casadi.casadi':
        poly_fun = cs.Function('poly', [coeffs, x], [poly])
    else:
        poly_fun = cs.Function('poly', [x], [poly])

    return poly_fun, coeffs



def polynomial_3variate(deg, coeffs=None):
    """Construct a 3-variate polynomial casadi expression of arbitrary degree.
    Returns a casadi function evaluating the polynomial parametrized the the coefficients c in x: poly_c(x), and the vector of coefficients.
    If coeffs is None, symbolic coefficients are computed. The returned function is f(x, c) = poly_c(x).
    Else, the returned function is f(x) = poly_c(x).
    """
    nb_coeffs = int(round(factorial(deg + 3) / 6 / factorial(deg)))
    if coeffs is None:
        coeffs = cs.SX.sym('coeffs', nb_coeffs, 1)
    x = cs.SX.sym('x', 3, 1)
    terms = []

    ## iterate over degrees
    for total_degree in range(deg + 1):
        ## generate all combinations of powers (a, b, c) such that a + b + c == total_degree
        for a in range(total_degree + 1):
            for b in range(total_degree + 1 - a):
                c = total_degree - a - b
                term = (x[0]**a) * (x[1]**b) * (x[2]**c)
                terms = cs.vertcat(terms, term)
    poly = cs.sum1(terms * coeffs)

    if type(coeffs).__module__ == 'casadi.casadi':
        poly_fun = cs.Function('poly', [coeffs, x], [poly])
    else:
        poly_fun = cs.Function('poly', [x], [poly])

    return poly_fun, coeffs



## GTMR allocation matrices
def axis_rot(axis, angle):
    """Compute the rotation matrix around a given axis (x, y, or z), angle in rad."""
    if axis == 'x':
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def GTMRP_props(n, l, alpha, beta, com=[0, 0, 0], alpha0=-1, s0=1):
    """Compute positions, orientations and spinning signs of propelers in a Generically Tilted Multi-Rotor Platform.
    n       -- number of actuators
    l       -- distance from propellers to CoM
    alpha   -- alpha tilting angles in absolute value (deg)
    beta    -- beta tilting angles in absolute value (deg)
    com     -- position of the geometrical center of props wrt CoM
    alpha0  -- sign of alpha tilting angle for the first propeller
    s0      -- rotation direction for first props (assumes alternating pattern)
    """
    alpha = rad(alpha)
    beta = rad(beta)
    R = [axis_rot('z', i * (pi / (n / 2))) @ axis_rot('y', beta) @ axis_rot('x', alpha0 * (-1) ** i * alpha) for i in range(n)]
    p = [l * axis_rot('z', i * (pi / (n / 2))) @ [1, 0, 0] + np.array(com).T for i in range(n)]
    signs = [(-1)**k * s0 for k in range(n)]
    return p, R, signs


def GTMRP_matrix(R, p, signs, c_f, c_t):
    """Compute allocation matrix for a Generically Tilted Multi-Rotor Platform.
    R       -- list of (3x3) orientation matrices of the n propellers
    p       -- list of (1x3) position vectors of the n propellers
    signs   -- list of rotation direction for props (-1:counter-clockwise, 1:clockwise)
    c_f     -- propellers' force coefficient
    c_t     -- propellers' torque coefficient
    """
    Rz = [r @ [0,0,1] for r in R]
    G_f = np.column_stack(Rz)
    G_t = np.column_stack([np.cross(p[i].flatten(), Rz[i].flatten(), 0, 0, 0) + c_t[i] / c_f[i] * signs[i] * Rz[i] for i in range(len(R))])
    return G_f, G_t


def allocation(n, l, alpha, beta, c_f, c_t, com=[0, 0, 0], alpha0=-1):
    """Compute the force and torque allocation matrices.
    n       -- number of actuators
    l       -- distance from propellers to CoM
    aplha   -- alpha tilting angles in absolute value (deg)
    beta    -- beta tilting angles in absolute value (deg)
    c_f     -- propellers' force coefficient
    c_t     -- propellers' torque coefficient
    com     -- 3D position offset between geometrical center of props and CoM
    # config  -- 'x' or '+' ; rotor configuration (describes if the body x axis is aligned with first prop)
    signs   -- list of rotation direction for props (-1:counter-clockwise, 1:clockwise)
                or single int (-1/1) depicting the rotation direction of the first prop
    alpha0  -- sign of tilting for 1st prop (1 or -1)
    """
    if not isinstance(c_f, list):
        c_f = [c_f for i in range(n)]
    if not isinstance(c_t, list):
        c_t = [c_t for i in range(n)]
    if not isinstance(signs, list):
        signs = [signs * (-1)**i for i in range(n)]
    p, R, signs = GTMRP_props(n, l, alpha, beta, com, alpha0)
    GF, GT = GTMRP_matrix(R, p, signs, c_f, c_t)

    return GF, GT
