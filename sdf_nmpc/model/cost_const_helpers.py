import numpy as np
import casadi as cs
from ..utils.math import *


def add_fov_const_normals(model, h_const, v_const, slack=None):
    """Creates a set of slack or hard constraints to constrain the motion to occur inside the sensor fov, both horizontally and vertically.
    The constrains are expressed as 2 half-space constraints for each direction.
    This DOES NOT handles fov > 180°!
    For implementation reasons, Acados expects an upper bound on the constrains.
    This is set such that the position cannot be more than dmax away from the plane, which trivially is encloses the whole fov.
    Only the lower bound of the constrains can become active.
    h_const     -- enable constraint on horizontal fov
    v_const     -- enable constraint on vertical fov
    slack       -- None if hard constraint, or two-element list for linear and quadratic weight on slack variables.
    """
    def hfov_const_left(arg):
        normal = cs.vertcat(cs.tan(model.cfg.sensor.hfov), -1, 0) / cs.sqrt(cs.tan(model.cfg.sensor.hfov) ** 2 + 1)
        return arg[0] * cs.dot(normal, arg[1])
    def hfov_const_right(arg):
        normal = cs.vertcat(-cs.tan(-model.cfg.sensor.hfov), 1, 0) / cs.sqrt(cs.tan(-model.cfg.sensor.hfov) ** 2 + 1)
        return arg[0] * cs.dot(normal, arg[1])
    def vfov_const_up(arg):
        normal = cs.vertcat(cs.tan(model.cfg.sensor.hfov), 0, -1) / cs.sqrt(cs.tan(model.cfg.sensor.hfov) ** 2 + 1)
        return arg[0] * cs.dot(normal, arg[1])
    def vfov_const_down(arg):
        normal = cs.vertcat(-cs.tan(-model.cfg.sensor.hfov), 0, 1) / cs.sqrt(cs.tan(-model.cfg.sensor.hfov) ** 2 + 1)
        return arg[0] * cs.dot(normal, arg[1])
    def args_Co_p_C(x,u,p):
        W_R_Co = p[model.cfg.mpc.p_idx.W_R_Co].reshape((3, 3)).T  # transpose because casadi is column-major
        W_p_Co = p[model.cfg.mpc.p_idx.W_p_Co]
        W_p_B = x[:3]
        Co_p_C = W_R_Co.T @ (W_p_B - W_p_Co) + model.cfg.sensor.B_R_C.T @ model.cfg.sensor.B_p_C
        return p[model.cfg.mpc.p_idx.flag], Co_p_C

    hfov_lim = model.cfg.sensor.hfov * model.cfg.mpc.fov_ratio
    vfov_lim = model.cfg.sensor.vfov * model.cfg.mpc.fov_ratio

    if h_const:
        model.add_const_stage(hfov_const_left, args_Co_p_C, [0, model.cfg.sensor.dmax], slack)
        model.add_const_stage(hfov_const_right, args_Co_p_C, [0, model.cfg.sensor.dmax], slack)
    if v_const:
        model.add_const_stage(vfov_const_up, args_Co_p_C, [0, model.cfg.sensor.dmax], slack)
        model.add_const_stage(vfov_const_down, args_Co_p_C, [0, model.cfg.sensor.dmax], slack)



def add_fov_const_trigo(model, h_const, v_const, slack=None):
    """Creates a set of slack or hard constraints to constrain the motion to occur inside the sensor fov, both horizontally and vertically.
    The constraints are expressed on the spherical coordinates of the evaluated positions.
    The constraints are singular in [0,0,0], thus the small +x offset is added on the evaluated position (defined in config file).
    h_const     -- enable constraint on horizontal fov
    v_const     -- enable constraint on vertical fov
    slack       -- None if hard constraint, or two-element list for linear and quadratic weight on slack variables.
    """
    def hfov_const(arg):
        return arg[0] * cs.atan2(arg[1][1], arg[1][0])
    def vfov_const(arg):
        return arg[0] * cs.atan2(arg[1][2], cs.norm_2(arg[1][:2]))
    def args_Co_p_C(x,u,p):
        W_R_Co = p[model.cfg.mpc.p_idx.W_R_Co].reshape((3, 3)).T  # transpose because casadi is column-major
        W_p_Co = p[model.cfg.mpc.p_idx.W_p_Co]
        W_p_B = x[:3]
        Co_p_C = W_R_Co.T @ (W_p_B - W_p_Co) + model.cfg.sensor.B_R_C.T @ model.cfg.sensor.B_p_C
        return p[model.cfg.mpc.p_idx.flag], Co_p_C + cs.vertcat(model.cfg.mpc.fov_const_offset, 0, 0)

    hfov_lim = model.cfg.sensor.hfov * model.cfg.mpc.fov_ratio
    vfov_lim = model.cfg.sensor.vfov * model.cfg.mpc.fov_ratio

    if h_const:
        model.add_const_stage(hfov_const, args_Co_p_C, np.array([-hfov_lim, hfov_lim]), slack)
        model.add_const_term(hfov_const, args_Co_p_C, np.array([-hfov_lim, hfov_lim]), slack)
    if v_const:
        model.add_const_stage(vfov_const, args_Co_p_C, np.array([-vfov_lim, vfov_lim]), slack)
        model.add_const_term(vfov_const, args_Co_p_C, np.array([-vfov_lim, vfov_lim]), slack)



def add_vel_const(model, stage, term, slack=None):
    """Creates a set of soft or hard constraints on velocities (body frame)."""
    bounds_v = np.array([model.cfg.robot.limits.vx, model.cfg.robot.limits.vy, model.cfg.robot.limits.vz])
    if stage:
        model.idxbx = np.append(model.idxbx, np.array([7, 8, 9]))
        model.nbx += 3
        if slack:
            model.nsbx += 3
            model.idxsbx = np.append(model.idxsbx, np.arange(len(model.idxbx) - 3, len(model.idxbx)))
            model.slack_x_w_L1_stage = np.append(model.slack_x_w_L1_stage, [slack[0]] * 3)
            model.slack_x_w_L2_stage = np.append(model.slack_x_w_L2_stage, [slack[1]] * 3)
        model.lbx = np.append(model.lbx, -bounds_v)
        model.ubx = np.append(model.ubx, bounds_v)

    if term:
        model.idxbxN = np.append(model.idxbxN, np.array([7, 8, 9]))
        model.nbxN += 3
        if slack:
            model.nsbxN += 3
            model.idxsbxN = np.append(model.idxsbxN, np.arange(len(model.idxbxN) - 3, len(model.idxbxN)))
            model.slack_x_w_L1_term = np.append(model.slack_x_w_L1_term, [slack[0]] * 3)
            model.slack_x_w_L2_term = np.append(model.slack_x_w_L2_term, [slack[1]] * 3)
        model.lbxN = np.append(model.lbxN, -bounds_v)
        model.ubxN = np.append(model.ubxN, bounds_v)


def add_roll_const(model, slack=None):
    """Creates a soft or hard constraint on roll."""
    model.add_const_stage(lambda x: quat2euler(x)[0], lambda x,u,p: x[3:7], [-model.cfg.robot.limits.roll, model.cfg.robot.limits.roll], slack)
    model.add_const_term(lambda x: quat2euler(x)[0], lambda x,u,p: x[3:7], [-model.cfg.robot.limits.roll, model.cfg.robot.limits.roll], slack)


def add_pitch_const(model, slack=None):
    """Creates a soft or hard constraint on pitch."""
    model.add_const_stage(lambda x: quat2euler(x)[1], lambda x,u,p: x[3:7], [-model.cfg.robot.limits.pitch, model.cfg.robot.limits.pitch], slack)
    model.add_const_term(lambda x: quat2euler(x)[1], lambda x,u,p: x[3:7], [-model.cfg.robot.limits.pitch, model.cfg.robot.limits.pitch], slack)


def add_yxvel_cost(model, w_y, w_z):
    """Creates a cost on vy and vy (body frame) with resp. weights w_y and w_z."""
    model.add_cost_stage(lambda x: x, lambda x,u,p: x[8], w_y)
    model.add_cost_stage(lambda x: x, lambda x,u,p: x[9], w_z)
