import numpy as np
import casadi as cs
from acados_template import AcadosModel
import yaml


class BaseModel:
    """Wrapper around the AcadosModel class, that contains all the necessary extra info for NMPC."""
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.model_ac = None

        ## parameters
        self.g = 9.81

        ## vectors of externally added constraints and evaluation variables
        self.extra_W = np.array([])
        self.extra_WN = np.array([])
        self.eval_vec = np.array([])

        ## problem dimensions
        self.nx = 0  # number of states
        self.nu = 0  # number of outputs
        # self.nz = 0  # number of algebraic states  # unused
        self.np = 0  # number of parameters
        self.ny = 0  # number of outputs
        self.nyN = 0  # number of terminal outputs
        self.nbx = 0  # number of state constraints
        self.nbxN = 0  # number of terminal state constraints
        self.nsbx = 0  # number of soft state constraints
        self.nsbxN = 0  # number of soft terminal state constraints
        self.nh = 0  # number of general constraints
        self.nhN = 0  # number of terminal general constraints
        self.nsh = 0  # number of soft general constraints
        self.nshN = 0  # number of soft terminal general constraints
        self.gen_symbols()

        ## input constraints
        self.lbu = np.array([])
        self.ubu = np.array([])

        ## state constraints
        self.idxbx = np.array([])  # indices of constrained variables among x
        self.idxsbx = np.array([])  # indices of softened constraints among idxbx
        self.lbx = np.array([])
        self.ubx = np.array([])

        self.idxbxN = np.array([])  # indices of constrained variables among xN
        self.idxsbxN = np.array([])  # indices of softened constraints among idxbxN
        self.lbxN = np.array([])
        self.ubxN = np.array([])

        ## general constraints
        self.idxsh = np.array([])  # indices of softened constraints among general constraints
        self.lh = np.array([])
        self.uh = np.array([])

        self.idxshN = np.array([])  # indices of softened constraints among terminal general constraints
        self.lhN = np.array([])
        self.uhN = np.array([])

        ## slack variable costs
        self.slack_x_w_L1_stage = np.array([])
        self.slack_x_w_L2_stage = np.array([])
        self.slack_x_w_L1_term = np.array([])
        self.slack_x_w_L2_term = np.array([])
        self.slack_h_w_L1_stage = np.array([])
        self.slack_h_w_L2_stage = np.array([])
        self.slack_h_w_L1_term = np.array([])
        self.slack_h_w_L2_term = np.array([])
        self.zl = np.array([])
        self.Zl = np.array([])
        self.zu = np.array([])
        self.Zu = np.array([])

        self.zlN = np.array([])
        self.ZlN = np.array([])
        self.zuN = np.array([])
        self.ZuN = np.array([])


    def gen_symbols(self):
        """Convenience function to generate all symbolic vectors. Should be called after all the model sizes are set."""
        self.x = cs.MX.sym('x', self.nx, 1)
        self.dx = cs.MX.sym('dx', self.nx, 1)
        self.u = cs.MX.sym('u', self.nu, 1)
        # self.z = cs.MX.sym('z', self.nz, 1)  # unused
        self.p = cs.MX.sym('p', self.np, 1)
        self.h = cs.MX.sym('h', self.nh, 1)
        self.hN = cs.MX.sym('hN', self.nhN, 1)
        self.eval = cs.Function('eval', [self.x, self.u, self.p], [self.eval_vec])


    def gen_acados_model(self):
        """Generate the Acados model and fill all fields from self. Should be called after the model is fully defined."""
        self.model_ac = AcadosModel()
        self.model_ac.name = self.name
        self.model_ac.f_impl_expr = self.f_impl
        self.model_ac.f_expl_expr = self.f_expl
        self.model_ac.x = self.x
        self.model_ac.xdot = self.dx
        self.model_ac.u = self.u
        # self.model_ac.z = self.z  # unused
        self.model_ac.p = self.p
        self.model_ac.con_h_expr = self.h
        self.model_ac.con_h_expr_e = self.hN
        self.model_ac.cost_y_expr = self.y
        self.model_ac.cost_y_expr_e = self.yN


    def formate_ref(self, p_des, yaw_des, v_ref, wz_ref, Wp, Wq, Wv, Ww, Wa):
        """Returns a pair (reference y, weights W) of size self.ny depending on the specific model.
        Must be implemented independently in each model instance class.
        """
        raise NotImplementedError


    def add_eval(self, function, args):
        """Add an evaluation variable to be returned by the eval() function.
        function    -- returns the value to be evaluated
        args        -- takes as input (x, u, p) and returns the subset that is required as input for function
        """
        self.eval_vec = cs.vertcat(self.eval_vec, function(args(self.x, self.u, self.p)))
        self.eval = cs.Function('eval', [self.x, self.u, self.p], [self.eval_vec])


    def add_cost_stage(self, function, args, weight):
        """Add a term to the running cost function."""
        self.extra_W = np.append(self.extra_W, weight)
        self.y = cs.vertcat(self.y, function(args(self.x, self.u, self.p)))
        self.ny += 1


    def add_cost_term(self, function, args, weight):
        """Add a term to the terminal cost function."""
        self.extra_WN = np.append(self.extra_WN, weight)
        self.yN = cs.vertcat(self.yN, function(args(self.x, self.u, self.p)))
        self.nyN += 1


    def add_const_stage(self, function, args, bounds, slack_weights=None):
        """Add a stage constraint.
        If slack_weights is not None, make it soft constraint with those weights (resp. on gradient and Hessian).
        """
        self.nh += 1
        self.lh = np.append(self.lh, [bounds[0]])
        self.uh = np.append(self.uh, [bounds[1]])
        self.h = cs.vertcat(self.h, function(args(self.x, self.u, self.p)))
        if slack_weights:
            self.idxsh = np.append(self.idxsh, [self.nh-1])
            self.slack_h_w_L1_stage = np.append(self.slack_h_w_L1_stage, [slack_weights[0]])
            self.slack_h_w_L2_stage = np.append(self.slack_h_w_L2_stage, [slack_weights[1]])
            self.nsh += 1

    def add_const_term(self, function, args, bounds, slack_weights=None):
        """Add a terminal constraint.
        If slack_weights is not None, make it soft constraint with those weights (resp. on gradient and Hessian).
        """
        self.nhN += 1
        self.lhN = np.append(self.lhN, [bounds[0]])
        self.uhN = np.append(self.uhN, [bounds[1]])
        self.hN = cs.vertcat(self.hN, function(args(self.x, self.u, self.p)))
        if slack_weights:
            self.idxshN = np.append(self.idxshN, [self.nhN-1])
            self.slack_h_w_L1_term = np.append(self.slack_h_w_L1_term, [slack_weights[0]])
            self.slack_h_w_L2_term = np.append(self.slack_h_w_L2_term, [slack_weights[1]])
            self.nshN += 1
