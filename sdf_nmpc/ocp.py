import os
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from . import cache_dir
from .utils.config import Config
from .gen_model import get_model_from_cfg


def build_solver(cfg_file):
    cfg = Config(cfg_file)

    model, *_ = get_model_from_cfg(cfg)
    Ocp(model, build=True)


class Ocp:
    def __init__(self, model, build=False):
        self.model = model
        self.T = self.model.cfg.mpc.T
        self.N = self.model.cfg.mpc.N
        if self.model.cfg.mpc.uniform_dt:
            self.shooting_nodes = np.linspace(0, self.T, self.N + 1)
        else:
            n_short = self.model.cfg.mpc.nb_short_nodes
            dt_short = self.model.cfg.mpc.control_loop_time * 1e-3
            self.shooting_nodes = np.hstack([np.linspace(0, dt_short * (n_short-1), n_short), np.linspace(dt_short * n_short, self.T, self.N - n_short + 1)])
        self.dt = np.diff(self.shooting_nodes)
        self.json = os.path.join(cache_dir(), 'codegen', self.model.cfg.name, 'solver.json')
        self.codegen_dir = os.path.join(cache_dir(), 'codegen', self.model.cfg.name, 'generated_c')

        ## acados ocp
        self.ocp_ac = AcadosOcp()
        self.ocp_ac.code_export_directory = self.codegen_dir
        self.ocp_ac.model = model.model_ac

        self.ocp_ac.dims.nx = model.nx
        self.ocp_ac.dims.nu = model.nu
        self.ocp_ac.dims.np = model.np
        self.ocp_ac.dims.ny = model.ny
        self.ocp_ac.dims.ny_e = model.nyN
        self.ocp_ac.dims.nbx = model.nbx
        self.ocp_ac.dims.nbx_e = model.nbxN
        self.ocp_ac.dims.nsbx = model.nsbx
        self.ocp_ac.dims.nsbx_e = model.nsbxN
        self.ocp_ac.dims.nbu = model.nu
        self.ocp_ac.dims.nh = model.nh
        self.ocp_ac.dims.nh_e = model.nhN
        self.ocp_ac.dims.nsh = model.nsh
        self.ocp_ac.dims.nsh_e = model.nshN
        self.ocp_ac.parameter_values = np.zeros(model.np)
        self.ocp_ac.dims.N = self.N

        ## cost
        self.ocp_ac.cost.W = np.diag(np.zeros(model.ny))
        self.ocp_ac.cost.W_e = np.diag(np.zeros(model.nyN))

        self.ocp_ac.cost.cost_type = 'NONLINEAR_LS'
        self.ocp_ac.cost.cost_type_e = 'NONLINEAR_LS'

        self.ocp_ac.cost.yref = np.zeros(model.ny)
        self.ocp_ac.cost.yref_e = np.zeros(model.nyN)

        ## constraints
        self.ocp_ac.constraints.x0 = np.zeros(model.nx)
        self.ocp_ac.constraints.idxbx = model.idxbx
        self.ocp_ac.constraints.lbx = model.lbx
        self.ocp_ac.constraints.ubx = model.ubx
        self.ocp_ac.constraints.idxbx_e = model.idxbxN
        self.ocp_ac.constraints.lbx_e = model.lbxN
        self.ocp_ac.constraints.ubx_e = model.ubxN
        self.ocp_ac.constraints.lbu = model.lbu
        self.ocp_ac.constraints.ubu = model.ubu
        self.ocp_ac.constraints.idxbu = np.arange(0, model.nu)
        self.ocp_ac.constraints.lh = model.lh
        self.ocp_ac.constraints.uh = model.uh
        self.ocp_ac.constraints.lh_e = model.lhN
        self.ocp_ac.constraints.uh_e = model.uhN

        ## soft constraints
        self.ocp_ac.constraints.idxsbx = model.idxsbx
        self.ocp_ac.constraints.idxsbx_e = model.idxsbxN
        self.ocp_ac.constraints.idxsh = model.idxsh
        self.ocp_ac.constraints.idxsh_e = model.idxshN

        self.ocp_ac.cost.zl = np.append(model.slack_x_w_L1_stage, model.slack_h_w_L1_stage)
        self.ocp_ac.cost.Zl = np.append(model.slack_x_w_L2_stage, model.slack_h_w_L2_stage)
        self.ocp_ac.cost.zu = self.ocp_ac.cost.zl
        self.ocp_ac.cost.Zu = self.ocp_ac.cost.Zl
        self.ocp_ac.cost.zl_e = np.append(model.slack_x_w_L1_term, model.slack_h_w_L1_term)
        self.ocp_ac.cost.Zl_e = np.append(model.slack_x_w_L2_term, model.slack_h_w_L2_term)
        self.ocp_ac.cost.zu_e = self.ocp_ac.cost.zl_e
        self.ocp_ac.cost.Zu_e = self.ocp_ac.cost.Zl_e

        ## solver parameters
        ## time horizon
        self.ocp_ac.solver_options.tf = self.T
        self.ocp_ac.solver_options.time_steps = self.dt

        ## l4casadi stuff
        if self.model.cfg.flags['enable_sdf']:
            self.ocp_ac.solver_options.model_external_shared_lib_dir = os.path.join(cache_dir(), 'codegen', self.model.cfg.name)
            self.ocp_ac.solver_options.model_external_shared_lib_name = 'sdf_l4c'

        ## see https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpOptions
        self.ocp_ac.solver_options.print_level = 0
        self.ocp_ac.solver_options.integrator_type = 'ERK'  # ERK, IRK, GNSF, DISCRETE, LIFTED_IRK

        ## NLP solver parameters
        self.ocp_ac.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp_ac.solver_options.rti_phase = 0  # 0: both; 1: preparation; 2: feedback

        ## QP solver parameters
        self.ocp_ac.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        self.ocp_ac.solver_options.hpipm_mode = 'ROBUST'
        self.ocp_ac.solver_options.qp_solver_iter_max = 100
        self.ocp_ac.solver_options.qp_solver_warm_start = 1  # 0: cold; 1: primal warm; 2: primal+dual warm

        ## Hessian approx
        self.ocp_ac.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp_ac.solver_options.levenberg_marquardt = float(self.model.cfg.mpc.lm_reg)

        ## init variables
        self.solver = None
        self.u = np.zeros(model.nu)
        self.t = 0

        self.solver = AcadosOcpSolver(self.ocp_ac, json_file=self.json, generate=build, build=build)

        ## if one want to use csython solver:  (doesn't work with relative paths?)
        # if build:
        #     AcadosOcpSolver.generate(self.ocp_ac, json_file=self.json)
        #     AcadosOcpSolver.build(self.ocp_ac.code_export_directory, with_cython=True)
        # self.solver = AcadosOcpSolver.create_cython_solver(self.json)


    def set_W(self, w):
        """Set weights."""
        model.W = w
        model.WN = w[:model.nyN]
        self.ocp_ac.cost.W = np.diag(model.W)
        self.ocp_ac.cost.W_e = np.diag(model.WN)


    def init(self, x0):
        self.solver.reset()
        for k in range(self.N):
            self.solver.set(k, 'x', x0)
            self.solver.set(k, 'u', self.model.u_hover)
        self.solver.set(self.N, 'x', x0)


    def shift(self, k=1):
        if k > 0:
            for i in range(k, self.N):
                self.solver.set(i-k, 'x', self.solver.get(i, 'x'))
                self.solver.set(i-k, 'u', self.solver.get(i, 'u'))


    def solve(self, x0, y, yN, W, WN, p):
        self.ocp_ac.constraints.x0 = x0
        self.solver.set(0, 'x', x0)
        for k in range(self.N):
            self.solver.cost_set(k, 'yref', y[k])
            self.solver.cost_set(k, 'W', np.diag(W[k]))
            self.solver.set(k, 'p', p[k])
        self.solver.cost_set(self.N, 'yref', yN)
        self.solver.cost_set(self.N, 'W', np.diag(WN))
        self.solver.set(self.N, 'p', p[-1])
        self.u = self.solver.solve_for_x0(x0)
        self.t = self.solver.get_stats('time_tot')


    def get_u(self):
        return np.array(self.u).flatten()


    def get_t(self):
        return float(self.t)
