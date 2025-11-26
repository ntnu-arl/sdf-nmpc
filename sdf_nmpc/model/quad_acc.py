import numpy as np
import casadi as cs
from .base_model import BaseModel
from ..utils.math import *


class Quad(BaseModel):
    def __init__(self, cfg):
        super().__init__('quad_acc', cfg)

        ## model constants
        self.nx = 10 # p, q, v
        self.nu = 4  # a, wz
        self.ny = 11
        self.nyN = 7
        self.np = self.cfg.mpc.p_idx.latent + self.cfg.nn.size_latent
        self.gen_symbols()

        ## symbols
        p = self.x[:3]
        q = self.x[3:7] ; q /= cs.norm_2(q)
        v = self.x[7:]
        wz = self.u[3] * self.cfg.robot.limits.wz
        W_a = cs.vertcat(self.u[0] * self.cfg.robot.limits.ax, self.u[1] * self.cfg.robot.limits.ay, self.u[2] * self.cfg.robot.limits.az)

        ## dynamics
        W_R_B = quat2rot(q)
        dq = hamilton_prod(q, cs.vertcat(0, 0, 0, wz)) / 2
        self.f_expl = cs.vertcat(
            v,
            dq,
            W_a,
        )
        self.f_impl = self.dx - self.f_expl

        self.u_hover = np.array([0, 0, 0, 0])
        self.u_to_acc = cs.Function('u_to_acc', [self.x, self.u, self.p], [cs.vertcat(W_R_B.T @ W_a, wz)])

        ## cost
        q_d = self.p[self.cfg.mpc.p_idx.q_d]
        q_e = hamilton_prod(q_d, invert(q))
        self.y = cs.vertcat(p, q_e[3], v, W_a, wz)
        if not (cfg.flags.enable_sdf and cfg.flags.recursive_feasibility and cfg.flags.stability):
            self.yN = cs.vertcat(p, q_e[3], v)
        else:
            self.yN = cs.vertcat(p, q_e[3], v) * self.p[0]


        ## input bounds
        self.lbu = np.array([-1, -1, -1, -1])
        self.ubu = np.array([ 1,  1,  1,  1])


    def formate_ref(self, ref):
        yr = np.concatenate([ref.p, [0], ref.v, [0, 0, 0], [ref.wz], np.zeros_like(self.extra_W)])
        W = np.concatenate([ref.Wp, ref.Wq[2:], ref.Wv, [ref.Wa, ref.Wa, ref.Wa], [ref.Ww[2]], self.extra_W])
        return yr, W
