import numpy as np
import casadi as cs
from .base_model import BaseModel
from ..utils.math import *


class Quad(BaseModel):
    def __init__(self, cfg):
        super().__init__('quad_wrench', cfg)

        ## model constants
        self.nx = 13  # p, q, v, w
        self.nu = 4  # T/m, J^-1 @ τ (mass-normalized z thrust, ineria-normalized commanded body torques)
        self.ny = 12
        self.nyN = 12
        self.np = self.cfg.mpc.p_idx.latent + self.cfg.nn.size_latent
        self.gen_symbols()

        ## symbols
        p = self.x[0:3]
        q = self.x[3:7] ; q /= cs.norm_2(q)
        eta = quat2euler(q)
        v = self.x[7:10]
        w = self.x[10:]
        gamma = self.u[0] * self.cfg.robot.limits.gamma
        torques = self.u[1:] * self.cfg.robot.limits.torques

        self.u_hover = np.array([self.g, 0, 0, 0])
        self.u_to_cmd = cs.Function('utocmd', [self.u], [cs.vertcat(self.cfg.robot.mass * gamma, cs.diag(self.cfg.robot.inertia) @ torques)])

        ## dynamics
        self.f_expl = cs.vertcat(
            quat2rot(q) @ v,
            hamilton_prod(q, cs.vertcat(0, w)) / 2,
            quat2rot(q).T @ cs.vertcat(0, 0, -self.g) + cs.vertcat(0, 0, gamma),
            torques - cs.cross(w, w),
        )
        self.f_impl = self.dx - self.f_expl

        ## cost
        q_d = self.p[self.cfg.mpc.p_idx.q_d]
        q_e = hamilton_prod(q_d, invert(q))
        self.y = cs.vertcat(p, eta[:2], q_e[3], quat2rot(q) @ v, w)
        self.yN = cs.vertcat(p, eta[:2], q_e[3], quat2rot(q) @ v, w)

        ## input bounds
        self.lbu = np.array([0, -1, -1, -1])
        self.ubu = np.array([1,  1,  1,  1])


    def formate_ref(self, ref):
        yr = np.concatenate([ref.p, [0, 0, 0], ref.v, [0, 0, wz_ref], np.zeros_like(self.extra_W)])
        W = np.concatenate([ref.Wp, ref.Wq, ref.Wv, ref.Ww, self.extra_W])
        return yr, W
