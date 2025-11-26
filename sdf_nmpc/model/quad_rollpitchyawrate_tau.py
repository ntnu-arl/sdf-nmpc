import numpy as np
import casadi as cs
from .base_model import BaseModel
from ..utils.math import *


class Quad(BaseModel):
    def __init__(self, cfg):
        super().__init__('quad_rollpitchyawrate', cfg)

        ## model constants
        self.nx = 10 # p, q, v  -- qy and qx are unused but this keeps a unified interface with the rest of the module
        self.nu = 4  # T/m, roll, pitch, wz
        self.ny = 11
        self.nyN = 4
        self.np = self.cfg.mpc.p_idx.latent + self.cfg.nn.size_latent
        self.gen_symbols()

        tau_roll = 0.12
        tau_pitch = 0.12

        ## symbols
        p = self.x[:3]
        q = self.x[3:7] ; q /= cs.norm_2(q)
        eta = quat2euler(q)
        v = self.x[7:]
        gamma = self.u[0] * self.cfg.robot.limits.gamma
        roll_des = self.u[1] * self.cfg.robot.limits.roll
        pitch_des = self.u[2] * self.cfg.robot.limits.pitch
        wz = self.u[3] * self.cfg.robot.limits.wz

        ## dynamics
        W_R_B = quat2rot(q)
        W_a = W_R_B @ cs.vertcat(0, 0, gamma) + cs.vertcat(0, 0, -self.g)
        dot_roll = (roll_des - eta[0]) / tau_roll
        dot_pitch = (pitch_des - eta[1]) / tau_pitch
        w = deuler_avel_map(eta) @ cs.vertcat(dot_roll, dot_pitch, 0)
        dq = hamilton_prod(q, cs.vertcat(0, w[0], w[1], wz)) / 2
        self.f_expl = cs.vertcat(
            v,
            dq,
            W_a,
        )
        self.f_impl = self.dx - self.f_expl

        self.u_hover = np.array([self.g / self.cfg.robot.limits.gamma, 0, 0, 0])
        self.u_to_acc = cs.Function('u_to_acc', [self.x, self.u, self.p], [cs.vertcat(W_R_B.T @ W_a, wz)])
        self.u_to_TRPYr = cs.Function('u_to_TRPYr', [self.x, self.u, self.p], [gamma * self.cfg.robot.mass, roll_des, pitch_des, wz])

        ## cost
        q_d = self.p[self.cfg.mpc.p_idx.q_d]
        q_e = hamilton_prod(q_d, invert(q))
        self.y = cs.vertcat(p, q_e[3], v, roll_des, pitch_des, wz, W_a[2])
        if not (cfg.flags.enable_sdf and cfg.flags.recursive_feasibility and cfg.flags.stability):
            self.yN = cs.vertcat(p, q_e[3])
        else:
            self.yN = cs.vertcat(p, q_e[3]) * self.p[0]

        ## input bounds
        self.lbu = np.array([0, -1, -1, -1])
        self.ubu = np.array([1,  1,  1,  1])


    def formate_ref(self, ref):
        yr = np.concatenate([ref.p, [0], ref.v, [0, 0], [ref.wz], [0], np.zeros_like(self.extra_W)])
        W = np.concatenate([ref.Wp, [ref.Wq[2]], ref.Wv, ref.Wq[:2], ref.Ww[2:], [ref.Wa], self.extra_W])
        return yr, W
