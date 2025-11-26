import numpy as np
import casadi as cs
from .base_model import BaseModel
from ..utils.math import *


class Quad(BaseModel):
    def __init__(self, cfg):
        super().__init__('quad_props', cfg)

        ## model constants
        self.nx = 13  # p, q, v, w
        self.nu = 4  # motor speeds
        self.ny = 16
        self.nyN = 12
        self.np = self.cfg.mpc.p_idx.latent + self.cfg.nn.size_latent
        self.gen_symbols()

        ## allocation
        px, py, pz, alpha, beta, sign = zip(*self.cfg.robot.alloc.motors)
        cf = [self.cfg.robot.alloc.cf] * len(sign)
        ct = [self.cfg.robot.alloc.ct] * len(sign)
        R = [axis_rot('z', i * (pi / (len(sign) / 2))) @ axis_rot('y', beta[i]) @ axis_rot('x', (-1) ** i * alpha[i]) for i in range(len(sign))]
        p = np.array([px, py, pz]).T
        Gf, Gt = GTMRP_matrix(R, p, sign, cf, ct)
        Gf = cf * Gf
        Gt = cf * Gt

        ## symbols
        p = self.x[0:3]
        q = self.x[3:7] ; q /= cs.norm_2(q)
        eta = quat2euler(q)
        v = self.x[7:10]
        w = self.x[10:]
        wp = self.u * self.cfg.robot.limits.wp
        m = self.cfg.robot.mass
        J = np.diag(self.cfg.robot.inertia)
        Jinv = np.linalg.inv(J)

        ## dynamics
        W_R_B = quat2rot(q)
        W_a = W_R_B @ Gf @ wp**2 / m + cs.vertcat(0, 0, -self.g)
        self.f_expl = cs.vertcat(
            v,
            hamilton_prod(q, cs.vertcat(0, w)) / 2,
            W_a,
            Jinv @ (Gt @ wp**2 - cs.cross(w, J @ w)),
        )
        self.f_impl = self.dx - self.f_expl

        self.wh = wh = np.sqrt(m * self.g / 4 / self.cfg.robot.alloc.cf)
        self.u_hover = np.array([wh, wh, wh, wh])
        self.u_to_props = cs.Function('u_to_props', [self.x, self.u, self.p], [wp])
        self.u_to_acc = cs.Function('u_to_acc', [self.x, self.u, self.p], [cs.vertcat(W_R_B.T @ W_a, w[2])])

        ## cost
        q_d = self.p[self.cfg.mpc.p_idx.q_d]
        q_e = hamilton_prod(q_d, invert(q))
        self.y = cs.vertcat(p, eta[:2], q_e[3], v, w, wp)
        self.yN = cs.vertcat(p, eta[:2], q_e[3], v, w)

        ## input bounds
        self.lbu = np.array([0, 0, 0, 0])
        self.ubu = np.array([1, 1, 1, 1])


    def formate_ref(self, ref):
        yr = np.concatenate([ref.p, [0, 0, 0], ref.v, [0, 0, ref.wz], [self.wh] * 4, np.zeros_like(self.extra_W)])
        W = np.concatenate([ref.Wp, ref.Wq, ref.Wv, ref.Ww, [ref.Wa] * 4, self.extra_W])
        return yr, W
