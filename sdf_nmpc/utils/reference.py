import numpy as np
from .config import Config, AttrDict
from .math import euler2rot, euler2quat, quat2euler, quat2rot, yaw2quat, quat2yaw


class Ref:
    def __init__(self, cfg):
        self.cfg = cfg

        self.p = [0.,0.,0.]
        self.q = [1.,0.,0.,0.]
        self.v = [0.,0.,0.]
        self.wz = 0.

        self.W_on = AttrDict({
            'Wp': self.cfg.mpc.weights.set_const_off.pos,
            'Wq': self.cfg.mpc.weights.set_const_off.att,
            'Wv': self.cfg.mpc.weights.set_const_off.vel,
            'Ww': self.cfg.mpc.weights.set_const_off.rates,
            'Wa': self.cfg.mpc.weights.set_const_off.acc,
        })
        self.W_off = AttrDict({
            'Wp': self.cfg.mpc.weights.set_const_on.pos,
            'Wq': self.cfg.mpc.weights.set_const_on.att,
            'Wv': self.cfg.mpc.weights.set_const_on.vel,
            'Ww': self.cfg.mpc.weights.set_const_on.rates,
            'Wa': self.cfg.mpc.weights.set_const_on.acc,
        })

    def from_state(self, x):
        self = Ref()
        self.p = x[:3]
        self.q = x[3:7]
        self.v = x[7:10]
        try:
            self.wz = x[12]
        except IndexError:
            self.wz = 0.

    def hover_at_state(self, x):
        self.p = x[:3]
        self.q = yaw2quat(quat2yaw(x[3:7]))
        self.v = [0., 0., 0.]
        self.wz = 0.


class Waypoint:
    def __init__(self, p, q=[1,0,0,0]):
        self.p = np.array(p, dtype=float)
        self.q = np.array(q, dtype=float)

    def __str__(self):
        return f'{self.p}, {quat2euler(self.q)}'
