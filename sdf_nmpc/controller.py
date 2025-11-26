import numpy as np
from .ocp import Ocp
from .utils.math import euler2rot, euler2quat, quat2euler, quat2rot, yaw2quat, quat2yaw
from .gen_model import get_model_from_cfg


class Nmpc:
    """Wrapper around the NMPC controller with range image-based collision prediction."""
    def __init__(self, cfg, rebuild=False):
        self.cfg = cfg

        ## generate model and OCP
        self.model, self.sdf = get_model_from_cfg(self.cfg)
        self.T = self.cfg.mpc.T
        self.N = self.cfg.mpc.N
        self.ocp = Ocp(self.model, build=rebuild)

        self.cmd_acc_hover = np.array([0, 0, 0, 0])
        self.cmd_acc_min = [-self.cfg.robot.limits.ax,-self.cfg.robot.limits.ay,-self.cfg.robot.limits.az,-self.cfg.robot.limits.wz]
        self.cmd_acc_max = [ self.cfg.robot.limits.ax, self.cfg.robot.limits.ay, self.cfg.robot.limits.az, self.cfg.robot.limits.wz]
        self.cmd_TRPYr_hover = np.array([self.cfg.robot.mass * self.model.g, 0, 0, 0])
        self.cmd_TRPYr_min = [ 0, -self.cfg.robot.limits.roll, -self.cfg.robot.limits.pitch, -self.cfg.robot.limits.wz]
        self.cmd_TRPYr_max = [self.cfg.robot.limits.gamma, self.cfg.robot.limits.roll, self.cfg.robot.limits.pitch, self.cfg.robot.limits.wz]
        self.cmd_props_hover = np.array([self.cfg.robot.mass * self.model.g / 4] * 4)
        self.cmd_props_min = [0] * 4
        self.cmd_props_max = [self.cfg.robot.limits.wp] * 4

        ## reset internal values
        self.reset()


    def reset(self):
        """Reset internal matrices to default values."""
        self.x0 = None
        self.p = np.zeros((self.N+1, self.model.np), dtype='float')
        self.y = np.zeros((self.N, self.model.ny), dtype='float')
        self.yN = np.zeros(self.model.nyN, dtype='float')
        self.W = np.zeros((self.N, self.model.ny), dtype='float')
        self.WN = np.zeros(self.model.nyN, dtype='float')
        self.fail_count = 0
        self.set_sdf_flag(False)
        self.reset_latent()

    ## parameter setters
    def set_sdf_flag(self, flag):
        """Set flag to enable/disable sdf cost and constraints."""
        self.p[:,self.cfg.mpc.p_idx.flag] = flag


    def set_latent(self, latent, W_p_Bo, W_R_Bo):
        """Set latent and state parameters."""
        self.p[:,self.cfg.mpc.p_idx.W_p_Co] = np.array((W_R_Bo @ self.cfg.sensor.B_p_C + W_p_Bo)).flatten()
        self.p[:,self.cfg.mpc.p_idx.W_R_Co] = (W_R_Bo @ self.cfg.sensor.B_R_C).reshape(9)
        self.p[:,self.cfg.mpc.p_idx.latent:] = latent


    def reset_latent(self):
        """Reset latent and state parameters."""
        self.p[:,self.cfg.mpc.p_idx.W_p_Co] = 0
        self.p[:,self.cfg.mpc.p_idx.W_R_Co] = 0
        self.p[:,self.cfg.mpc.p_idx.latent:] = 0


    ## control iteration
    def set_x0(self, x0):
        """Current state feedback."""
        x0 = x0[:self.model.nx]
        if self.x0 is None: self.ocp.init(x0)  # init ocp x and u matrices at first control loop
        self.x0 = x0


    def solve(self):
        """Solve the NLP."""
        try:
            self.ocp.shift(self.cfg.mpc.shift)
            self.ocp.solve(self.x0, self.y, self.yN, self.W, self.WN, self.p)
            self.fail_count = 0
        except Exception as e:
            print('solver failed:', e)
            self.fail_count += 1
        return self.fail_count


    ## getters
    def get_matrices(self):
        """Returns x, u matrices, respectively for k = [0,N] and [0,N-1]."""
        x = np.zeros([self.N+1, self.model.nx])
        u = np.zeros([self.N, self.model.nu])
        for k in range(self.N):
            x[k,:] = self.ocp.solver.get(k, 'x')
            u[k,:] = self.ocp.solver.get(k, 'u')
        x[self.N,:] = self.ocp.solver.get(self.N, 'x')
        return x, u


    def get_u(self):
        """Returns last computed MPC inputs."""
        return self.ocp.get_u()


    def get_cmd_acc(self):
        """Returns clipped last computed system commands."""
        return np.clip(np.array(self.model.u_to_acc(self.x0, self.get_u(), self.p[0,:])).flatten(), self.cmd_acc_min, self.cmd_acc_max)


    def get_cmd_TRPYr(self):
        """Returns clipped last computed system commands."""
        return np.clip(np.array(self.model.u_to_TRPYr(self.x0, self.get_u(), self.p[0,:])).flatten(), self.cmd_TRPYr_min, self.cmd_TRPYr_max)


    def get_cmd_props(self):
        """Returns clipped last computed system commands."""
        return np.clip(np.array(self.model.u_to_props(self.x0, self.get_u(), self.p[0,:])).flatten(), self.cmd_props_min, self.cmd_props_max)


    def get_openloop_traj(self):
        """Returns the last computed predicted (x,y,z, qw,qx,qy,qz) trajectory."""
        path = [(self.x0[[0,1,2]], self.x0[[3,4,5,6]])]
        for k in range(1, self.N + 1):
            x = self.ocp.solver.get(k, 'x')
            path.append((x[[0,1,2]], x[[3,4,5,6]]))
        return path


    def eval(self, k):
        """Return value of model evaluation vector for shooting node k."""
        if self.model.eval_vec.shape[0] == 0:
            return [0]
        else:
            return np.array(self.model.eval(self.ocp.solver.get(k, 'x'), self.ocp.solver.get(k, 'u'), self.p[k])).flatten()


    def set_ref(self, ref, k):
        """Set self.y, self.W and self.p from a ref object for shooting node k."""
        self.p[k, self.cfg.mpc.p_idx.q_d] = ref.q
        y, W = self.model.formate_ref(ref)
        if k < self.N:
            self.y[k,:] = y
            self.W[k,:] = W
        else:
            self.WN[:] = W[:self.model.nyN]
            self.yN[:] = y[:self.model.nyN]
