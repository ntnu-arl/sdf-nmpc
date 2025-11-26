import copy
import numpy as np
from .utils.math import quat2rot, yaw2quat, quat2yaw
from .utils.reference import Ref


class RefGen:
    def __init__(self, cfg):
        self.cfg = cfg
        self.x0 = None
        self.ref = Ref(cfg)
        self.force_yaw_current = (self.cfg.ref.yaw_mode == 'curent')

    def _reset(self):
        self.ref = Ref(self.cfg)

    def from_x0(self):
        ref = copy.copy(self.ref)
        ref.p = self.x0[:3]
        ref.q = yaw2quat(quat2yaw(self.x0[3:7]))
        ref.v = [0.,0.,0.]
        ref.wz = 0.
        return [ref] * self.cfg.mpc.N

    def gen_ref_list_wps(self, wps):
        self._reset()
        trajectory = []

        path_p = np.vstack([self.x0[:3], [wp.p for wp in wps]])
        path_q = np.vstack([self.x0[3:7], [wp.q for wp in wps]])
        path_yaw = list(map(quat2yaw, path_q))

        ## handle stop and turn
        if self.cfg.ref.stop_and_turn.enable:
            yaw_curr = path_yaw[0]
            yaw_r = yaw_curr  # default case
            if self.cfg.ref.yaw_mode == 'topic':
                yaw_r = quat2yaw(path_q[1])
            elif self.cfg.ref.yaw_mode == 'align':
                dxy = path_p[1][:2] - self.x0[:2]
                if np.linalg.norm(dxy) > self.cfg.ref.yaw_align_dmin:
                    yaw_r = np.arctan2(dxy[1], dxy[0])
                yaw_r += self.cfg.ref.align_yaw_offset
            if abs(yaw_curr - yaw_r) > self.cfg.ref.stop_and_turn.dang_min:
                ref = copy.copy(self.ref)
                ref.p = self.x0[:3]
                ref.v = [0,0,0]
                ref.q = yaw2quat(yaw_r)
                return [ref] * self.cfg.mpc.N

        ## compute cumulative distances along the path
        distances = np.linalg.norm(np.diff(path_p, axis=0), axis=1)
        cumulative_distances = np.cumsum(distances)
        cumulative_distances = np.insert(cumulative_distances, 0, 0)

        total_distance = cumulative_distances[-1]
        if total_distance / 1e-3:
            vref = min(self.cfg.ref.vref, total_distance)  # heuristic saturation of vref to avoid overshoot

            ## compute evenly spaced points at desired velocity
            even_distances = np.arange(0, total_distance, self.cfg.mpc.T / self.cfg.mpc.N * vref)
            for i, d in enumerate(even_distances):
                segment_index = np.searchsorted(cumulative_distances, d) - 1
                segment_index = max(0, min(segment_index, len(distances) - 1))
                direction = (path_p[segment_index + 1] - path_p[segment_index]) / distances[segment_index]
                delta_dist = (d - cumulative_distances[segment_index])

                ref = copy.copy(self.ref)
                ref.p = path_p[segment_index] + direction * delta_dist
                ref.v = direction * vref

                ## align yaw
                if self.force_yaw_current:  # 'current'
                    ref.q = path_q[0]
                elif self.cfg.ref.yaw_mode == 'ref':
                    yaw_r = path_yaw[segment_index+1]
                    ref.q = yaw2quat(yaw_r)
                elif self.cfg.ref.yaw_mode == 'align':
                    dxy = path_p[1][:2] - self.x0[:2]
                    if np.linalg.norm(dxy) > self.cfg.ref.yaw_align_dmin:
                        yaw_r = np.arctan2(ref.v[1], ref.v[0])
                        yaw_r += self.cfg.ref.align_yaw_offset
                        ref.q = yaw2quat(yaw_r)
                    else:
                        ref.q = path_q[0]
                else:
                    ref.q = [1,0,0,0]

                trajectory.append(ref)
                if len(trajectory) > self.cfg.mpc.N:
                    break

        while len(trajectory) <= self.cfg.mpc.N:
            ref = copy.copy(self.ref)
            ref.p = trajectory[-1].p if trajectory else path_p[-1]
            ref.q = trajectory[-1].q if trajectory else path_q[-1]
            trajectory.append(ref)

        return trajectory

    def gen_ref_joystick(self, vwref):
        """Set (vx, vy, vz, wz) reference."""
        ref = copy.copy(self.ref)

        ref.v = np.array(vwref[:3]) * self.cfg.ref.vref
        ref.wz = np.array(vwref[3]) * self.cfg.ref.wzref

        ## yaw alignment, if enable
        if self.force_yaw_current:  # 'current'
            ref.q = yaw2quat(quat2yaw(self.x0[3:7]))
        elif self.cfg.ref.yaw_mode == 'align':
            vxy = ref.v[:2]
            if np.linalg.norm(vxy) > self.cfg.ref.yaw_align_dmin:
                yaw_r = np.arctan2(vxy[1], vxy[0])
                ref.q = yaw2quat(yaw_r)
            else:
                ref.q = yaw2quat(quat2yaw(self.x0[3:7]))
        else:
            ref.q = [1,0,0,0]

        ## set unused position reference for visualization
        ref.Wp = [0,0,0]
        trajectory = []
        for i in range(self.cfg.mpc.N+1):
            trajectory.append(copy.copy(ref))
            trajectory[-1].p = self.x0[:3] + (ref.v * i * self.cfg.mpc.T / self.cfg.mpc.N)

        return trajectory
