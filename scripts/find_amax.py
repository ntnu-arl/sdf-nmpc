import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import casadi as cs
from sdf_nmpc import default_config_dir, default_data_dir
from sdf_nmpc.utils.config import Config
from sdf_nmpc.utils.math import euler2rot, polynomial_3variate
from sdf_nmpc.network.mlp import Mlp
import argparse
import io
import sys
import os


def silent_solve(solver, **kwargs):
    sys.stdout = io.StringIO()
    sol = solver(**kwargs)
    sys.stdout = sys.__stdout__
    return sol


class MinBrakingAccNlp:
    """NLP for computing the minimum maximum braking accel for all given velocities, given an approximate of this min braking distance.
    """
    def __init__(self, db_approx, v_max):
        self.x = cs.SX.sym('v', 3, 1)

        self.lbx = cs.vertcat([-v_max] * 3)
        self.ubx = cs.vertcat([v_max] * 3)

        db = db_approx(self.x)
        a = self.x.T @ self.x / 2 / db
        self.obj = a.T @ a

        self.g = self.x.T @ self.x - a**2 * 0.1**2
        self.lbg = 0
        self.ubg = v_max**2

        self.f = cs.Function('f', [self.x], [db])

    def solve(self):
        nlp = {'x': self.x, 'f': self.obj, 'g': self.g}
        solver = cs.nlpsol('min_acc_solver', 'ipopt', nlp, {'max_iter': 100})
        sol = silent_solve(solver, x0=[-3,-3,-3], lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
        return float(cs.sqrt(sol['f'])), np.array(sol['x']).flatten()


class BrakingAccNlp:
    """NLP for computing the maximum braking accel for a given velocity, under thrust and attitude constraints.
    The specific bounds for a given system are retrieved in the corresponding config file.
    """
    def __init__(self, cfg):
        m = cfg.robot.mass
        Tmax = cfg.robot.limits.gamma * m
        roll_max = cfg.robot.limits.roll
        pitch_max = cfg.robot.limits.pitch

        l = cs.SX.sym('lambda', 1, 1)
        T = cs.SX.sym('T', 1, 1)
        r = cs.SX.sym('r', 1, 1)
        p = cs.SX.sym('p', 1, 1)
        V_R_B = euler2rot(cs.vertcat(r, p, 0))

        self.lbx = cs.vertcat(0, 0, -roll_max, -pitch_max)
        self.ubx = cs.vertcat(cs.inf, Tmax, roll_max, pitch_max)

        self.x = cs.vertcat(l, T, r, p)
        self.a = cs.vertcat(0, 0, -9.81) + V_R_B @ cs.vertcat(0, 0, T / m)
        self.obj = - self.a.T @ self.a

    def solve(self, v):
        nlp = {'x': self.x, 'f': self.obj, 'g': self.a + self.x[0] * v}
        solver = cs.nlpsol('acc_solver', 'ipopt', nlp)
        sol = silent_solve(solver, x0=[0, 9.81, 0, 0], lbx=self.lbx, ubx=self.ubx, lbg=0., ubg=0.)
        return float(cs.sqrt(-  sol['f']))



class BrakeData(Dataset):
    """Data generator for fitting a function to approximate the braking distance given a 3D velocity, under thrust and attitude constraints.
    The data generation uses a BrakingAccNlp object to get the exact max allowed deceleration value for a given velocity direction.
    The corresponding braking distance is computed as braking_dist(v) = 1/2 * |v|^2 / braking_acc(v).
    Speeding up the data generation is achieved with memoization: normalized velocities and the corresponding braking acc are stored,
    and new velocities are normalized and compared to the stored ones. If the norm 2 error is less than the allowed tolerance, the
    stored braking acc value is used.
    Memoization arrays can be saved and loaded to disk as numpy arrays using the save and load methods.
    """
    def __init__(self, cpt, vmax, nb_samples, tol=2e-2, device='cpu'):
        self.cpt = cpt
        self.vmax = vmax
        self.nb_samples = nb_samples
        self.tol = tol
        self.device = device
        self.mem_x = None
        self.mem_y = None


    def save_mem(self, folder):
        np.save(f'{folder}/mem_x.npy', self.mem_x.cpu().numpy())
        np.save(f'{folder}/mem_y.npy', self.mem_y.cpu().numpy())


    def load_mem(self, folder):
        try:
            self.mem_x = torch.from_numpy(np.load(f'{folder}/mem_x.npy')).to(self.device)
            self.mem_y = torch.from_numpy(np.load(f'{folder}/mem_y.npy')).to(self.device)
        except FileNotFoundError:
            print('memoization matrices not found')


    def __len__(self):
        return self.nb_samples


    def __getitem__(self, idx):
        vel = (torch.rand(3, device=self.device) * 2 - 1) * self.vmax
        vel_norm = torch.nn.functional.normalize(vel, dim=0)
        if self.mem_x is None:
            acc = torch.tensor([self.cpt.solve(vel_norm.cpu().numpy())], device=self.device)
            self.mem_x = vel_norm
            self.mem_y = acc

        else:
            mem_idx = torch.nonzero(torch.linalg.vector_norm(self.mem_x - vel_norm, dim=-1) < self.tol)
            if len(mem_idx):
                acc = self.mem_y[mem_idx[0],0]
            else:
                acc = torch.tensor([self.cpt.solve(vel_norm.cpu().numpy())], device=self.device)

                self.mem_x = torch.vstack([self.mem_x, vel_norm])
                self.mem_y = torch.vstack([self.mem_y, acc])

        bdist = 0.5 * torch.linalg.vector_norm(vel)**2 / acc

        return vel, bdist



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='params', help='Compute for given param file.')
    parser.add_argument(dest='mode', choices=['grid', 'min_acc', 'poly_fit', 'poly_eval', 'mlp_fit', 'mlp_eval'], help='Mode.')
    args = parser.parse_args()

    cfg_file = f'params_{args.params}.yaml'

    ## param
    vmax = 3
    step = 0.05
    poly_deg = 4
    folder = 'braking_dist/' + args.params
    path = os.path.join(COLPREDMPC_TMP_DIR, folder)

    ## read cfg parameters
    mission_config = os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file)
    acc_cpt = BrakingAccNlp(Config(mission_config))

    if args.mode == 'grid':
        v = np.arange(-vmax, vmax + 0.01, step)
        vel = np.zeros((v.shape[0]**3, 3))
        vel[:,0] = np.repeat(v, len(v) * len(v))
        vel[:,1] = np.tile(np.repeat(v, len(v)), len(v))
        vel[:,2] = np.tile(v, len(v) * len(v))
        vel = vel[np.linalg.norm(vel, axis=1) < vmax]
        bdist = np.zeros(vel.shape[0])
        min_acc = float('inf')
        min_v = np.zeros(3)

        ## loop for each v
        for i, v in enumerate(vel):
            a = acc_cpt.solve(v)
            if a < 1e-8:
                bdist[i] = 0
            else:
                bdist[i] = 0.5 * np.linalg.norm(v)**2 / a
            if a > 1e-8 and a < min_acc:
                min_acc = a
                min_v = v
            print(f'solving for vel {i} out of {len(vel)}... (min acc is {min_acc:.3f} for [{min_v[0]:.2f},{min_v[1]:.2f},{min_v[2]:.2f}])', end='\r')
        print('\ndone!')

        ## save output to file
        if not os.path.isdir(path): os.makedirs(path)
        np.save(path + f'/{vmax}_{step}_bdist.npy', bdist)
        np.save(path + f'/{vmax}_{step}_vel.npy', vel)


    if args.mode == 'min_acc':
        vel = np.load(path + f'/{vmax}_{step}_vel.npy')
        bdist = np.load(path + f'/{vmax}_{step}_bdist.npy')

        idx = bdist > 1e-8
        print(f'amin: {np.min(np.linalg.norm(vel[idx], axis=1)**2 / 2 / bdist[idx])}')
        # coeffs = np.load(path + f'/bdist_poly_deg{poly_deg}_{vmax}.npy')
        # poly, _ = polynomial_3variate(poly_deg, coeffs)
        #
        # solver = MinBrakingAccNlp(poly, vmax)
        # a, v = solver.solve()
        # # a = 1
        # # v = np.random.uniform(-3,3,3)
        # # v = [0,0,-2]
        # # print(v)
        # print(f'a = {a} ; v = [{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}] ; |v| = {np.linalg.norm(v)} ; db = {np.linalg.norm(v)**2/2/a}')
        # print(f'db (poly) = {poly(v)} ; a (nlp) = {acc_cpt.solve(v)}, db (nlp) = {0.5 * np.linalg.norm(v)**2 / acc_cpt.solve(v)}')


    if args.mode == 'poly_fit':
        bdist = np.load(path + f'/{vmax}_{step}_bdist.npy')[::50]
        vel = np.load(path + f'/{vmax}_{step}_vel.npy')[::50]

        ## polynomial fitting
        poly, coeffs = polynomial_3variate(poly_deg)

        print('computing cost function...')
        ## least square fitting
        obj = 0
        for v, d in zip(vel, bdist):
            p_v = poly(coeffs, v)
            obj += (p_v - d)**2

        print('solving least square...')

        nlp = {'x': coeffs, 'f': obj}
        solver = cs.nlpsol('ls', 'ipopt', nlp)
        sol = silent_solve(solver, x0=0)

        print('done!')

        np.save(path + f'/bdist_poly_deg{poly_deg}_{vmax}.npy', sol['x'])


    if args.mode == 'mlp_fit':
        nb_epoch = 500
        nb_samples = 100000
        batch_size = 25
        learning_rate = 2e-5
        inner_size = [20,20,20]

        poly_nn = MLP(3, 1, inner_size, torch.nn.Tanh(), filename=f'{folder}/bdist_mlp_{str(inner_size)[1:-1].replace(", ","_")}_{vmax}', device='cuda')
        summary(poly_nn, input_size=[3], depth=4, device=poly_nn.device)
        # poly_nn.load_weights()
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(poly_nn.parameters(), lr=learning_rate)

        data = BrakeData(acc_cpt, vmax, nb_samples, 5e-3, device=poly_nn.device)
        data.load_mem(path)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

        best_max_loss = np.inf
        for idx_epoch in range(nb_epoch):
            print(f'-------------------------------\nepoch {idx_epoch}')
            poly_nn.train()
            optimizer.zero_grad()

            aggreg_losses = []
            for idx_batch, (x, y) in enumerate(dataloader):
                l = loss(poly_nn(x), y)
                l.backward()
                optimizer.step()
                aggreg_losses.append(l.item())
                print(f'loss (batch {idx_batch}): {l.item()}', end='\r')
            print(f'average loss: {np.mean(aggreg_losses)}, {np.max(aggreg_losses)}')

            poly_nn.save_weights()
            data.save_mem(path)


    if args.mode in ['poly_eval', 'mlp_eval']:
        bdist = np.load(path + f'/{vmax}_{step}_bdist.npy')
        vel = np.load(path + f'/{vmax}_{step}_vel.npy')
        idx = np.linalg.norm(vel, axis=-1) < 3
        vel = vel[idx]
        bdist = bdist[idx]
        print(f'eval on {bdist.shape[0]} samples')

        if args.mode == 'poly_eval':
            coeffs = np.load(path + f'/bdist_poly_deg{poly_deg}_{vmax}.npy')
            poly, _ = polynomial_3variate(poly_deg, coeffs)
            print(f'3-variate polynomial of degree {poly_deg} has {len(coeffs)} coefficients')
        else:
            inner_size = [20,20,20]
            poly_nn = MLP(3, 1, inner_size, torch.nn.Tanh(), filename=f'{folder}/bdist_mlp_{str(inner_size)[1:-1].replace(", ","_")}_{vmax}', device='cpu')
            poly_nn.load_weights()
            poly = lambda x: poly_nn(torch.from_numpy(x).to(torch.float32).to(poly_nn.device))

        error = np.array([float(poly(v)) for v in vel]) - bdist

        print(f'rmse: {np.sqrt(np.mean(np.square(error)))}')
        print(f'max error: {np.max(np.abs(error))}')
