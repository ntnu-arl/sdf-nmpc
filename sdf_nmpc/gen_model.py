import os
import numpy as np
import torch
import l4casadi
import casadi as cs
from . import default_data_dir, cache_dir
from .model import cost_const_helpers, quad_rollpitchyawrate, quad_rollpitchyawrate_tau, quad_acc, quad_props
from .utils.math import polynomial_3variate
from .utils.stability import get_r_tilde_max


def get_model_from_cfg(cfg):
    """Generate nmpc model and its neural nets from the config."""
    model = None
    if cfg.mpc.model == 'acc': model = quad_acc.Quad(cfg)
    if cfg.mpc.model == 'att': model = quad_rollpitchyawrate.Quad(cfg)
    if cfg.mpc.model == 'att_tau': model = quad_rollpitchyawrate_tau.Quad(cfg)
    if cfg.mpc.model == 'rates': pass
    if cfg.mpc.model == 'wrench': pass
    if cfg.mpc.model == 'props': model = quad_props.Quad(cfg)
    if model is None:
        raise AssertionError('control mpc model')

    ## sdf
    sdf = None
    if cfg.flags['enable_sdf']:
        torch.set_num_threads(1)

        model.name += '_sdf'

        ## sdf
        sdf = torch.jit.load(os.path.join(default_data_dir(), cfg.nn.sdf_weights))
        sdf.to(cfg.nn.sdf_device)
        sdf.eval()
        sdf_bounds = [cfg.robot.size.xy + cfg.mpc.bound_margin, sdf.max_df + 0.2]

        ## l4casadi
        build_dir = os.path.join(cache_dir(), 'codegen', cfg.name)
        sdf_l4c = l4casadi.L4CasADi(sdf, model_expects_batch_dim=True, build_dir=build_dir, name='sdf_l4c', device=cfg.nn.sdf_device, with_jacobian=True, with_hessian=False)

        ## add fov constraint
        cost_const_helpers.add_fov_const_trigo(model, h_const=cfg.sensor.hfov < 3.14, v_const=cfg.flags['vfov_constraint'], slack=cfg.mpc.weights.slack_fov)
        # cost_const_helpers.add_fov_const_normals(model, h_const=cfg.sensor.hfov < 3.14, v_const=cfg.flags['vfov_constraint'], slack=cfg.mpc.weights.slack_fov)

        ## argument functions for sdf cost and constraints
        def sdf_args(x, u, p):
            W_R_Co = p[cfg.mpc.p_idx.W_R_Co].reshape((3, 3)).T  # transpose because casadi's reshape is column-major
            W_p_Co = p[cfg.mpc.p_idx.W_p_Co]
            W_p_B = x[:3]
            Co_p_B = W_R_Co.T @ (W_p_B - W_p_Co)
            return p[cfg.mpc.p_idx.flag], Co_p_B, p[cfg.mpc.p_idx.latent:]

        ## convenience arg function that ignores flag
        def sdf_args_noflag(x, u, p):
            return 1, *sdf_args(x, u, p)[1:]

        ## inference wrapper that accounts for flag value
        def sdf_flag(args):
            flag, Co_p_C, latent = args
            df_approx = sdf_l4c(cs.vertcat(Co_p_C, latent))
            return (flag * df_approx) + (1 - flag) * sdf.max_df

        ## add cost and constraints to model
        model.add_eval(sdf_flag, sdf_args_noflag)
        if cfg.flags['sdf_cost']:
            model.add_cost_stage(lambda x: (1-0.5*(sdf_flag(x)))**4, sdf_args, 20)
        if cfg.flags['sdf_constraint']:
            model.add_const_stage(sdf_flag, sdf_args, sdf_bounds, cfg.mpc.weights.slack_df)
            if not cfg.flags['recursive_feasibility']:  # terminal sdf constraint is redundant with rec feas constraint
                model.add_const_term(sdf_flag, sdf_args, sdf_bounds, cfg.mpc.weights.slack_df)

        ## terminal constraint for recursive feasibility
        if cfg.flags['recursive_feasibility']:
            assert cfg.control_mode == 'att', 'recursive feasibility and stability implemented only for \'att\' control mode'

            coeff_file = os.path.join(cache_dir(), cfg.mpc.braking_dist.coeff_file)
            coeffs = np.load(coeff_file)
            braking_dist_poly, _ = polynomial_3variate(cfg.mpc.braking_dist.degree, coeffs)

            ## argument functions for braking const
            def braking_args(x, u, p):
                v = x[7:10]
                return *sdf_args(x, u, p), v

            ## convenience arg function that ignores flag
            def braking_args_noflag(x, u, p):
                return 1, *braking_args(x, u, p)[1:]

            ## inference wrapper that accounts for flag value
            def braking_dist_flag(args):
                flag, _, _, v = args
                return flag * braking_dist_poly(v)

            def rec_feas(args):
                return sdf_flag(args[:-1]) - braking_dist_flag(args)

            # terminal fov constraints
            hfov_lim = model.cfg.sensor.hfov * model.cfg.mpc.fov_ratio
            vfov_lim = model.cfg.sensor.vfov * model.cfg.mpc.fov_ratio
            def hfov_const(arg):
                return arg[0] * cs.atan2(arg[1][1], arg[1][0])

            def vfov_const(arg):
                return arg[0] * cs.atan2(arg[1][2], cs.norm_2(arg[1][:2]))

            def args_Co_p_E(x,u,p):
                W_R_Co = p[model.cfg.mpc.p_idx.W_R_Co].reshape((3, 3)).T  # transpose because casadi is column-major
                W_p_Co = p[model.cfg.mpc.p_idx.W_p_Co]
                smooth_norm = cs.sqrt(cs.dot(x[7:], x[7:]) + 1e-4)
                W_p_E = x[:3] + braking_dist_flag(braking_args_noflag(x,u,p)) * x[7:] / smooth_norm
                Co_p_E = W_R_Co.T @ (W_p_E - W_p_Co) + model.cfg.sensor.B_R_C.T @ model.cfg.sensor.B_p_C
                return p[model.cfg.mpc.p_idx.flag], Co_p_E + cs.vertcat(cfg.mpc.fov_const_offset, 0, 0)


            ## add constraints to model
            model.add_eval(braking_dist_flag, braking_args_noflag)
            model.add_eval(rec_feas, braking_args_noflag)
            model.add_const_term(rec_feas, braking_args, [cfg.robot.size.xy, sdf.max_df], cfg.mpc.weights.slack_brake)
            model.add_const_term(hfov_const, args_Co_p_E, np.array([-hfov_lim, hfov_lim]))
            if cfg.flags['vfov_constraint']:
                model.add_const_term(vfov_const, args_Co_p_E, np.array([-vfov_lim, vfov_lim]))

            ## terminal cont for stability
            if cfg.flags['stability']:
                ## impose bound on velocity, such that stage cost is upper bounded
                cost_const_helpers.add_vel_const(model, stage=False, term=True)

                ## compute upper bound on stage cost
                max_vel_error = (2 * model.cfg.ref.vref)**2 * max(cfg.mpc.weights.vel)
                ## case 1
                max_att = np.array([cfg.robot.limits.roll, cfg.robot.limits.pitch, cfg.robot.limits.wz]).T
                max_att_error = max_att.T @ np.diag(cfg.mpc.weights.att[:2] + cfg.mpc.weights.rates[2:]) @ max_att
                max_thrust_error = max(cfg.mpc.weights.acc * (cfg.robot.limits.gamma - model.g)**2, cfg.mpc.weights.acc * model.g**2)
                sc_max = max_vel_error + max_att_error + max_thrust_error
                ab_min = cfg.mpc.stability.a_b_min
                dt = cfg.mpc.T / cfg.mpc.N

                ## case 2
                r_tilde = get_r_tilde_max(cfg)

                ## terminal cost
                def stab_cost_args(x, u, p):
                    return p[cfg.mpc.p_idx.flag], x[7:]

                def stab_cost(args):
                    return args[0] * args[1].T @ args[1]

                p_term = max(r_tilde + max_vel_error, sc_max / ab_min**2 / dt**2)
                model.add_cost_term(stab_cost, stab_cost_args, p_term)


    model.gen_acados_model()

    return model, sdf
