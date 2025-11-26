import numpy as np
import torch


def _loss_with_invalid_pixels(loss, target):
    """Compute loss masked for invalid pixels with ratio compensation."""
    mask_valid_pixels = target > 0
    # invalid_ratio_compensation = torch.sum(mask_valid_pixels, dim=[1, 2, 3]) / (target.shape[-1] * target.shape[-2])
    masked_loss = torch.where(mask_valid_pixels, loss, 0)
    # return torch.mean(torch.sum(masked_loss, dim=[1, 2, 3]) * invalid_ratio_compensation)  # sum over pixel errors, mean over batch
    return torch.mean(torch.sum(masked_loss, dim=[1, 2, 3]))  # sum over pixel errors, mean over batch


def loss_MSE_valid_pixels(target, reconst):
    """MSE loss for VAE ignoring invalid pixels."""
    mse = torch.nn.functional.mse_loss(reconst, target, reduction="none")
    return _loss_with_invalid_pixels(mse, target)


def loss_MSE_valid_pixels_bias_distance(target, reconst, weight_ratio=0.1, degree=2):
    """MSE loss for VAE ignoring invalid pixels with a biased weighting toward predicting pixels closer to the sensor.
    weight_ratio    -- ratio of weights for min range against max range
    degree          -- degree of the distance bias interpolation function
    """
    mse = torch.nn.functional.mse_loss(reconst, target, reduction="none")
    biased_mse = mse * (target**degree * (weight_ratio - 1) + 1)  # quadratic interp with flat tangent in 0
    return _loss_with_invalid_pixels(biased_mse, target)


def loss_MSE_valid_pixels_bias_positive(target, reconst, weight_ratio=0.1):
    """MSE loss for VAE ignoring invalid pixels with a biased weighting toward predicting pixels closer or further than their target value.
    weight_ratio    -- ratio of weights for negative errors (closer) against positive errors (further)
    """
    mse = torch.nn.functional.mse_loss(reconst, target, reduction="none")
    biased_mse = torch.where(target > reconst, mse * weight_ratio, mse)
    return _loss_with_invalid_pixels(biased_mse, target)


def loss_MSE_valid_pixels_bias_pos_dist(target, reconst, pos_ratio=1, dist_ratio=1, degree=2):
    """MSE loss for VAE ignoring invalid pixels with a biased weighting toward predicting pixels closer to the sensor and closer or further than their target value.
    pos_ratio       -- ratio of weights for negative errors (closer) against positive errors (further)
    dist_ratio      -- ratio of weights for min range against max range
    degree          -- degree of the distance bias interpolation function
    """
    mse = torch.nn.functional.mse_loss(reconst, target, reduction="none")
    biased_mse = torch.where(target > reconst, mse * pos_ratio, mse)
    biased_mse = biased_mse * (target**degree * (dist_ratio - 1) + 1)  # quadratic interp with flat tangent in 0
    return _loss_with_invalid_pixels(biased_mse, target)

def loss_KLD(mean, logvar, beta, size_latent, size_img):
    """KLD loss for VAE with normalized beta parameter, see https://openreview.net/pdf?id=Sy2fzU9gl."""
    beta_norm = (beta * size_latent) / (size_img[0] * size_img[1])
    kld_term = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
    kld = kld_term * beta_norm
    return kld


def loss_weighted_BCE(predictions, labels, weights=(1,1)):
    """BCE classification loss with different weighting on both classes.
    weights     -- 2-sized tuple, weights[0] and weights[1] are respectively the weights on 0 and 1 classes.
    """
    clamped = torch.clamp(predictions, min=1e-7, max=1-1e-7)  # avoiding numerical instabilities
    bce = - weights[1] * labels * torch.log(clamped) - weights[0] * (1-labels) * torch.log(1 - clamped)
    return torch.mean(bce)


## SDF losses
def loss_sdf(nn_outputs, nn_inputs, target_grad, target_ouputs):
    """Multi component loss for training SDFs network. Largely inspired from https://joeaortiz.github.io/iSDF.
    1. regression loss
    2. gradient direction loss against expected gradient direction
    3. eikonal loss: gradient norm equal to 1 in non-truncated regions, and 0 otherwise
    The gradients are computed through the network via autograd.
    """
    ## regression loss against SDF target
    # loss_regression = torch.nn.functional.mse_loss(nn_outputs.flatten(), target_ouputs.flatten(), reduction="mean")
    mse = torch.nn.functional.mse_loss(nn_outputs.flatten(), target_ouputs.flatten(), reduction="none")
    different_sign = (torch.sign(target_ouputs).flatten() - torch.sign(nn_outputs).flatten()).to(torch.bool)
    loss_regression = torch.where(different_sign, mse * 10, mse).mean()

    ## compute gradients and norms
    nn_grad = torch.autograd.grad(nn_outputs, nn_inputs, grad_outputs=torch.ones_like(nn_outputs), retain_graph=True, create_graph=True)[0]
    norm_nn_grad = torch.linalg.vector_norm(nn_grad, dim=-1)

    ## gradient loss
    loss_gradient_mse = torch.nn.functional.mse_loss(nn_grad, target_grad, reduction="mean")

    ## gradient direction loss
    norm_target_grad = torch.linalg.vector_norm(target_grad, dim=-1)
    mask_unsat = (norm_target_grad > 0)  # True for points in the unsat region of the TSDFs
    loss_gradient_dir = torch.acos((nn_grad[mask_unsat] * target_grad[mask_unsat]).sum(dim=-1) / (norm_nn_grad[mask_unsat] + torch.ones_like(norm_nn_grad[mask_unsat])*1e-6)).mean()

    ## eikonal loss
    loss_eikonal = torch.nn.functional.mse_loss(norm_nn_grad, norm_target_grad, reduction="mean")

    return loss_regression, loss_gradient_mse, torch.rad2deg(loss_gradient_dir), loss_eikonal
