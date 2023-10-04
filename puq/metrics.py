import math
import torch


def compute_pcs_masks(svs, lambda1s):
    K = svs.shape[1]
    all_energies = []
    for i in range(K):
        selected_sigmas = svs[:, :i]
        weights = (selected_sigmas ** 2) / \
            (svs ** 2).sum(dim=1, keepdim=True)
        energy = weights.sum(dim=1)
        all_energies.append(energy)
    all_energies = torch.stack(all_energies, dim=1)
    pcs_masks = all_energies.unsqueeze(
        2) < lambda1s.unsqueeze(0).unsqueeze(0)
    return pcs_masks


def compute_coverage_loss(svs, lower, upper, projected_ground_truths_minus_mean, lambda1s, lambda2s):
    pcs_masks = compute_pcs_masks(svs, lambda1s)
    lower_with_lambda = lambda2s.unsqueeze(
        0).unsqueeze(1) * lower.unsqueeze(2)
    upper_with_lambda = lambda2s.unsqueeze(
        0).unsqueeze(1) * upper.unsqueeze(2)
    misscoverage = ((lower_with_lambda > projected_ground_truths_minus_mean.unsqueeze(
        2)) | (projected_ground_truths_minus_mean.unsqueeze(2) > upper_with_lambda)).float()
    selected_sigmas = svs.unsqueeze(2) * pcs_masks
    if lambda1s.shape[0] != 1:
        weights = (selected_sigmas[:, :, 1:] ** 2) / \
            (selected_sigmas[:, :, 1:] ** 2).sum(dim=1, keepdim=True)
        weights = torch.cat(
            [torch.zeros_like(weights[:, :, [0]]), weights], dim=2)
    else:
        weights = (selected_sigmas ** 2) / \
            (selected_sigmas ** 2).sum(dim=1, keepdim=True)
    weights = torch.nan_to_num(weights, 0.0)
    losses = (pcs_masks.unsqueeze(3) * weights.unsqueeze(3)
              * misscoverage.unsqueeze(2)).sum(dim=1)
    return losses


def compute_reconstruction_loss(pcs, svs, ground_truths_minus_mean, projected_ground_truths_minus_mean, lambdas_list, pixel_ratio=0.9):
    pcs_masks = compute_pcs_masks(svs, lambdas_list)

    reconstructed_ground_truths = []
    for i in range(lambdas_list.shape[0]):
        reconstructed_ground_truths.append(torch.bmm(pcs * pcs_masks[:, :, i].unsqueeze(1),
                                                     projected_ground_truths_minus_mean.unsqueeze(-1))[:, :, 0])
    reconstructed_ground_truths = torch.stack(
        reconstructed_ground_truths, dim=1)

    losses = (reconstructed_ground_truths -
              ground_truths_minus_mean.unsqueeze(1)).abs()
    losses = torch.quantile(losses, pixel_ratio, dim=2)

    return losses


def coverage_risk(svs, lower, upper, projected_ground_truths_minus_mean, lambda1, lambda2):
    lambda1s = torch.tensor([lambda1], device=svs.device)
    lambda2s = torch.tensor([lambda2], device=svs.device)
    coverage_losses = compute_coverage_loss(
        svs, lower, upper, projected_ground_truths_minus_mean, lambda1s=lambda1s, lambda2s=lambda2s)[:, 0, 0]
    return coverage_losses.mean().item()


def reconstruction_risk(pcs, svs, ground_truths_minus_mean, projected_ground_truths_minus_mean, lambda1, pixel_ratio=0.9):
    lambda1s = torch.tensor([lambda1], device=svs.device)
    reconstruction_losses = compute_reconstruction_loss(
        pcs, svs, ground_truths_minus_mean, projected_ground_truths_minus_mean, lambda1s, pixel_ratio)
    return reconstruction_losses.mean().item()


def interval_size(lower, upper, svs, lambda1):
    lambda1s = torch.tensor([lambda1], device=svs.device)
    pcs_masks = compute_pcs_masks(svs, lambda1s)[:, :, 0]
    intervals = (upper - lower) * pcs_masks
    return intervals.mean().item()


def dimension(svs, lambda1):
    lambda1s = torch.tensor([lambda1], device=svs.device)
    dim = compute_pcs_masks(svs, lambda1s).sum(dim=1).float()
    return dim.mean().item()


def uncertainty_volume(lower, upper, svs, lambda1, eps=1e-10):
    lambda1s = torch.tensor([lambda1], device=svs.device)
    pcs_masks = compute_pcs_masks(svs, lambda1s)[:, :, 0]
    intervals = (upper - lower) * pcs_masks
    volumes = (intervals + eps).log().mean(dim=1).exp() - eps
    return volumes.mean().item()
