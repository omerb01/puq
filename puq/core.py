import logging
from tqdm import tqdm

import torch
import numpy as np

import metrics
from data.data import DiffusionSamplesDataset, GroundTruthsDataset
from utils import statistics, misc


class PUQUncertaintyRegion:
    '''
    Base class for PUQ procedures.
    '''

    def __init__(self, opt):
        self.opt = opt
        self.calibrated = False
        self.max_pcs = None

        self.device = torch.device(
            f'cuda:{self.opt.gpu}' if self.opt.gpu is not None else 'cpu')

    def _init_lambdas(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_batch_losses(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_p_values(self, *args, **kwargs):
        raise NotImplementedError

    def _calibration_failed_message(self, *args, **kwargs):
        raise NotImplementedError

    def _assign_lambdas(self, *args, **kwargs):
        raise NotImplementedError

    def _apply_lambdas(self, *args, **kwargs):
        raise NotImplementedError

    def _eval_metrics(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_intervals(self, samples):

        # Conditional mean
        mu = samples.mean(dim=1)

        # Principal componenets and singular values (importance weights)
        samples_minus_min = samples - mu.unsqueeze(1)
        _, svs, pcs = torch.svd(samples_minus_min)

        # Lower and upper bounds
        projections = torch.bmm(samples_minus_min, pcs)
        lower = torch.quantile(projections, self.opt.alpha/2, dim=1)
        upper = torch.quantile(projections, 1-(self.opt.alpha/2), dim=1)

        return mu, pcs, svs, lower, upper

    def approximation(self, dataset: DiffusionSamplesDataset, verbose=True):
        if verbose:
            logging.info('Applying approximation phase...')

        K = dataset.num_samples_per_image

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=K*self.opt.batch,
            num_workers=self.opt.num_workers,
            shuffle=False,
            collate_fn=None if self.opt.patch_res is None else misc.concat_patches
        )

        all_mu = []
        all_pcs = []
        all_svs = []
        all_lower = []
        all_upper = []

        dataloader = tqdm(dataloader) if verbose else dataloader
        for samples in dataloader:
            samples = samples.view(samples.shape[0]//K, K, -1).to(self.device)
            mu, pcs, svs, lower, upper = self._compute_intervals(samples)
            all_mu.append(mu.cpu())
            all_pcs.append(pcs.cpu())
            all_svs.append(svs.cpu())
            all_lower.append(lower.cpu())
            all_upper.append(upper.cpu())

        all_mu = torch.cat(all_mu, dim=0)
        all_pcs = torch.cat(all_pcs, dim=0)
        all_svs = torch.cat(all_svs, dim=0)
        all_lower = torch.cat(all_lower, dim=0)
        all_upper = torch.cat(all_upper, dim=0)

        return all_mu, all_pcs, all_svs, all_lower, all_upper

    def _compute_losses(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset, verbose=True):
        all_mu, all_pcs, all_svs, all_lower, all_upper = self.approximation(
            samples_dataset, verbose=verbose)

        dataloader = torch.utils.data.DataLoader(
            ground_truths_dataset,
            batch_size=self.opt.batch,
            num_workers=self.opt.num_workers,
            shuffle=False,
            collate_fn=None if self.opt.patch_res is None else misc.concat_patches
        )
        dataloader = iter(dataloader)

        num_images = all_mu.shape[0]
        if self.opt.patch_res is None:
            step = self.opt.batch
        else:
            step = self.opt.batch * samples_dataset.num_patches_per_sample

        all_losses = []
        for i in range(0, num_images, step):
            mu = all_mu[i:i+step].to(self.device)
            pcs = all_pcs[i:i+step].to(self.device)
            svs = all_svs[i:i+step].to(self.device)
            lower = all_lower[i:i+step].to(self.device)
            upper = all_upper[i:i+step].to(self.device)

            # Project ground truths
            ground_truths = dataloader.next().flatten(1).to(self.device)

            ground_truths_minus_mean = ground_truths - mu
            projected_ground_truths_minus_mean = torch.bmm(
                ground_truths_minus_mean.unsqueeze(1), pcs)[:, 0]

            # Compute losses
            losses = self._compute_batch_losses(
                pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean)
            all_losses.append(losses.cpu())
        all_losses = torch.cat(all_losses, dim=1)

        return all_losses

    def calibration(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset, verbose=True):
        loss_table = self._compute_losses(
            samples_dataset, ground_truths_dataset, verbose=verbose)

        if verbose:
            logging.info('Applying calibration phase...')

        # Compute risks
        risks = loss_table.mean(dim=1)

        # Compute p-values
        pvals = self._compute_p_values(risks, samples_dataset.num_images)

        # Find valid lambdas using bonferroni correction
        valid_indices = statistics.bonferroni_search(
            pvals, self.opt.delta, downsample_factor=20)

        # Assign lambdas that minimize the uncertainty volume
        if valid_indices.shape[0] == 0:
            misc.handdle_error(self._calibration_failed_message(
                samples_dataset.num_images))
        self.max_pcs = samples_dataset.num_samples_per_image
        self._assign_lambdas(valid_indices, verbose=verbose)
        self.calibrated = True

    def inference(self, samples):
        assert self.calibrated

        if samples.shape[1] != self.max_pcs:
            misc.handdle_error(
                f'PUQ was calibrated for K={self.max_pcs} samples per image, but {self.max_pcs} samples were given.')

        mu, pcs, svs, lower, upper = self._compute_intervals(samples)
        return self._apply_lambdas(mu, pcs, svs, lower, upper)

    def eval(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset, verbose=True):
        assert self.calibrated
        
        if verbose:
            logging.info('Applying evaluation...')

        K = samples_dataset.num_samples_per_image

        samples_dataloader = torch.utils.data.DataLoader(
            samples_dataset,
            batch_size=K*self.opt.batch,
            num_workers=self.opt.num_workers,
            shuffle=False,
            collate_fn=None if self.opt.patch_res is None else misc.concat_patches
        )

        ground_truths_dataloader = torch.utils.data.DataLoader(
            ground_truths_dataset,
            batch_size=self.opt.batch,
            num_workers=self.opt.num_workers,
            shuffle=False,
            collate_fn=None if self.opt.patch_res is None else misc.concat_patches
        )

        results_list = []

        dataloader = zip(samples_dataloader, ground_truths_dataloader)
        dataloader = tqdm(dataloader, total=len(
            ground_truths_dataloader)) if verbose else dataloader
        for samples, ground_truths in dataloader:

            samples = samples.view(samples.shape[0]//K, K, -1).to(self.device)
            ground_truths = ground_truths.flatten(1).to(self.device)

            mu, pcs, svs, lower, upper = self._compute_intervals(samples)

            ground_truths_minus_mean = ground_truths - mu
            projected_ground_truths_minus_mean = torch.bmm(
                ground_truths_minus_mean.unsqueeze(1), pcs)[:, 0]

            results_list.append(
                self._eval_metrics(mu, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean))

        results = {}
        for c in results_list[0]:
            metric = 0
            for d in results_list:
                metric += d[c]
            metric /= len(results_list)
            results[c] = metric
        
        if verbose:
            logging.info(results)
        
        return results


class EPUQUncertaintyRegion(PUQUncertaintyRegion):
    '''
    Implementation for Exact-PUQ (E-PUQ) procedure.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coverage_lambda = None

    def calibration(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset):
        if samples_dataset.dim > samples_dataset.num_samples_per_image:
            misc.handdle_error(
                f'Applying Exact PUQ requires at least {samples_dataset.dim} samples per image but only {samples_dataset.num_samples_per_image} were given.')
        return super().calibration(samples_dataset, ground_truths_dataset)

    def _init_lambdas(self):
        return torch.linspace(1, self.opt.max_coverage_lambda, self.opt.num_coverage_lambdas).to(self.device)

    def _compute_batch_losses(self, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean):
        lambdas = self._init_lambdas()
        coverage_losses = metrics.compute_coverage_loss(
            svs, lower, upper, projected_ground_truths_minus_mean, lambda1s=torch.tensor([1], device=lambdas.device), lambda2s=lambdas)
        return coverage_losses[:, 0].unsqueeze(0)

    def _compute_p_values(self, risks, num_images):
        pvals = np.array([statistics.hb_p_value(r_hat, num_images, self.opt.alpha)
                         for r_hat in risks[0].flatten().cpu()])
        return pvals

    def _calibration_failed_message(self, num_images):
        return f'Calibration failed: cannot guarantee alpha={self.opt.alpha}, delta={self.opt.delta} using n={num_images} calibration instances.'

    def _assign_lambdas(self, valid_indices, verbose=True):
        lambdas = self._init_lambdas()
        valid_lambdas = lambdas[valid_indices]

        lamb = valid_lambdas.min()
        if verbose:
            logging.info(f'Successfully calibrated: lambda={lamb}')

        self.coverage_lambda = lamb.item()

    def _apply_lambdas(self, mu, pcs, svs, lower, upper):
        return mu, pcs, svs, self.coverage_lambda * lower, self.coverage_lambda * upper

    def _eval_metrics(self, mu, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean):
        return {
            "coverage_risk": metrics.coverage_risk(svs, lower, upper, projected_ground_truths_minus_mean, lambda1=1.0, lambda2=self.coverage_lambda),
            "interval_size": metrics.interval_size(lower, upper, mu.shape[1]),
            "uncertainty_volume": metrics.uncertainty_volume(lower, upper, mu.shape[1])
        }


class DAPUQUncertaintyRegion(PUQUncertaintyRegion):
    '''
    Implementation for Dimension-Adaptive-PUQ (DA-PUQ) procedure.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_lambda = None
        self.coverage_lambda = None

    def _init_lambdas(self):
        lambda1s = torch.linspace(
            0, 1, self.opt.num_reconstruction_lambdas).to(self.device)
        lambda2s = torch.linspace(
            1, self.opt.max_coverage_lambda, self.opt.num_coverage_lambdas).to(self.device)
        return lambda1s, lambda2s

    def _compute_batch_losses(self, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean):
        lambdas = self._init_lambdas()
        lambda1s, lambda2s = lambdas[:2]
        reconstruction_losses = metrics.compute_reconstruction_loss(
            pcs, svs, ground_truths_minus_mean, projected_ground_truths_minus_mean, lambda1s, pixel_ratio=self.opt.q)
        reconstruction_losses = reconstruction_losses.unsqueeze(
            2).repeat(1, 1, lambda2s.shape[0])
        coverage_losses = metrics.compute_coverage_loss(
            svs, lower, upper, projected_ground_truths_minus_mean, lambda1s, lambda2s)
        return torch.stack([reconstruction_losses, coverage_losses], dim=0)

    def _compute_p_values(self, risks, num_images):
        pval1s = np.array([statistics.hb_p_value(r_hat, num_images, self.opt.beta)
                           for r_hat in risks[0].flatten().cpu()])
        pval2s = np.array([statistics.hb_p_value(r_hat, num_images, self.opt.alpha)
                           for r_hat in risks[1].flatten().cpu()])
        pvals = np.maximum(pval1s, pval2s)
        pvals = np.flip(np.flip(pvals.reshape(
            risks.shape[1], risks.shape[2]), axis=0), axis=1).reshape(-1)
        return pvals

    def _calibration_failed_message(self, num_images):
        return f'Calibration failed: cannot guarantee alpha={self.opt.alpha}, beta={self.opt.beta}, q={self.opt.q}, delta={self.opt.delta} using n={num_images} calibration instances.'

    def _assign_lambdas(self, valid_indices, verbose=True):
        lambda1s, lambda2s = self._init_lambdas()

        valid_lambdas = torch.cartesian_prod(
            lambda1s.flip(0), lambda2s.flip(0))[valid_indices]
        lambda1, _ = valid_lambdas[-1]
        lambda2 = valid_lambdas[valid_lambdas[:, 0] == lambda1, 1].min()

        if verbose:
            logging.info(
                f'Successfully calibrated: lambda1={lambda1}, lambda2={lambda2}')

        self.reconstruction_lambda = lambda1.item()
        self.coverage_lambda = lambda2.item()

    def _apply_lambdas(self, mu, pcs, svs, lower, upper):
        pcs_masks = metrics.compute_pcs_masks(
            svs, torch.tensor([self.reconstruction_lambda], device=svs.device))
        pcs_masks = pcs_masks[:, :, 0]
        pcs = [pcs[:, :, :mask.sum()] for mask in pcs_masks]
        svs = [svs[:, :mask.sum()] for mask in pcs_masks]
        lower = [self.coverage_lambda * lower[:, :mask.sum()]
                 for mask in pcs_masks]
        upper = [self.coverage_lambda * upper[:, :mask.sum()]
                 for mask in pcs_masks]

        return mu, pcs, svs, lower, upper

    def _eval_metrics(self, mu, pcs, svs, lower, upper, ground_truths_minus_mean, projected_ground_truths_minus_mean):
        return {
            "coverage_risk": metrics.coverage_risk(svs, lower, upper, projected_ground_truths_minus_mean, lambda1=self.reconstruction_lambda, lambda2=self.coverage_lambda),
            "reconstruction_risk": metrics.reconstruction_risk(pcs, svs, ground_truths_minus_mean, projected_ground_truths_minus_mean, lambda1=self.reconstruction_lambda, pixel_ratio=self.opt.q),
            "interval_size": metrics.interval_size(lower, upper, mu.shape[1]),
            "dimension": metrics.dimension(svs, lambda1=self.reconstruction_lambda),
            "uncertainty_volume": metrics.uncertainty_volume(lower, upper, mu.shape[1])
        }


class RDAPUQUncertaintyRegion(DAPUQUncertaintyRegion):
    '''
    Implementation for Reduced Dimension-Adaptive PUQ (RDA-PUQ) procedure.
    '''

    def _init_lambdas(self):
        lambda1s = torch.linspace(
            0, 1, self.opt.num_reconstruction_lambdas).to(self.device)
        lambda2s = torch.linspace(
            1, self.opt.max_coverage_lambda, self.opt.num_coverage_lambdas).to(self.device)
        lambda3s = torch.linspace(
            1 / self.opt.num_pcs_lambdas, 1, self.opt.num_pcs_lambdas).to(self.device)
        return lambda1s, lambda2s, lambda3s

    @staticmethod
    def _lambda3_to_K(lambda3s, K_max):
        Ks = (K_max * lambda3s).long().tolist()
        return Ks

    def _compute_losses(self, samples_dataset: DiffusionSamplesDataset, ground_truths_dataset: GroundTruthsDataset, verbose=True):
        _, _, lambda3s = self._init_lambdas()
        Ks = RDAPUQUncertaintyRegion._lambda3_to_K(
            lambda3s, samples_dataset.num_samples_per_image)

        if verbose:
            logging.info('Applying approximation phase...')
            Ks = tqdm(Ks)

        all_losses = []
        for K in Ks:
            subset = samples_dataset.create_subset(num_samples_per_image=K)
            loss_table = super()._compute_losses(subset, ground_truths_dataset, verbose=False)
            all_losses.append(loss_table)
        return torch.stack(all_losses, dim=-1)

    def _compute_p_values(self, risks, num_images):
        pval1s = np.array([statistics.hb_p_value(r_hat, num_images, self.opt.beta)
                           for r_hat in risks[0].flatten().cpu()])
        pval2s = np.array([statistics.hb_p_value(r_hat, num_images, self.opt.alpha)
                           for r_hat in risks[1].flatten().cpu()])
        pvals = np.maximum(pval1s, pval2s)
        pvals = np.flip(np.flip(np.flip(pvals.reshape(
            risks.shape[1], risks.shape[2], risks.shape[3]), axis=0), axis=1), axis=2).reshape(-1)
        return pvals

    def _assign_lambdas(self, valid_indices, verbose=True):
        lambda1s, lambda2s, lambda3s = self._init_lambdas()

        valid_lambdas = torch.cartesian_prod(lambda1s.flip(
            0), lambda2s.flip(0), lambda3s.flip(0))[valid_indices]

        lambda3 = valid_lambdas[:, 2].min()
        lambda1 = valid_lambdas[valid_lambdas[:, 2] == lambda3, 0].min()
        lambda2 = valid_lambdas[(valid_lambdas[:, 0] == lambda1) & (
            valid_lambdas[:, 2] == lambda3), 1].min()

        _, lambda2, lambda3 = valid_lambdas[-1]
        lambda1 = valid_lambdas[(valid_lambdas[:, 1] == lambda2) & (
            valid_lambdas[:, 2] == lambda3), 0].min()

        if verbose:
            logging.info(
                f'Successfully calibrated: lambda1={lambda1}, lambda2={lambda2}, lambda3={lambda3}')

        self.reconstruction_lambda = lambda1.item()
        self.coverage_lambda = lambda2.item()
        self.max_pcs = RDAPUQUncertaintyRegion._lambda3_to_K(
            lambda3, self.max_pcs)
