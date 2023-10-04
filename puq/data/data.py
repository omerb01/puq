from collections import defaultdict

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T

from utils import misc


class DiffusionSamplesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, calibration=True, transform=None):
        super().__init__()
        self.opt = opt
        self.calibration = calibration
        self.transform = transform

        self.samples = ImageFolder(
            f'{self.opt.data}/samples', transform=self.transform)

        random_indices = torch.randperm(
            len(self.samples.classes), generator=torch.Generator().manual_seed(self.opt.seed))
        if calibration:
            self.image_ids = random_indices[:int(
                random_indices.shape[0]*(1 - self.opt.test_ratio))]
        else:
            self.image_ids = random_indices[int(
                random_indices.shape[0]*(1 - self.opt.test_ratio)):]

        self.imageid_to_sampleid = defaultdict(list)
        samples_ids = []
        for i, (_, c) in enumerate(self.samples.samples):
            if c in self.image_ids:
                samples_ids.append(i)
                self.imageid_to_sampleid[c].append(len(samples_ids) - 1)
        self.samples = torch.utils.data.Subset(self.samples, samples_ids)

        self.num_images = len(self.image_ids)
        self.num_samples_per_image = len(self.samples) // self.num_images

        example_image = ImageFolder(
            f'{self.opt.data}/samples', transform=T.ToTensor())[0][0]
        if self.opt.patch_res is None:
            self.dim = int(torch.tensor(example_image.shape).prod().item())
        else:
            self.dim = 3 * (self.opt.patch_res ** 2)
            self.num_patches_per_sample = (
                example_image.shape[-1] // self.opt.patch_res) ** 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, _ = self.samples[index]
        if self.opt.patch_res is not None:
            sample = misc.split_image_into_patches(sample, self.opt.patch_res)
        return sample
    
    def create_subset(self, num_samples_per_image):
        assert num_samples_per_image <= self.num_samples_per_image
        subset = DiffusionSamplesDataset(self.opt, self.calibration, self.transform)

        imageid_to_sampleid = {}
        samples_ids = []
        for c in subset.imageid_to_sampleid:
            ids = subset.imageid_to_sampleid[c]
            rand_indices = torch.randperm(len(ids), generator=torch.Generator().manual_seed(self.opt.seed))[:num_samples_per_image]
            ids = torch.tensor(ids, dtype=torch.long)[rand_indices].tolist()
            imageid_to_sampleid[c] = list(range(len(samples_ids), len(samples_ids) + len(ids)))
            samples_ids.extend(ids)
        
        subset.samples = torch.utils.data.Subset(subset.samples, samples_ids)
        subset.imageid_to_sampleid = imageid_to_sampleid
        subset.num_samples_per_image = num_samples_per_image
        return subset


class GroundTruthsDataset(torch.utils.data.Dataset):
    def __init__(self, opt, samples_dataset, transform=None):
        self.opt = opt
        indices = torch.sort(samples_dataset.image_ids)[0]
        self.dataset = ImageFolder(
            f'{self.opt.data}/ground_truths', transform=transform)
        self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        if self.opt.patch_res is not None:
            image = misc.split_image_into_patches(image, self.opt.patch_res)
        return image


class DiffusionSamplesDataLoader(torch.utils.data.DataLoader):
    def __init__(self, samples_dataset, batch_size, patch_res, K, num_workers, shuffle):
        super().__init__(
            samples_dataset,
            batch_size=K*batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=misc.SamplesGlobalCollator(K) if patch_res is None else misc.SamplesPatchesCollator(K)
        )

class GroundTruthsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, ground_truths_dataset, batch_size, patch_res, num_workers, shuffle):
        super().__init__(
            ground_truths_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=None if patch_res is None else misc.GroundTruthsPatchesCollator()
        )