import os
import logging

import torch
from torch.utils.data.dataloader import default_collate


def setup_logging(log_file='log.txt', resume=True, dummy=False):
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_file,
            filemode=file_mode,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def handdle_error(message):
    logging.error(message)
    raise Exception(message)


def split_image_into_patches(image, patch_res):
    patches = torch.stack([torch.stack(t.split(patch_res, dim=1), dim=0)
                          for t in image.split(patch_res, dim=2)], dim=1)
    return patches.view(patches.shape[0] * patches.shape[1], *patches.shape[2:])


def concat_patches(batch):
    batch = default_collate(batch)
    batch = batch.transpose(0, 1)
    shape = batch.shape
    batch = batch.reshape(shape[0] * shape[1], *shape[2:])
    return batch