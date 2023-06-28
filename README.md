# PUQ: Principal Uncertainty Quantification - Leveraging Diffusion Models for Building Tight Uncertainty Regions

This repository contains the official implementation of our paper: [Principal Uncertainty Quantification with Spatial
Correlation for Image Restoration Problems](https://arxiv.org/abs/2305.10124).

**Abstract**:
Uncertainty quantification for inverse problems in imaging has drawn much attention lately. Existing approaches towards this task define uncertainty regions based on probable values per pixel, while ignoring spatial correlations within the image, resulting in an exaggerated volume of uncertainty. In this paper, we propose PUQ (Principal Uncertainty Quantification) -- a novel definition and corresponding analysis of uncertainty regions that takes into account spatial relationships within the image, thus providing reduced volume regions. Using recent advancements in stochastic generative models, we derive uncertainty intervals around principal components of the empirical posterior distribution, forming an ambiguity region that guarantees the inclusion of true unseen values with a user confidence probability. To improve computational efficiency and interpretability, we also guarantee the recovery of true unseen values using only a few principal directions, resulting in ultimately more informative uncertainty regions. Our approach is verified through experiments on image colorization, super-resolution, and inpainting; its effectiveness is shown through comparison to baseline methods, demonstrating significantly tighter uncertainty regions.

<p align="center">
  <img src="images/demo.gif" />
</p>

*Animated figure illustrates uncertainty regions produced by RDA-PUQ*.

**TLDR**:
Given an input image, we aim to predict a linear space called the uncertainty region.
This space is constructed using adaptively-assigned linear uncertainty axes that take into account the spatial dependencies within the image. The uncertainty region is defined by lower and upper bounds on projection values along these axes.
Our method guarantees, by design, that the uncertainty region produced will highly likely contain the unknown ground truth image. Additionally, the linear subspace is guaranteed to restore the ground truth image with a small error while a small number of axes are capable of capturing the majority of the uncertainty within the image.

## Citation

If you find our paper/code helpful, please cite our paper:

    @article{belhasin2023principal,
      title={Principal Uncertainty Quantification with Spatial Correlation for Image Restoration Problems},
      author={Belhasin, Omer and Romano, Yaniv and Freedman, Daniel and Rivlin, Ehud and Elad, Michael},
      journal={arXiv preprint arXiv:2305.10124},
      year={2023}
    }

## Data

Currently, we make the underlying assumption that a diffusion model has been trained and utilized to sample solutions for any selected image restoration task.
For your convenience, we provide an illustrative instance of an [example data folder](puq/data/example) comprising 8 images, each accompanied by 10 samples.
We adopt the following prescribed structure:

    ├── data
    │   ├── ground_truths
    │   │   └── images
    │   │       ├── <image-id>.png
    │   └── samples
    │       ├── <image-id>.png
    │       │   ├── <sample-id>.png

Training of diffusion models for image restoration tasks can be achieved (for example) through the following approaches:

- [SR3](https://arxiv.org/abs/2104.07636) unofficial implementation: [here](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).
- [Palette](https://arxiv.org/abs/2111.05826) unofficial implementation: [here](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models).

## Usage

To clone and install this repository run the following commands:

    git clone https://github.com/omerb01/puq.git
    cd puq

To run PUQ (calibration & inference):

    usage: run.py [-h] --method METHOD --data DATA [--patch-res PATCH_RES] [--test-ratio TEST_RATIO] [--seed SEED] [--gpu GPU]
                [--batch BATCH] [--num-workers NUM_WORKERS] [--alpha ALPHA] [--beta BETA] [--q Q] [--delta DELTA]
                [--num-reconstruction-lambdas NUM_RECONSTRUCTION_LAMBDAS] [--num-coverage-lambdas NUM_COVERAGE_LAMBDAS]
                [--num-pcs-lambdas NUM_PCS_LAMBDAS] [--max-coverage-lambda MAX_COVERAGE_LAMBDA]

    Official implementation of "Principal Uncertainty Quantification with Spatial Correlation for Image Restoration Problems" paper.
    link: https://arxiv.org/abs/2305.10124

    optional arguments:
    -h, --help            show this help message and exit
    --method METHOD       Method to use: e_puq, da_puq or rda_puq.
    --data DATA           Data folder path.
    --patch-res PATCH_RES
                            Patch resolution to use: None to use the full image or int to use patches with 3xpxp dimensions where p
                            is the chosen number.
    --test-ratio TEST_RATIO
                            Test instances ratio out of the data folder.
    --seed SEED           Seed.
    --gpu GPU             GPU index to use, None for CPU.
    --batch BATCH         Batch size to work on.
    --num-workers NUM_WORKERS
                            Number of workers for dataloaders.
    --alpha ALPHA         Coverage guarantee parameter.
    --beta BETA           Reconstruction guarantee parameter.
    --q Q                 Pixels ratio parameter for reconstruction guarantee.
    --delta DELTA         Error level of guarantees.
    --num-reconstruction-lambdas NUM_RECONSTRUCTION_LAMBDAS
                            Number of fine-grained lambdas parameters to be checked for reconstruction guarantee.
    --num-coverage-lambdas NUM_COVERAGE_LAMBDAS
                            Number of fine-grained lambdas parameters to be checked for coverage guarantee.
    --num-pcs-lambdas NUM_PCS_LAMBDAS
                            Number of fine-grained lambdas parameters to be checked for reducting the number of PCs at inference.
    --max-coverage-lambda MAX_COVERAGE_LAMBDA
                            Maximal coverage lambda. (20.0 should be enougth for various tasks).

For example, to run the Exact PUQ procedure on 3x8x8 patchs, run:

    python run.py --method e_puq --data /path-to-data-folder/ --patch-res 8 --gpu 0

## Acknowledgments

<img src="images/verily.png" alt="verily" width="30%" />

This research was partially supported by Verily Life Sciences (formerly Google Life Sciences).