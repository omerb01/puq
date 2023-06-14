# PUQ: Principal Uncertainty Quantification - Leveraging Diffusion Models for Building Tight Uncertainty Regions

This repository contains the official implementation of our paper: [Principal Uncertainty Quantification with Spatial
Correlation for Image Restoration Problems](https://arxiv.org/abs/2305.10124).

**Abstract**:
Uncertainty quantification for inverse problems in imaging has drawn much attention lately. Existing approaches towards this task define uncertainty regions based on probable values per pixel, while ignoring spatial correlations within the image, resulting in an exaggerated volume of uncertainty. In this paper, we propose PUQ (Principal Uncertainty Quantification) -- a novel definition and corresponding analysis of uncertainty regions that takes into account spatial relationships within the image, thus providing reduced volume regions. Using recent advancements in stochastic generative models, we derive uncertainty intervals around principal components of the empirical posterior distribution, forming an ambiguity region that guarantees the inclusion of true unseen values with a user confidence probability. To improve computational efficiency and interpretability, we also guarantee the recovery of true unseen values using only a few principal directions, resulting in ultimately more informative uncertainty regions. Our approach is verified through experiments on image colorization, super-resolution, and inpainting; its effectiveness is shown through comparison to baseline methods, demonstrating significantly tighter uncertainty regions.

<p align="center">
  <img src="images/demo.gif" />
</p>

*Animated figure demonstrates the operation of RDA-PUQ*.

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

## Acknowledgments

<img src="images/verily.png" alt="verily" width="30%" />

This research was supported by Verily Life Sciences (formerly Google Life Sciences).