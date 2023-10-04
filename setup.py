from setuptools import setup, find_packages

setup(name='PUQ-Principal-Uncertainty-Quantification',
      version='1.0.0',
      description='Official implementation of "Principal Uncertainty Quantification with Spatial Correlation for Image Restoration Problems" paper.',
      url='https://github.com/omerb01/puq',
      packages=find_packages(),
      install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
      ])