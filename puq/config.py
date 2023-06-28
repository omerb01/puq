from argparse import Namespace

opt = Namespace()

opt.seed = 42
# opt.data_dir = '/home/omerbe/puq/puq/data'
opt.data_dir = '/mnt/paper_celeba/parse'
opt.num_workers = 12

opt.test_ratio = 0.5

opt.gpu = 3                                         # GPU id or None for CPU
opt.batch_size = 64

# Calibration
opt.alpha = 0.1
opt.beta = 0.1
opt.q = 0.9
opt.delta = 0.1
opt.max_coverage_lambda = 20
opt.num_coverage_lambdas = 100
opt.num_reconstruction_lambdas = 100
opt.num_max_pcs_lambdas = 20


opt.patch_res = None                                   # patch resolution or None