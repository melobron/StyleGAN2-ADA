import argparse
from train import TrainStyleGAN

# Arguments
parser = argparse.ArgumentParser(description='Train StyleGAN ADA')

parser.add_argument('--exp_detail', default='Finetune StyleGAN', type=str)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)

# Training parameters
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-3, type=float)

# Model parameters
parser.add_argument('--do_finetune', default=True, type=bool)  # True: finetune / False: start from scratch
parser.add_argument('--model_path', default='./pre-trained/stylegan-256px-new.model', type=str)
parser.add_argument('--feature_loc', default=3, type=int, help='Feature location for D')
parser.add_argument('--freezeD', default=True, type=bool)

# Augmentations parameters
parser.add_argument('--p', default=0, type=float, help='Augmentation Probability')

# Dataset parameters
parser.add_argument('--dataset_name', default='Dog', type=str)  # Dog, Cat, ...

# Transformations
parser.add_argument('--resize', type=bool, default=True)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5))
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5))

opt = parser.parse_args()

# Train StyleGAN
train_StyleGAN = TrainStyleGAN(args=opt)
if opt.do_finetune:
    train_StyleGAN.finetune()
else:
    train_StyleGAN.train()
