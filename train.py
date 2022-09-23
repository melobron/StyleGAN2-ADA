import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

import os
import argparse
import random
import numpy as np
import math
import cv2
from tqdm import tqdm
import json

from dataset import FaceDataset
from models.StyleGAN2 import StyledGenerator, Discriminator
from utils import *


class TrainStyleGAN:
    def __init__(self, args):
        # Arguments
        self.args = args

        # Device
        self.gpu_num = args.gpu_num
        self.device = torch.device('cuda:{}'.format(self.gpu_num) if torch.cuda.is_available() else 'cpu')

        # Random Seeds
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = True

        # Training Parameters
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr

        # Model Parameters
        self.do_finetune = args.do_finetune
        self.model_path = args.model_path
        self.feature_loc = args.feature_loc
        self.freezeD = args.freezeD

        # Transformation Parameters
        self.mean = args.mean
        self.std = args.std

        # Dataset Parameters
        self.dataset_name = args.dataset_name
        self.step = int(math.log2(args.img_size)) - 2

        # Transform
        transform = transforms.Compose(get_transforms(args))

        # Model
        self.G = StyledGenerator().to(self.device)
        self.D = Discriminator(from_rgb_activate=True).to(self.device)

        # FreezeD
        if self.freezeD:
            requires_grad(self.D, False)

        # Load Weight if Fine-Tune
        if self.do_finetune:
            ckpt = torch.load(self.model_path)
            self.G.load_state_dict(ckpt['generator'], strict=False)
            self.D.load_state_dict(ckpt['discriminator'])

        # Dataset
        self.train_dataset = FaceDataset(dataset=self.dataset_name, train=True, transform=transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer
        self.G_optimizer = optim.Adam(self.G.generator.parameters(), lr=self.lr, betas=(0.0, 0.99))
        self.G_optimizer.add_param_group({'params': self.G.style.parameters(), 'lr': self.lr * 0.01, 'mult': 0.01})

        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.0, 0.99))

        # Directories
        self.exp_dir = make_exp_dir('./experiments/')['new_dir']
        self.exp_num = make_exp_dir('./experiments/')['new_dir_num']
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.result_path = os.path.join(self.exp_dir, 'results')

        # Tensorboard
        self.summary = SummaryWriter('runs/exp{}'.format(self.exp_num))

    def prepare(self):
        # Save Paths
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # Save Argument file
        param_file = os.path.join(self.exp_dir, 'params.json')
        with open(param_file, mode='w') as f:
            json.dump(self.args.__dict__, f, indent=4)

    def train(self):
        return

    def finetune(self):
        print(self.device)
        self.prepare()

        for epoch in range(1, self.n_epochs + 1):
            with tqdm(self.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
                for batch, img in enumerate(tepoch):
                    batch_size = img.shape[0]
                    gen_in1, gen_in2 = sample_noise(batch_size)

                    ########### Update D ###########
                    self.D.zero_grad()
                    requires_grad(self.G, False, target_layer=None)

                    if self.freezeD:
                        for loc in range(self.feature_loc):
                            requires_grad(self.D, True, target_layer='progression.{}'.format(8-loc))
                            requires_grad(self.D, True, target_layer='linear')
                    else:
                        requires_grad(self.D, True)

                    D_loss_val, grad_loss_val = self.backward_D(self.G, self.D, img, gen_in1)

                    self.D_optimizer.step()

                    ########### Update G ###########
                    self.G.zero_grad()
                    requires_grad(self.G, True)

                    if self.freezeD:
                        for loc in range(self.feature_loc):
                            requires_grad(self.D, False, target_layer='progression.{}'.format(8-loc))
                            requires_grad(self.D, False, target_layer='linear')
                    else:
                        requires_grad(self.D, False)

                    G_loss_val = self.backward_G(self.G, self.D, gen_in2)

                    self.G_optimizer.step()

                    ########### Save Results ###########
                    tepoch.set_postfix(G_loss=G_loss_val, D_loss=D_loss_val, grad_penalty=grad_loss_val)
                    self.summary.add_scalar('G_loss', G_loss_val, epoch)
                    self.summary.add_scalar('D_loss', D_loss_val, epoch)
                    self.summary.add_scalar('grad_penalty', grad_loss_val, epoch)

            ########### Checkpoints ###########
            if epoch % 100 == 0 or epoch == self.n_epochs:
                torch.save(self.G.state_dict(), os.path.join(self.checkpoint_dir, 'G_{}epochs.pth'.format(epoch)))




