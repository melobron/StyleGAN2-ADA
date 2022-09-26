import os
import random

import torch
from torchvision.transforms import transforms

# from metrics.metric import compute_fid,


################################# Path & Directory #################################
def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)


def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def make_exp_dir(main_dir):
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    dirs = os.listdir(main_dir)
    dir_nums = []
    for dir in dirs:
        dir_num = int(dir[3:])
        dir_nums.append(dir_num)
    if len(dirs) == 0:
        new_dir_num = 1
    else:
        new_dir_num = max(dir_nums) + 1
    new_dir_name = 'exp{}'.format(new_dir_num)
    new_dir = os.path.join(main_dir, new_dir_name)
    return {'new_dir': new_dir, 'new_dir_num': new_dir_num}


################################# Transforms #################################
def get_transforms(args):
    transform_list = [transforms.ToTensor()]
    if args.resize:
        transform_list.append(transforms.Resize((args.img_size, args.img_size)))
    if args.normalize:
        transform_list.append(transforms.Normalize(mean=args.mean, std=args.std))
    return transform_list


################################# Finetune Functions #################################
# Override requires_grad function
def requires_grad(model, flag=True, target_layer=None):
    for name, param in model.named_parameters():
        if target_layer is None:  # every layer
            param.requires_grad = flag
        elif target_layer in name:  # target layer
            param.requires_grad = flag


# Sampling Noise
def sample_noise(batch_size):
    if random.random() < 0.9:
        gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(4, batch_size, 512, device='cuda').chunk(4, 0)
        gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
        gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

    else:
        gen_in1, gen_in2 = torch.randn(2, batch_size, 512, device='cuda').chunk(2, 0)
        gen_in1 = gen_in1.squeeze(0)
        gen_in2 = gen_in2.squeeze(0)

    return gen_in1, gen_in2


################################# ETC #################################
def denorm(tensor, mean=0.5, std=0.5, max_pixel=1.):
    return std*max_pixel*tensor + mean*max_pixel


def tensor_to_numpy(x):
    x = x.detach().cpu().numpy()
    if x.ndim == 4:
        return x.transpose((0, 2, 3, 1))
    elif x.ndim == 3:
        return x.transpose((1, 2, 0))
    elif x.ndim == 2:
        return x
    else:
        raise