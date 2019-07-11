import visdom

from omnidepth_trainer import OmniDepthTrainer
from network import *
from criteria import *
from dataset import *
from util import mkdirs, set_caffe_param_mult

import os.path as osp

# --------------
# PARAMETERS
# --------------
network_type = 'RectNet'  # 'RectNet' or 'UResNet'
experiment_name = 'omnidepth'
train_file_list = './data/training'  # File with list of training files
val_file_list = './data/validation'  # File with list of validation files
max_steps_file = './data/num_samples.txt'  # File with number of max_steips for progress show
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = None
checkpoint_path = osp.join(checkpoint_dir, 'epoch_latest.pth')
load_weights_only = True
batch_size = 12
num_workers = 10
lr = 2e-4
step_size = 3
lr_decay = 0.5
num_epochs = 9999
validation_freq = 1
visualization_freq = 5
validation_sample_freq = -1
device_ids = [0]
num_samples = 0

# -------------------------------------------------------
# Fill in the rest
vis = visdom.Visdom()
env = experiment_name
device = torch.device('cuda', device_ids[0])

# UResNet
if network_type == 'UResNet':
    model = UResNet()
    alpha_list = [0.445, 0.275, 0.13]
    beta_list = [0.15, 0., 0.]
# RectNet
elif network_type == 'RectNet':
    model = RectNet()
    alpha_list = [0.535, 0.272]
    beta_list = [0.134, 0.068, ]
else:
    assert False, 'Unsupported network type'

# Make the checkpoint directory
mkdirs(checkpoint_dir)

# Read_Write max_steps
if load_weights_only:
    n = 0
    for img_name in os.listdir(train_file_list):
        if re.match(r'.+d\.jpeg', img_name):
            n += 1
    num_samples = n
    with open(max_steps_file, 'w') as f:
        f.write(str(num_samples))
else:
    with open(max_steps_file, 'r') as f:
        num_samples = int(f.read())

# -------------------------------------------------------
# Set up the training routine
network = nn.DataParallel(
    model.float(),
    device_ids=device_ids).to(device)

train_dataloader = torch.utils.data.DataLoader(
    dataset=OmniDepthDataset(train_file_list),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=False)

val_dataloader = torch.utils.data.DataLoader(
    dataset=OmniDepthDataset(val_file_list),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False)

criterion = MultiScaleL2Loss(alpha_list, beta_list)

# Set up network parameters with Caffe-like LR multipliers
param_list = set_caffe_param_mult(network, lr, 0)
optimizer = torch.optim.Adam(
    params=param_list,
    lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=step_size,
                                            gamma=lr_decay)

trainer = OmniDepthTrainer(
    experiment_name,
    network,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    checkpoint_dir,
    device,
    visdom=[vis, env],
    scheduler=scheduler,
    num_epochs=num_epochs,
    validation_freq=validation_freq,
    visualization_freq=visualization_freq,
    validation_sample_freq=validation_sample_freq,
    num_samples=num_samples)

trainer.train(checkpoint_path, load_weights_only)
