import visdom

from omnidepth_trainer import OmniDepthTrainer
from network import *
from dataset import *

import os.path as osp
from criteria import *

# --------------
# PARAMETERS
# --------------
network_type = 'RectNet'  # 'RectNet' or 'UResNet'
experiment_name = 'omnidepth'
val_file_list = '/home/nowburn/python_projects/cv/OmniDepth/data/show/tes/'  # List of evaluation files
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = None
checkpoint_path = osp.join(checkpoint_dir, 'epoch_latest.pth')
batch_size = 1
num_workers = 8
validation_sample_freq = -1
device_ids = [0]

# -------------------------------------------------------
# Fill in the rest
vis = visdom.Visdom()
env = 'predict'
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
    assert True, 'Unsupported network type'

criterion = MultiScaleL2Loss(alpha_list, beta_list)
# -------------------------------------------------------
# Set up the training routine
network = nn.DataParallel(
    model.float(),
    device_ids=device_ids).to(device)

val_dataloader = torch.utils.data.DataLoader(
    dataset=OmniDepthDataset(val_file_list),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False)

trainer = OmniDepthTrainer(
    experiment_name,
    network,
    None,
    val_dataloader,
    criterion,
    None,
    checkpoint_dir,
    device,
    visdom=[vis, env],
    validation_sample_freq=validation_sample_freq)

trainer.predict(checkpoint_path)

