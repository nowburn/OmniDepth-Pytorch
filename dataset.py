__author__ = "Marc Eder"

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

import torch
import torch.utils.data

import numpy as np
from skimage import io
import Imath, array

from PIL import Image
import os.path as osp
import os
import re


class OmniDepthDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self, imgs_path):
        self.image_list = []
        for img_name in os.listdir(imgs_path):
            if re.match(r'.+\d\.jpeg', img_name):
                self.image_list.append(imgs_path + "/" + os.path.splitext(img_name)[0])

        self.max_depth = 255.0

    def __getitem__(self, idx):
        '''Load the data'''
        # Select the panos to load
        common_paths = self.image_list[idx]

        # Load the panos
        rgb = self.readRGBPano(common_paths + ".jpeg")
        depth = self.readDepthPanoFromJPEG(common_paths + "_d.jpeg")
        depth_mask = ((depth <= self.max_depth) & (depth > 0.)).astype(np.uint8)
        # Threshold depths
        depth *= depth_mask

        # Make a list of loaded data
        pano_data = [rgb, depth, depth_mask, common_paths]

        # Convert to torch format
        pano_data[0] = torch.from_numpy(pano_data[0].transpose(2, 0, 1)).float()
        pano_data[1] = torch.from_numpy(pano_data[1][None, ...]).float()
        pano_data[2] = torch.from_numpy(pano_data[2][None, ...]).float()
        # Return the set of pano data
        return pano_data

    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readRGBPano(self, path):
        '''Read RGB and normalize to [0,1].'''
        rgb = io.imread(path).astype(np.float32) / 255.
        return rgb

    def readDepthPanoFromJPEG(self, path):
        img = Image.open(path)
        img.load()
        data = np.asarray(img, dtype="float32")
        res = np.reshape(data, (img.height, img.width, 3))[..., 0]
        return res
