from arch_unet import UNet
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import os
import glob


# Load the model
model = UNet(1, 1, 96, True)

# Load the weights
model.load_state_dict(torch.load('results/unet_n2n/2024-05-23-14-13/best.pth'))
model.eval()
model.cuda()


# Inference
from data import ImageDataset_Val
from torch.utils.data import DataLoader

train_dataset = ImageDataset_Val('dataset/train')
train_loader = DataLoader(dataset=train_dataset,
                          num_workers=8,
                          batch_size=1, 
                          shuffle=False,
                          pin_memory=False,
                          drop_last=False)

# inference loop
monitor = tqdm(train_loader, total=len(train_loader), desc='Infer')
with torch.no_grad():
    for image, path in monitor:
        image = image.cuda()
        out = model(image)

        # compute save path
        rel_path = os.path.relpath(path[0], 'dataset/train')
        out_path = os.path.join('results/unet_n2n/2024-05-23-14-13/infer', rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # save image with the new path
        save_image(out, 'results/unet_n2n/2024-05-23-14-13/infer/' + os.path.basename(path[0]), format='jpg')