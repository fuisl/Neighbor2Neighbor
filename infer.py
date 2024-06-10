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

device = 'cuda:1'

# Load the model
model = UNet(1, 1, 96, True)

path = 'checkpoint/nbr2nbr/lightning_logs/version_6/checkpoints/epoch=49-val_loss=0.0011.ckpt'
checkpoint = torch.load(path, map_location='cuda:1')
new_state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
model.load_state_dict(new_state_dict)
# Load the weights
# model.load_state_dict(torch.load('results/unet_n2n/2024-05-23-14-13/best.pth'))
model.eval()
model.to(device)


# Inference
from data import ImageDataset_Val
from torch.utils.data import DataLoader

train_dataset = ImageDataset_Val('/data1tb/trietdao/dataset_OCT/self_fusion_1c')
train_loader = DataLoader(dataset=train_dataset,
                          num_workers=8,
                          batch_size=1, 
                          shuffle=False,
                          pin_memory=False,
                          drop_last=False)

# inference loop
output_base = 'results/unet_n2n/lightning_logs/version_6/infer'
monitor = tqdm(train_loader, total=len(train_loader), desc='Infer')
with torch.no_grad():
    for image, path in monitor:
        image = image.to(device)
        out = model(image)

        # compute save path
        rel_path = os.path.relpath(path[0], 'dataset/train')
        out_path = os.path.join(output_base, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # save image with the new path
        save_image(out, out_path, format='JPEG')
        monitor.set_description(f"Processing {os.path.basename(path[0])}")