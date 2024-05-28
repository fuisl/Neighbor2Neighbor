from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms
import numpy as np
from typing import Optional
import torch
import pytorch_lightning as pl

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = glob.glob(self.root_dir + "/**/*.jpg", recursive=True)
        self.img_paths.sort()
        print('Found {} images for training'.format(len(self.img_paths)))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        fetch_img = self.img_paths[index]

        img = Image.open(fetch_img)

        # # convert to log scale
        # img = np.array(img)
        # img = np.log1p(img)
        # img = Image.fromarray((img * 255).astype(np.uint8))

        transformer = transforms.Compose([
            transforms.Resize((256, 256)),   # Resize the image to 256x256
            transforms.ToTensor()            # Convert the image to a tensor
        ])

        img = transformer(img)

        return img
    
class ImageDataset_Val(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = glob.glob(self.root_dir + "/**/*.jpg", recursive=True)
        self.img_paths.sort()
        print('Found {} images for training'.format(len(self.img_paths)))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        fetch_img = self.img_paths[index]

        img = Image.open(fetch_img)

        # # convert to log scale
        # img = np.array(img)
        # img = np.log1p(img)
        # img = Image.fromarray((img * 255).astype(np.uint8))

        transformer = transforms.Compose([
            transforms.Resize((256, 256)),   # Resize the image to 256x256
            transforms.ToTensor()            # Convert the image to a tensor
        ])

        img = transformer(img)

        return img, self.img_paths[index]
    
dataset = ImageDataset('/data1tb/haiduong/n2n/dataset/train')
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=dataset,
                            num_workers=8,
                            batch_size=10,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)

class OCT_Data(pl.LightningDataModule):
    def __init__(self,         
        batch_size: int = 10,
        workers: int = 5,
        train_data: str = "/home/fuisloy/data1tb/Neighbor2Neighbor/dataset/train",
        val_data: str = "/home/fuisloy/data1tb/Neighbor2Neighbor/dataset/train",
        test_data: str = "/home/fuisloy/data1tb/Neighbor2Neighbor/dataset/train",
        ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
    
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ImageDataset(root_dir=self.train_data)
            self.val_dataset = ImageDataset(root_dir=self.val_data)
        if stage == "test" or stage is None:
            self.test_dataset = ImageDataset(root_dir=self.test_data)

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
            persistent_workers=True, 
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader( 
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.workers,
            persistent_workers=True, 
        )
        return val_loader
    def test_dataloader(self):
        return self.val_dataloader()

for img in train_loader:
    continue
dataset.__getitem__(0)