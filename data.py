from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms
import numpy as np

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

for img in train_loader:
    continue
dataset.__getitem__(0)