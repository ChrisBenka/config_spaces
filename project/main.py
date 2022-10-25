from config_spaces.project.model import SegNet
from config_spaces.project.dataset import ConfigSpaceDataset
from torchvision import transforms
from torchvision.transforms import ToTensor,Grayscale,Resize,Compose
data_path = "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/scripts/data"
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis("off")

if __name__ == '__main__':
    get_rid_alpha = transforms.Lambda(lambda x: x[:3])
    transforms = Compose([ToTensor(),get_rid_alpha,Resize(512),Grayscale()])
    dataset = ConfigSpaceDataset(data_path,transform=transforms)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset,batch_size=12,shuffle=True)
