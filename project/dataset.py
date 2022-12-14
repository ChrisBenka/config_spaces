import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from PIL import Image


class ConfigSpaceDataset(Dataset):
    def __init__(self, root_dir, workspace_transform=transforms.Compose([ToTensor(), Resize(256)]), configspace_transform=transforms.Compose([ToTensor(), Resize(256)])):
        self.workspace_dir = root_dir + "/workspace/"
        self.configspace_dir = root_dir + "/cobs/"

        assert os.path.exists(self.workspace_dir), "Expected 2 folders to be present in root directory, worksapce/ and " \
                                                   "cobs/. Missing worksapce/ "
        assert os.path.exists(
            self.configspace_dir), "Expected 2 folders to be present in root directory, worksapce/ and " \
                                   "cobs/. Missing cobs/ "

        self.num_workspace_images = len(
            [name for name in os.listdir(self.workspace_dir) if os.path.isfile(os.path.join(self.workspace_dir, name))])
        self.num_configspace_images = len(
            [name for name in os.listdir(self.configspace_dir) if
             os.path.isfile(os.path.join(self.configspace_dir, name))])

        assert self.num_configspace_images == self.num_workspace_images, f"Expected number of cobs images to " \
                                                                         f"equal number of workspace images. Number " \
                                                                         f"of workspace images: " \
                                                                         f"{self.num_workspace_images} Number of " \
                                                                         f"cobs images:" \
                                                                         f"{self.num_configspace_images}"
        self.workspace_transform = workspace_transform
        self.configspace_transform = configspace_transform

    def __len__(self):
        return self.num_configspace_images

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        workspace_image_name = os.path.join(self.workspace_dir, f"{index}.png")
        configspace_image_name = os.path.join(self.configspace_dir, f"{index}.png")
        try:
            workspace = Image.open(workspace_image_name)
            configspace = Image.open(configspace_image_name)
        except FileNotFoundError:
            workspace_image_name = os.path.join(self.workspace_dir, f"{1}.png")
            configspace_image_name = os.path.join(self.configspace_dir, f"{1}.png")
            workspace = Image.open(workspace_image_name)
            configspace = Image.open(configspace_image_name)

        if self.workspace_transform:
            workspace = self.workspace_transform(workspace)
        if self.configspace_transform:
            configspace = self.configspace_transform(configspace)

        sample = {'workspace': workspace, 'cobs': configspace, 'id': index}
        return sample


if __name__ == '__main__':
    print(os.getcwd())
    configspace_dataset = ConfigSpaceDataset("./data")
    for i in range(len(configspace_dataset))[:1]:
        sample = configspace_dataset[i]
        print(i, sample['workspace'].shape, sample['cobs'].shape)
