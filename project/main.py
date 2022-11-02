import time
import argparse
from model import SegNet
from dataset import ConfigSpaceDataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from captum.attr import GuidedBackprop

data_path = "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/scripts/data"
ouput_imgs_path = "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/scripts/data/l2/generated"

criterion = nn.MSELoss(reduction='sum')

parser = argparse.ArgumentParser(description='workspace-cobs')
parser.add_argument('--data', type=str, default=data_path, help='location of dataset')
parser.add_argument('--output-imgs-path', type=str, default=ouput_imgs_path, help='location of dataset')
parser.add_argument('--weights-file', type=str, default='./02-11-22-11_51_SEGNET-l2-1.pth', help='weights file name')
parser.add_argument('--generate-images-interval', type=int, default=10)
parser.add_argument('--num-epochs', type=int, default=22, help='num epochs')
parser.add_argument('--seed', type=int, default=26, help='seed')
parser.add_argument('--cuda', default=True, action='store_true', help='cuda')
parser.add_argument('--batch_size', type=int, default=3, help='training batch size')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--use-last-checkpoint', default=True, action='store_true', help='use last checkpoint')

writer = SummaryWriter()


def relu_hook_function(module, grad_in, grad_out):
    if isinstance(module, torch.nn.ReLU):
        return (torch.clamp(grad_in[0], min=0.),)


def save_img(img, file_name):
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis("off")
    plt.savefig(file_name)
    plt.close()


def train_epoch(model: nn.Module, opt, train_data: DataLoader, epoch, device, hyperparams):
    model.train()
    total_train_loss = 0
    i = 0
    start_time = time.time()
    for batch in train_data:
        i += 1
        workspaces, config_spaces = batch['workspace'], batch['cobs']
        workspaces = workspaces.to(device)
        config_spaces = config_spaces.to(device)
        opt.zero_grad()
        predicted_config_spaces = model(workspaces)
        loss = criterion(predicted_config_spaces, config_spaces)
        loss.backward()
        opt.step()
        total_train_loss += loss.item()

        if i % hyperparams.log_interval == 0:
            curr_loss = total_train_loss / hyperparams.log_interval
            writer.add_scalar('training_loss', curr_loss, epoch * len(train_data) + i)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f}'.format(epoch, i,
                                                                                                 len(train_data),
                                                                                                 elapsed * 1000 / hyperparams.log_interval,
                                                                                                 curr_loss))
            total_train_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, data, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data:
            workspaces, config_spaces = batch['workspace'], batch['cobs']
            workspaces = workspaces.to(device)
            config_spaces = config_spaces.to(device)
            predicted_config_spaces = model(workspaces)
            total_loss += criterion(predicted_config_spaces, config_spaces)
    return total_loss / len(data)


def generate_workspace_configspace_pair(model, epoch, device, validation, hyperparams):
    for batch in validation:
        workspaces, config_spaces, image_ids = batch['workspace'], batch['cobs'], batch['id']
        workspaces = workspaces.to(device)
        predicted_config_spaces = model(workspaces)
        for configspace, image_id in zip(predicted_config_spaces, image_ids):
            if not os.path.exists(f"{hyperparams.output_imgs_path}/{epoch}/"):
                os.makedirs(f"{hyperparams.output_imgs_path}/{epoch}/")
            save_img(configspace, f"{hyperparams.output_imgs_path}/{epoch}/cobs-{image_id}")
        break


def train(model: nn.Module, opt: torch.optim, scheduler: MultiStepLR, train_data: DataLoader, val_data: DataLoader,
          device,
          hyperparams) -> object:
    best_val_loss = float("inf")
    epoch = 1
    while epoch <= hyperparams.num_epochs + 1:
        train_epoch(model, opt, train_data, epoch, device, hyperparams)
        scheduler.step()
        val_loss = evaluate(model, val_data, device)
        writer.add_scalar('validation_loss', val_loss, epoch)
        writer.flush()
        print("=" * 89)
        print('End of epoch {} | val loss {:5.2f} | lr {:5.5f}'.format(epoch, val_loss, scheduler.get_last_lr()[0]))
        if epoch % hyperparams.generate_images_interval == 0:
            generate_workspace_configspace_pair(model, epoch, device, val_data, hyperparams)
        print("= * 89")
        epoch += 1

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            filename = datetime.now().strftime('%d-%m-%y-%H_%M_SEGNET-l2-1.pth')
            torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    get_rid_alpha = transforms.Lambda(lambda x: x[:3])
    transforms = Compose([ToTensor(), get_rid_alpha, Resize((512, 512)), Grayscale()])
    dataset = ConfigSpaceDataset(data_path, transform=transforms)

    train_size = int(0.7 * len(dataset))
    val_size = int(.15 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    hyperparams = parser.parse_args()
    np.random.seed(hyperparams.seed)
    torch.manual_seed(hyperparams.seed)

    train_dataset, val_size, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    val_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True)

    device = torch.device("cuda")

    print(f"USING {device}")
    print(f"Training on {hyperparams.data}")
    in_channels, out_channels = 1, 1
    model = SegNet()
    if hyperparams.use_last_checkpoint:
        print(f"Loading from last checkpoint file: {hyperparams.weights_file}")
        model.load_state_dict(torch.load(hyperparams.weights_file))

    total_params = sum(x.data.nelement() for x in model.parameters())
    print("Total number of params: {}".format(total_params))
    model = model.to(device)
    opt = optim.Adadelta(model.parameters(), lr=.01, weight_decay=1e-4)
    scheduler = MultiStepLR(opt,
                            milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                                        190, 200, 210, 220, 230, 240, 250], gamma=.75)
    print(scheduler.get_last_lr()[0])
    try:
        print('-' * 100)
        print("Starting training...")
        train(model, opt, scheduler, train_dataloader, val_dataloader, device, hyperparams)
    except KeyboardInterrupt:
        print('=' * 100)
        print("Exiting from training...")
    writer.close()

    test_loss = evaluate(model, test_dataloader, device)
    print("=" * 100)
    print("| test loss {:5.2f}".format(test_loss))
    print("=" * 100)
    generate_workspace_configspace_pair(model, hyperparams.num_epochs + 1, device, test_dataloader, hyperparams)
