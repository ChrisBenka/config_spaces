import time
import argparse
from model import SegNet
from dataset import ConfigSpaceDataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose, CenterCrop
from Transforms import ThresholdTransform
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

data_path = "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/data/three_10k"
weights_folder = "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/"

criterion = None
criterion_2 = None

parser = argparse.ArgumentParser(description='workspace-cobs')
parser.add_argument('--data', type=str, default=data_path, help='location of dataset')
parser.add_argument('--weights-folder', type=str, default=weights_folder, help='location to save weights')
parser.add_argument('--weights-file', type=str, default='./02-11-22-11_51_SEGNET-l2-1.pth', help='weights file name')
parser.add_argument('--use-last-checkpoint', default=False, action='store_true', help='use last checkpoint')
parser.add_argument('--num-obstacles', type=int, default=1, help='number of obstacles in dataset images')
parser.add_argument('--loss-fn', type=str, default='l2_l1',
                    help='Loss function to train on. supported options: l1,l2,l2_l1,bce')
parser.add_argument('--generate-images-interval', type=int, default=1)
parser.add_argument('--num-epochs', type=int, default=250, help='num epochs')
parser.add_argument('--seed', type=int, default=26, help='seed')
parser.add_argument('--cuda', default=True, action='store_true', help='cuda')
parser.add_argument('--batch_size', type=int, default=5, help='training batch size')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')

writer = SummaryWriter()


def save_img(img, file_name):
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis("off")
    plt.savefig(file_name)
    plt.close()


def train_epoch(model: nn.Module, opt, train_data: DataLoader, epoch, device, hyperparams):
    model.train()
    total_train_loss_log = 0
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
        if criterion_2:
            if epoch % 2 == 1:
                loss = criterion(predicted_config_spaces, config_spaces)
            else:
                loss = criterion_2(predicted_config_spaces, config_spaces)
        else:
            loss = criterion(predicted_config_spaces, config_spaces)
        loss.backward()
        opt.step()
        total_train_loss_log += loss.item()
        total_train_loss += loss.item()

        if i % hyperparams.log_interval == 0:
            curr_loss = total_train_loss_log / hyperparams.log_interval
            elapsed = time.time() - start_time
            if criterion_2:
                loss_fn = 'l2' if epoch % 2 == 1 else 'l1'
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | {} {:5.2f}'.format(epoch, i,
                                                                                                   len(train_data),
                                                                                                   elapsed * 1000 / hyperparams.log_interval,
                                                                                                   loss_fn,
                                                                                                   curr_loss))
            else:
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f}'.format(epoch, i,
                                                                                                     len(train_data),
                                                                                                     elapsed * 1000 / hyperparams.log_interval,
                                                                                                     curr_loss))
            total_train_loss_log = 0
            start_time = time.time()

        writer.add_scalar('training_loss', total_train_loss / len(train_data), epoch)


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


def generate_workspace_configspace_pair_tensorboard(model, epoch, device, validation, hyperparams):
    for batch in validation:
        workspaces, config_spaces, image_ids = batch['workspace'][:2], batch['cobs'][:2], batch['id'][:2]
        workspaces = workspaces.to(device)
        predicted_config_spaces = model(workspaces)
        fig = plt.figure(figsize=(20, 30))
        i = 1
        for workspace, config_space, predicted_config_space, image_id in zip(workspaces, config_spaces,
                                                                             predicted_config_spaces, image_ids):
            ax1 = fig.add_subplot(2, 3, i)
            workspace_img = workspace.cpu().detach().numpy()
            ax1.imshow(np.transpose(workspace_img, (1, 2, 0)), cmap='gray')
            ax1.set_title(f"Workspace-{image_id}")
            ax1.axis("off")
            i += 1

            ax2 = fig.add_subplot(2, 3, i)
            configspace_img = config_space.cpu().detach().numpy()
            ax2.imshow(np.transpose(configspace_img, (1, 2, 0)), cmap='gray')
            ax2.set_title(f"configspace-{image_id}")
            ax2.axis("off")
            i += 1

            ax3 = fig.add_subplot(2, 3, i)
            pred_img = predicted_config_space.cpu().detach().numpy()
            ax3.imshow(np.transpose(pred_img, (1, 2, 0)), cmap='gray')
            ax3.set_title(f"predicted-{image_id}")
            ax3.axis("off")
            i += 1
        plt.tight_layout()
        return fig


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
            writer.add_figure('workspace and true configspace vs predicted',
                              generate_workspace_configspace_pair_tensorboard(model, epoch, device, val_data,
                                                                              hyperparams),
                              global_step=epoch)
        print("=" * 89)
        epoch += 1

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            filename = datetime.now().strftime(
                f'{hyperparams.weights_folder}/%d-%m-%y-%H_%M_SEGNET-{hyperparams.loss_fn}-{hyperparams.num_obstacles}.pth')
            torch.save(model.state_dict(), filename)


def set_critertion(loss_fn):
    global criterion_2
    global criterion
    if loss_fn == 'l1':
        criterion = nn.L1Loss(reduction='sum')
    elif loss_fn == 'l2':
        criterion = nn.MSELoss(reduction='sum')
    elif loss_fn == 'l2_l1':
        criterion = nn.MSELoss(reduction='sum')
        criterion_2 = nn.L1Loss(reduction='sum')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')


def get_transforms(loss_fn):
    get_rid_alpha = transforms.Lambda(lambda x: x[:3])
    w_transform = Compose([ToTensor(), get_rid_alpha, Resize((512, 512)), Grayscale(), ThresholdTransform(thr_255=240)])
    c_transform = Compose([
        transforms.ToTensor(),
        Resize((512, 512)),
        get_rid_alpha,
        transforms.Grayscale(),
        ThresholdTransform(thr_255=240)
    ])
    return w_transform, c_transform


if __name__ == '__main__':

    hyperparams = parser.parse_args()
    np.random.seed(hyperparams.seed)
    torch.manual_seed(hyperparams.seed)

    set_critertion(hyperparams.loss_fn)

    workspace_transforms, configspace_transforms = get_transforms(hyperparams.loss_fn)

    dataset = ConfigSpaceDataset(data_path, workspace_transform=workspace_transforms,
                                 configspace_transform=configspace_transforms)

    train_size = int(0.4 * len(dataset))
    val_size = int(.15 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    train_dataset, val_size, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    save_img(train_dataset[0]['workspace'], "test-workspace")
    save_img(train_dataset[0]['cobs'], "test-cobs")

    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=False)
    val_dataloader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=hyperparams.batch_size, shuffle=True)

    if hyperparams.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"USING {device}")
    print(f"Training on {hyperparams.data}")
    print(F"Training on objective {hyperparams.loss_fn}")

    model = SegNet()
    if hyperparams.use_last_checkpoint:
        print(f"Loading from last checkpoint file: {hyperparams.weights_file}")
        model.load_state_dict(torch.load(hyperparams.weights_file))

    total_params = sum(x.data.nelement() for x in model.parameters())
    print("Total number of params: {}".format(total_params))
    model = model.to(device)
    opt = optim.Adadelta(model.parameters(), lr=.01)
    scheduler = MultiStepLR(opt, milestones=[25, 50, 75, 100, 125, 150, 175, 200, 225], gamma=.75)

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
