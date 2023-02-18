import time
import argparse
from model import SegNet
from dataset import ConfigSpaceDataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose, Normalize
from torchmetrics import Recall, Precision, Accuracy, F1Score, ConfusionMatrix
from Transforms import ThresholdTransform
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib
from tqdm import tqdm
import multiprocessing
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

data_path = "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/data/3_shape"
results_directory =  "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/final_results/3_shape"

criterion = None
criterion_2 = None

recall = None
precision = None
acc = None
f1 = None
confmat = None

parser = argparse.ArgumentParser(description='workspace-cobs')
parser.add_argument('--data', type=str, default=data_path, help='location of dataset')
parser.add_argument('--weights-folder', type=str, default=results_directory + "/weights", help='location to save weights')
parser.add_argument('--weights-file', type=str, default='./weights/14-02-23-19_55_SEGNET-l2-1.pth', help='weights file name')
parser.add_argument('--use-last-checkpoint', default=False, action='store_false', help='use last checkpoint')
parser.add_argument('--obstacles', type=int, default=3, help='number of obstacles in dataset images')
parser.add_argument('--output-imgs-path', type=int, default=1, help=results_directory + "/images")
parser.add_argument('--loss-fn', type=str, default='l2',
                    help='Loss function to train on. supported options: l1,l2,l2_l1,bce')
parser.add_argument('--generate-images-interval', type=int, default=1)
parser.add_argument('--num-epochs', type=int, default=200, help='num epochs')
parser.add_argument('--seed', type=int, default=26, help='seed')
parser.add_argument('--cuda', default=True, action='store_true', help='cuda')
parser.add_argument('--batch_size', type=int, default=5, help='training batch size')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--is-train', default=True, action='store_true', help='do training')
parser.add_argument('--num_images-to-generate-from-test', type=int, default=20,
                    help='number of images to generate from test set')

writer = SummaryWriter(log_dir=results_directory+"/runs")

threshold = .0
collision_scale = 1
def save_img(img, file_name):
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis("off")
    plt.savefig(file_name)
    plt.close()


def generate_metric_stats(predicted_config_spaces, config_spaces):
    pred_config_space = (predicted_config_spaces.flatten() > 0 ).float()
    config_spaces = (config_spaces.flatten() != -1).float()
    pred_config_space = pred_config_space.flatten()
    config_spaces = config_spaces.flatten()
    r = recall(pred_config_space, config_spaces)
    p = precision(pred_config_space, config_spaces)
    a = acc(pred_config_space, config_spaces)
    f = f1(pred_config_space, config_spaces)
    return r, p, a, f


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
        predicted_config_spaces = torch.nn.functional.tanh(predicted_config_spaces)

        if criterion_2:
            if epoch % 2 == 1:
                loss = criterion(predicted_config_spaces, config_spaces)
            else:
                loss = criterion_2(predicted_config_spaces, config_spaces)
        else:
            config_spaces.flatten()[(config_spaces.flatten() == 0).nonzero().flatten()] = -1
            loss = my_loss_fn(predicted_config_spaces,config_spaces)
        loss.backward()
        opt.step()
        total_train_loss_log += loss.item()
        total_train_loss += loss.item()
        if i % hyperparams.log_interval == 0:
            curr_loss = total_train_loss_log / hyperparams.log_interval
            elapsed = time.time() - start_time
            if criterion_2:
                loss_fn = 'l2' if epoch % 2 == 1 else 'l1'
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | {} {:5.2f} | Precision {:5.3f} | '
                      'Recall {:5.3f} | F1Score {:5.3f} | Accuracy {:5.3f} '.format(epoch, i,
                                                                                    len(train_data),
                                                                                    elapsed * 1000 / hyperparams.log_interval,
                                                                                    loss_fn,
                                                                                    curr_loss,
                                                                                    0,
                                                                                    0,
                                                                                    0,
                                                                                    0
                                                                                    ))
            else:
                print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f}'.format(epoch, i,
                                                                                                     len(train_data),
                                                                                                     elapsed * 1000 / hyperparams.log_interval,
                                                                                                     curr_loss))
            total_train_loss_log = 0
            start_time = time.time()

    writer.add_scalar('training_loss', total_train_loss / len(train_data), epoch)


def update_img(params):
    pred_config_spaces, i = params
    free_space_indices = (pred_config_spaces[i,:] == 1 ).nonzero()
    x_offset = [0,1,0,-1,1,-1,-1,1]
    y_offset = [1,0,-1,0,1,1,-1,-1]
    for free_space_idx in free_space_indices:
        num_collisions = 0
        for x_off,y_off in zip(x_offset,y_offset):
            new_pos = torch.cat([torch.tensor([0]),free_space_idx]) + torch.tensor([i,0,x_off,y_off])
            if torch.all(torch.tensor([i,0,0,0]) <= new_pos) and torch.all(new_pos <= torch.tensor([i,1,255,255])):
                if pred_config_spaces[tuple(new_pos)] == 0:
                    num_collisions += 1
                if num_collisions >= 5:
                    pred_config_spaces[tuple(torch.cat([torch.tensor([i]),free_space_idx]))] = 0
                    break

def evaluate(model: nn.Module, data, device):
    model.eval()
    total_loss = 0.0
    preds, actual = [], []
    with torch.no_grad():
        a = 0
        for batch in tqdm(data):
            workspaces, config_spaces = batch['workspace'], batch['cobs']
            workspaces = workspaces.to(device)
            config_spaces = config_spaces.to(device)
            predicted_config_spaces = model(workspaces)
            predicted_config_spaces = torch.nn.functional.tanh(predicted_config_spaces)
            config_spaces.flatten()[(config_spaces.flatten() == 0).nonzero().flatten()] = -1
            total_loss += my_loss_fn(predicted_config_spaces, config_spaces)
            predicted_config_spaces, config_spaces = predicted_config_spaces.to("cpu"), config_spaces.to("cpu")
            pred_config_space = (predicted_config_spaces.flatten() > 0 ).float()
            config_spaces = (config_spaces.flatten() != -1).float()
            preds.append(pred_config_space.flatten())
            actual.append(config_spaces.flatten())
        preds = torch.cat(preds,dim=-1).cpu()
        actual = torch.cat(actual,dim=-1).cpu()
    print(confmat(preds,actual))
    r = recall(preds, actual)
    p = precision(preds, actual)
    a = acc(preds, actual)
    f = f1(preds, actual)
    return total_loss / len(data), r, p, a, f


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


def generate_test_images(model, test_dataloader, device, num_images_to_generate_from_test):
    num_images = 0
    for batch in test_dataloader:
        workspaces, config_spaces, image_ids = batch['workspace'][:2], batch['cobs'][:2], batch['id'][:2]
        workspaces = workspaces.to(device)
        predicted_config_spaces = model(workspaces)
        fig = plt.figure(figsize=(20, 50))
        i = 1
        for workspace, config_space, predicted_config_space, image_id in zip(workspaces, config_spaces,
                                                                             predicted_config_spaces, image_ids):
            ax1 = fig.add_subplot(2, 5, i)
            workspace_img = workspace.cpu().detach().numpy()
            ax1.set_frame_on(False)
            ax1.imshow(np.transpose(workspace_img, (1, 2, 0)), cmap='gray')
            ax1.set_title(f"Workspace-{image_id}")
            ax1.axis("off")
            i += 1

            ax2 = fig.add_subplot(2, 5, i)
            configspace_img = config_space.cpu().detach().numpy()
            ax2.imshow(np.transpose(configspace_img, (1, 2, 0)), cmap='gray', extent=[-7, 360, -7, 360])
            ax2.set_frame_on(False)
            ax2.set_title(f"configspace-{image_id}")
            ax2.set_xlabel("Q1")
            ax2.set_ylabel("Q2")
            i += 1

            ax3 = fig.add_subplot(2, 5, i)
            predicted_config_space = (predicted_config_space > threshold).float()
            pred_img = predicted_config_space.cpu().detach().numpy()
            ax3.set_frame_on(False)
            ax3.imshow(np.transpose(pred_img, (1, 2, 0)), cmap='gray', extent=[-7, 360, -7, 360])
            ax3.set_title(f"predicted-{image_id}")
            ax3.set_xlabel("Q1")
            ax3.set_ylabel("Q2")
            i += 1

            ax4 = fig.add_subplot(2, 5, i)
            # 0 would be obstacle 1 would be no obstacle.
            idxs = np.where(configspace_img.flatten() < pred_img.flatten())
            missed_collisions = np.ones((1, 512, 512)).flatten()
            missed_collisions[idxs] = 0
            missed_collisions = missed_collisions.reshape(1,512,512)
            ax4.set_frame_on(False)
            ax4.imshow(np.transpose(missed_collisions, (1, 2, 0)), cmap='gray',extent=[-7, 360, -7, 360])
            ax4.set_title(f"undetected-collisions-{image_id}")
            ax4.set_xlabel("Q1")
            ax4.set_ylabel("Q2")
            i += 1

            ax5 = fig.add_subplot(2, 5, i)
            # 0 would be obstacle 1 would be no obstacle.
            idxs = np.where(configspace_img.flatten() > pred_img.flatten())
            missed_free = np.ones((1, 512, 512)).flatten()
            missed_free[idxs] = 0
            missed_free = missed_free.reshape(1,512,512)
            ax5.set_frame_on(False)
            ax5.imshow(np.transpose(missed_free, (1, 2, 0)), cmap='gray',extent=[-7, 360, -7, 360])
            ax5.set_title(f"undetected-free-{image_id}")
            ax5.set_xlabel("Q1")
            ax5.set_ylabel("Q2")
            i += 1


        plt.tight_layout()
        plt.savefig(f"generated-imgs-{num_images}")
        num_images += len(workspaces)
        if num_images == num_images_to_generate_from_test:
            return

def get_confusion_matrix_indices(pred,actual):
    unq = np.array([x + 2*y for x, y in zip(pred,actual)])
    tp = np.array(np.where(unq == 3)).tolist()[0]
    fp = np.array(np.where(unq == 1)).tolist()[0]
    tn = np.array(np.where(unq == 0)).tolist()[0]
    fn = np.array(np.where(unq == 2)).tolist()[0]

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
            ax1.set_frame_on(False)
            ax1.imshow(np.transpose(workspace_img, (1, 2, 0)), cmap='gray')
            ax1.set_title(f"Workspace-{image_id}")
            ax1.axis("off")
            i += 1

            ax2 = fig.add_subplot(2, 3, i)
            configspace_img = config_space.cpu().detach().numpy()
            ax2.imshow(np.transpose(configspace_img, (1, 2, 0)),extent=[-7,360,-7,360],cmap='gray')
            ax2.set_frame_on(False)
            ax2.set_title(f"configspace-{image_id}")
            ax2.set_xlabel("Q1")
            ax2.set_ylabel("Q2")
            i += 1

            ax3 = fig.add_subplot(2, 3, i)
            predicted_config_space = (predicted_config_space > threshold).float()
            pred_img = predicted_config_space.cpu().detach().numpy()
            ax3.set_frame_on(False)
            ax3.imshow(np.transpose(pred_img, (1, 2, 0)),extent=[-7,360,-7,360],cmap='gray')
            ax3.set_title(f"predicted-{image_id}")
            ax3.set_xlabel("Q1")
            ax3.set_ylabel("Q2")
            i += 1
        plt.box(False)
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
        val_loss, recall, precision, acc, f1 = evaluate(model, val_data, device)
        writer.add_scalar('validation_loss', val_loss, epoch)
        writer.add_scalar('validation_recall', recall, epoch)
        writer.add_scalar('validation_precision', precision, epoch)
        writer.add_scalar('validation_acc', acc, epoch)
        writer.add_scalar('validation_f1', f1, epoch)
        writer.flush()
        print("=" * 89)
        print('End of epoch {} | val loss {:5.2f} | lr {:5.5f}'.format(epoch, val_loss, scheduler.get_last_lr()[0]))
        if epoch % hyperparams.generate_images_interval == 0:
            writer.add_figure(f'workspace and true configspace vs predicted epoch-{epoch}',
                              generate_workspace_configspace_pair_tensorboard(model, epoch, device, val_data,
                                                                              hyperparams),
                              global_step=epoch)
        print("=" * 89)
        epoch += 1

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
        filename = datetime.now().strftime(
            f'{hyperparams.weights_folder}/%d-%m-%y-%H_%M_SEGNET-{hyperparams.loss_fn}-{hyperparams.obstacles}.pth')
        torch.save(model.state_dict(), filename)


def set_critertion(loss_fn):
    global criterion_2
    global criterion
    if loss_fn == 'l1':
        criterion = nn.L1Loss(reduction='sum')
    elif loss_fn == 'l2':
        criterion = nn.MSELoss()
    elif loss_fn == 'l2_l1':
        criterion = nn.MSELoss(reduction='sum')
        criterion_2 = nn.L1Loss(reduction='sum')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

def my_loss_fn(pred,actual):
    pred_collision = pred.flatten()[(actual.flatten() == -1).nonzero().flatten()]
    actual_collision = actual.flatten()[(actual.flatten() == -1).nonzero().flatten()]
    pred_free = pred.flatten()[(actual.flatten() == 1).nonzero().flatten()]
    actual_free = actual.flatten()[(actual.flatten() == 1).nonzero().flatten()]
    a = collision_scale * criterion(pred_collision,actual_collision)
    b = criterion(pred_free,actual_free)
    return a + b

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

    train_size = int(0.7 * len(dataset))
    val_size = int(.15 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    save_img(train_dataset[0]['workspace'], "test-workspace")
    save_img(train_dataset[0]['cobs'], "test-cobs")

    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    if hyperparams.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"USING {device}")
    print(f"Training on {hyperparams.data}")
    print(F"Training on objective {hyperparams.loss_fn}")
    print(f"USING Threshold {threshold}")

    recall = Recall(task="binary", average="macro")
    precision = Precision(task="binary", average="macro")
    acc = Accuracy(task="binary", average="macro")
    f1 = F1Score(task="binary", average="macro")
    confmat = ConfusionMatrix(task='binary', num_classes=2, normalize='true')

    model = SegNet()
    if hyperparams.use_last_checkpoint:
        print(f"Loading from last checkpoint file: {hyperparams.weights_file}")
        model.load_state_dict(torch.load(hyperparams.weights_file))

    total_params = sum(x.data.nelement() for x in model.parameters())
    print("Total number of params: {}".format(total_params))
    model = model.to(device)
    opt = optim.Adadelta(model.parameters(), lr=.01)
    scheduler = MultiStepLR(opt, milestones=[25, 50, 75, 100,125,150,175,200], gamma=.75)

    if hyperparams.is_train:
        print(scheduler.get_last_lr()[0])
        try:
            print('-' * 100)
            print("Starting training...")
            train(model, opt, scheduler, train_dataloader, val_dataloader, device, hyperparams)
        except KeyboardInterrupt:
            print('=' * 100)
            print("Exiting from training...")
        writer.close()

    generate_test_images(model, test_dataloader, device, hyperparams.num_images_to_generate_from_test)
    test_loss, recall, precision, acc, f1 = evaluate(model, test_dataloader, device)
    writer.add_scalar('test_loss', test_loss)
    writer.add_scalar('test_recall', recall)
    writer.add_scalar('test_precision', precision)
    writer.add_scalar('test_acc', acc)
    writer.add_scalar('test_f1', f1)
    print("=" * 100)
    print("| test loss {:5.5f} | recall {:5.5f} | precision {:5.5f} | acc {:5.5f} | f1 {:5.5f}|".format(test_loss,recall,precision,acc,f1))
    print("=" * 100)
    generate_workspace_configspace_pair(model, hyperparams.num_epochs + 1, device, test_dataloader, hyperparams)
