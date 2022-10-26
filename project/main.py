import time
import argparse
from config_spaces.project.model import SegNet
from config_spaces.project.dataset import ConfigSpaceDataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose
from torch import nn, optim

data_path = "/home/chris/Documents/columbia/fall_22/config_space/config_spaces/project/scripts/data"
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

criterion = nn.L1Loss(reduction='sum')

parser = argparse.ArgumentParser(description='workspace-configspace')
parser.add_argument('--data', type=str, default='./data/wikitext-103', help='location of corpus')
parser.add_argument('--epochs', type=int, default=50, help='num epochs')
parser.add_argument('--seed', type=int, default=26, help='seed')
parser.add_argument('--cuda', default=True, action='store_true', help='cuda')
parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--seed', type=int, default=26, help='seed')


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis("off")


def train_epoch(model: nn.Module, opt, train_data: DataLoader, epoch, hyperparams):
    model.train()
    total_train_loss = 0
    i = 0
    start_time = time.time()
    for batch in train_data:
        i += 1
        workspaces, config_spaces = batch['workspace'], batch['configspace']

        opt.zero_grad()
        predicted_config_spaces = model(workspaces)
        loss = criterion(predicted_config_spaces, config_spaces)
        loss.backward()
        opt.step()

        total_train_loss += loss.item()

        if i % hyperparams.log_interval == 0:
            curr_loss = total_train_loss / hyperparams.log_interval
            elapsed = time.time()() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f}'.format(epoch, batch,
                                                                                                 len(train_data),
                                                                                                 elapsed * 1000 / hyperparams.log_interval,
                                                                                                 curr_loss))
            total_train_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, val_data, hyperparams):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_data:
            workspaces, config_spaces = batch['workspace'], batch['configspace']
            predicted_config_spaces = model(workspaces)
            total_loss += criterion(predicted_config_spaces, config_spaces)
    return total_loss / len(val_data)


def generate_workspace_configspace_pair(val_data):
    pass


def train(model: nn.Module, opt: torch.optim, train_data: DataLoader, val_data: DataLoader, hyperparams) -> object:
    best_val_loss = float("inf")
    epoch = 1
    while epoch <= hyperparams.num_epochs + 1:
        train_epoch(train_data, opt, epoch, hyperparams)
        val_loss = evaluate(val_data, hyperparams)
        print("=" * 89)
        print('End of epoch {} | val loss {:5.2f}'.format(epoch, val_loss))
        generate_workspace_configspace_pair(val_data)
        print("= * 89")

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            PATH = "SEG-NET-WEIGHTS.pth"
            torch.save(model.state_dict(), PATH)
        else:
            epoch += 1


if __name__ == '__main__':
    get_rid_alpha = transforms.Lambda(lambda x: x[:3])
    transforms = Compose([ToTensor(), get_rid_alpha, Resize(512), Grayscale()])
    dataset = ConfigSpaceDataset(data_path, transform=transforms)

    train_size = int(0.6 * len(dataset))
    val_size = int(.2 * len(dataset))
    test_size = len(dataset) - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=12,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=12,shuffle=True)

    hyperparams = parser.parse_args()
    np.random.seed(hyperparams.seed)
    torch.manual_seed(hyperparams.seed)

    device = torch.device("cuda" if hyperparams.cuda else "cpu")

    if hyperparams.cuda and not torch.cuda.is_available():
        raise Exception("CUDA is not available for use try running without --cuda")

    print(f"USING {device}")
    print(f"Training on {hyperparams.data}")
    in_channels, out_channels = 1, 1
    model = SegNet(in_channels, out_channels)
    total_params = sum(x.data.nelement() for x in model.parameters())
    print("Total number of params: {}".format(total_params))

    model.to(device)

    OPT = optim.Adam(model.parameters(),lr=hyperparams.lr)

    try:
        print('-' * 100)
        print("Starting training...")
        train(model,optim,train_dataloader,val_dataloader,hyperparams)
    except KeyboardInterrupt:
        print('=' * 100)
        print("Exiting from training...")

    test_loss = evaluate(model,test_dataloader,hyperparams)
    print("=" * 100)
    print("| test loss {:5.2f}".format(test_loss))
    print("="* 100)