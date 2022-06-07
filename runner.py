import argparse
from torch.cuda import is_available
from utils import Permutator
from ewc import EWC
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch
from model import Net
from torch.optim import Adam
import numpy as np
import torch.nn as nn
import wandb
from datetime import datetime
from tqdm import tqdm

# argparse
parser = argparse.ArgumentParser(
    description="This is an implementation of Elastic Weight Consolidation"
)
parser.add_argument("--num_tasks", type=int, default=3, help="number of tasks")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--use-ewc", type=bool, default=True, help="Use EWC")
parser.add_argument("--device", type=str, default=None, help="Device to use")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--lmda", type=float, default=1000, help="Lambda")
parser.add_argument(
    "--wandb-project", type=str, default="mnist", help="Wandb project name"
)
args = parser.parse_args()

if args.device is None:
    DEVICE = "cuda:0" if is_available() else "cpu"
else:
    DEVICE = args.device
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
NUM_TASKS = args.num_tasks
LMDA = args.lmda
USE_EWC = args.use_ewc
LR = args.lr


# dataloaders
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_ds = MNIST("./data", train=True, download=True, transform=trans)
test_ds = MNIST("./data", train=False, download=True, transform=trans)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# instantiate model
net = Net().to(DEVICE)
optimizer = Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

permutator = Permutator(NUM_TASKS)
ewc = EWC(net, train_loader, DEVICE, NUM_TASKS, criterion, permutator)


def calc_accuracies(net, task_id):
    accuracies = []
    for batch in test_loader:
        net.eval()
        X, y = permutator.permute_batch(batch, task_id)
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = net(X)
        accuracies.append((torch.argmax(logits, dim=1) == y).float().mean().item())
    accuracy = np.mean(accuracies)
    print(f"task {task_id} accuracy: {accuracy}")
    return accuracy


def train(net, task_id, ep, lmda=100, use_ewc=True):
    net.train()
    for i, batch in enumerate(tqdm(train_loader, desc=f"epoch {ep+1}/{NUM_EPOCHS}")):
        net.zero_grad()
        X, y = permutator.permute_batch(batch, task_id)
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = net(X)
        loss = criterion(logits, y)

        # add fishers from previous tasks
        if use_ewc:
            for prev_task_id in range(task_id):
                for n, p in net.named_parameters():
                    if p.requires_grad:
                        loss += (
                            lmda
                            / 2
                            * torch.sum(
                                ewc.fisher_matrices[prev_task_id][n]
                                * (p - ewc.best_params[prev_task_id][n]) ** 2
                            )
                        )
        loss.backward()

        optimizer.step()
        wandb.log(
            {
                f"Loss/train_{task_id}": loss,
                f"Acc/train_{task_id}": (torch.argmax(logits, dim=1) == y)
                .float()
                .mean()
                .item(),
            }
        )


tags = [f"lambda_{LMDA}", "ewc"] if USE_EWC else []
tmstp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
name = f"ewc-lambda_{LMDA}-{tmstp}" if USE_EWC else f"baseline-{tmstp}"
wandb.init(project=args.wandb_project, tags=tags, name=name)

for task_id in range(NUM_TASKS):
    print(f"--- Task {task_id} Training ---")
    for ep in range(NUM_EPOCHS):
        train(net, task_id, ep, lmda=LMDA, use_ewc=USE_EWC)
        for id in range(NUM_TASKS):
            acc = calc_accuracies(net, id)
            wandb.log({"epoch": ep + NUM_EPOCHS * task_id, f"Acc/test_{id}": acc})
    ewc.update_fisher(task_id)
    ewc.save_params(task_id)
