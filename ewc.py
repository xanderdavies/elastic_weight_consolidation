from copy import deepcopy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class EWC:
    def __init__(
        self, net: nn.Module, dataloader: DataLoader, device, num_tasks, criterion, permutator):
        self.permutator = permutator
        self.net = net
        self.dataloader = dataloader
        self.best_params = [None for _ in range(num_tasks)]
        self.fisher_matrices = [
            {
                n: torch.zeros_like(p.data)
                for n, p in self.net.named_parameters()
                if p.requires_grad
            }
            for _ in range(num_tasks)
        ]
        self.device = device
        self.criterion = criterion

    def update_fisher(self, task_id):
        for batch in self.dataloader:
            self.net.zero_grad()
            X, y = self.permutator.permute_batch(batch, task_id)
            X, y = X.to(self.device), y.to(self.device)
            logits = self.net(X)
            loss = self.criterion(logits, y)
            loss.backward()
            for n, p in self.net.named_parameters():
                self.fisher_matrices[task_id][n] += p.grad ** 2

        for n, p in self.net.named_parameters():
            self.fisher_matrices[task_id][n] /= len(self.dataloader)

    def save_params(self, task_id):
        self.best_params[task_id] = {
            n: deepcopy(p.data)
            for n, p in self.net.named_parameters()
            if p.requires_grad
        }

