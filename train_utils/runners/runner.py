import math
from abc import ABC, abstractmethod

import torch as T

#from tqdm import tqdm

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class Runner(ABC):
    def __init__(self, criterion, device=DEVICE):
        self.device = device
        self.criterion = criterion

    def run_epoch(
        self,
        model: T.nn.Module,
        optimizer: T.optim.Optimizer,
        train_dataloader: T.utils.data.DataLoader,
    ):
        running_loss, running_accuracy, total = 0.0, 0.0, 0.0

        for inputs, target in train_dataloader:

            inputs = inputs.to(self.device)
            target = self.prep_target(target.to(self.device))

            with T.set_grad_enabled(True):
                output = model(inputs)
                pred = self.predict(output)
                loss = self.compute_cost(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.detach()
            total += target.size(0)
            running_accuracy += (pred.detach()
                                 == target.detach()).sum().item() * 1.0

            del output, pred, loss

        return running_loss / total, running_accuracy / total

    def fit(self,
            model: T.nn.Module,
            optimizer: T.optim.Optimizer,
            train_dataloader: T.utils.data.DataLoader,
            num_epochs: int,
            verbose: int = 2):

        model.to(self.device)
        model.train()

        acc_loss, acc_accuracy = 0, 0
        for _ in range(num_epochs):

            running_loss, running_accuracy = self.run_epoch(
                model, optimizer, train_dataloader)

            if (verbose > 1):
                print('\nTraining: Loss: {:.4f} Acc: {:.4f}'.format(
                    running_loss, running_accuracy))

            acc_loss += running_loss
            acc_accuracy += running_accuracy

        return acc_loss / num_epochs, acc_accuracy / num_epochs

    def eval(self, model, eval_dataloader, verbose=2):

        model.to(self.device)
        model.eval()

        running_loss, running_corrects, total = 0.0, 0.0, 0.0

        for inputs, target in eval_dataloader:

            inputs = inputs.to(self.device)
            target = self.prep_target(target.to(self.device))

            with T.no_grad():
                output = model(inputs)
                pred = self.predict(output)
                loss = self.compute_cost(output, target)

            running_loss += loss.detach()
            total += target.size(0)
            running_corrects += (pred.detach()
                                 == target.detach()).sum().item() * 1.0

            del output, pred, loss

        final_loss, final_accuracy = running_loss / total, running_corrects / total

        if (verbose > 1):
            print('Evaluation: Loss: {:.4f} Acc: {:.4f}'.format(
                final_loss, final_accuracy))

        return final_loss, final_accuracy

    @abstractmethod
    def predict(self, output):
        pass

    @abstractmethod
    def compute_cost(self, output, target):
        pass

    def prep_target(self, target):
        return target
