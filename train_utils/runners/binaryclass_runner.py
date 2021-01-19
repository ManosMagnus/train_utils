import torch as T
from train_utils.runners.runner import Runner

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class BinaryclassRunner(Runner):
    def __init__(self, criterion=T.nn.BCELoss, device=DEVICE):
        super(BinaryclassRunner, self).__init__(criterion=criterion(),
                                                device=device)

    def predict(self, output):
        pred = T.gt(output, 0.5)
        return pred.flatten()

    def compute_cost(self, output, labels):
        return self.criterion(output.flatten(), labels.float())
