import torch as T
from train_utils.runners.runner import Runner

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class MulticlassRunner(Runner):
    def __init__(self, criterion=T.nn.NLLLoss, device=DEVICE):
        super(MulticlassRunner, self).__init__(criterion=criterion(),
                                               device=device)

    def predict(self, output):
        return T.argmax(output, dim=1)

    def compute_cost(self, output, labels):
        return self.criterion(output, labels)
