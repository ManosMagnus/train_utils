from torch.utils.tensorboard import SummaryWriter
from train_utils.runners.runner import Runner


class TensorboardDecor(Runner):
    def __init__(self, runner: Runner, tb_writer: SummaryWriter):
        self.runner = runner
        self.tb_writer = tb_writer

    def fit(self, model, optimizer, train_dataloader, num_epochs, verbose):
        model.to(self.runner.device)
        model.train()

        acc_loss, acc_accuracy = 0, 0
        for epoch in range(num_epochs):

            running_loss, running_accuracy = self.runner.run_epoch(
                model, optimizer, train_dataloader)

            if (verbose > 1):
                print('\nTraining: Loss: {:.4f} Acc: {:.4f}'.format(
                    running_loss, running_accuracy))

            self.tb_writer.add_scalar("loss/train",
                                      running_loss,
                                      global_step=epoch)
            self.tb_writer.add_scalar("accuracy/train",
                                      running_accuracy,
                                      global_step=epoch)

            acc_loss += running_loss
            acc_accuracy += running_accuracy

        return acc_loss / num_epochs, acc_accuracy / num_epochs

    def eval(self, model, eval_dataloader, verbose=2):
        return self.runner.eval(model, eval_dataloader, verbose)

    def predict(self, output):
        self.runner.predict(output)

    def compute_cost(self, output, target):
        self.runner.compute_cost(output, target)

    def prep_target(self, target):
        return self.runner.prep_target(target)
