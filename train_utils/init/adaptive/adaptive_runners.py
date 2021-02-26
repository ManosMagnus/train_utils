import torch as T
import torch.nn as nn
import torch.optim as optim
from train_utils.losses.qmi_loss import qmi_loss

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class ContrainLossWrapper(object):
    def __init__(self, model, layer, projector, criterion, cl_weight,
                 norm_weight):
        self.model = model
        self.projector = projector
        self.criterion = criterion
        self.cl_weight = cl_weight
        self.norm_weight = norm_weight
        self.layer = layer

        self.actual_loss = 0
        self.final_loss = None
        self.counter = 0

    def __call__(
        self,
        outputs,
        targets,
    ):
        loss = actual_loss = self.criterion(outputs, targets)
        result = None

        if self.cl_weight <= 0 and self.norm_weight <= 0:
            self.add_actual_loss(actual_loss)
            result = actual_loss

        elif (self.model.constrain_loss[self.layer]
              is not None) and self.cl_weight > 0:
            loss = actual_loss + self.cl_weight * self.model.constrain_loss[
                self.layer]
            if self.norm_weight > 0:
                loss += self.norm_weight * T.norm(self.projector.weight)
            self.add_contrain_loss(actual_loss, loss)
            result = loss

        elif self.norm_weight > 0:
            loss = actual_loss + self.norm_weight * T.norm(
                self.projector.weight)
            self.add_contrain_loss(actual_loss, loss)
            result = loss

        else:
            self.add_actual_loss(actual_loss)
            result = actual_loss

        return result

    def add_actual_loss(self, actual_loss):
        self.actual_loss += actual_loss.item()

    def add_contrain_loss(self, actual_loss, final_loss):
        self.add_actual_loss(actual_loss)
        if self.final_loss is None:
            self.final_loss = 0

        self.final_loss += final_loss.detach().item()

    def get_losses(self, counter):
        if self.final_loss is None:
            return self.actual_loss / counter, None
        else:
            return self.actual_loss / counter, self.final_loss / counter


def constrain_norm_loss(model, projector, layer, actual_loss, cl_weight,
                        norm_weight):

    if cl_weight <= 0 and norm_weight <= 0:
        return actual_loss, False

    if (model.constrain_loss[layer] is not None) and cl_weight > 0:
        loss = actual_loss + cl_weight * model.constrain_loss[layer]
        if norm_weight > 0:
            loss += norm_weight * T.norm(projector.weight)
    elif norm_weight > 0:
        loss = actual_loss + norm_weight * T.norm(projector.weight)
    else:
        return actual_loss, False

    return loss, True


class CLossAdaptiveRunner():
    def __init__(self, optimizer, criterion, loader, norm_weight, cl_weight):
        self.optimizer = optimizer
        self.criterion = criterion
        self.loader = loader
        self.norm_weight = norm_weight
        self.cl_weight = cl_weight

    def __call__(
        self,
        model,
        scaler=None,
        projector=None,
        layer=0,
    ):
        projector = projector.to(DEVICE)
        params = list([scaler])
        params.extend(list(projector.parameters()))
        model_optimizer = self.optimizer(params)

        model.train()
        projector.train()

        constrain_loss_wrapper = ContrainLossWrapper(model, layer, projector,
                                                     self.criterion,
                                                     self.cl_weight,
                                                     self.norm_weight)

        counter = 0
        qmi_counter = 0
        qmi_loss_acc = 0

        for (inputs, targets) in self.loader:
            # Calculate statistics
            qmi_counter += 1
            counter += inputs.size(0)
            loss = self.batch_train(
                model_optimizer,
                model,
                inputs,  #, _qmi_loss
                targets,
                scaler,
                layer,
                projector,
                constrain_loss_wrapper)

            # qmi_loss_acc += _qmi_loss
        projector_norm = T.norm(projector.weight)
        actual_loss, loss = constrain_loss_wrapper.get_losses(counter)

        return actual_loss, scaler, projector_norm, loss,  # (qmi_loss_acc /
        #  qmi_counter *
        #  1.0).item()

    def batch_train(self, model_optimizer, model, inputs, targets, scaler,
                    layer, projector, constrain_loss_wrapper):
        # Reset gradients
        model_optimizer.zero_grad()
        model.zero_grad()

        # Get the data
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # Feed forward the network and update
        outputs = None
        outputs = model(inputs, scaler=scaler, layer=layer)

        outputs = T.flatten(outputs, start_dim=1)
        outputs = projector(outputs)

        _qmi_loss = qmi_loss(outputs,
                             targets,
                             sigma=3,
                             n_classes=64,
                             eps=1e-8,
                             use_cosine=False)

        loss = constrain_loss_wrapper(outputs, targets)

        # Optimization
        loss.backward()
        model_optimizer.step()

        return loss, _qmi_loss
