import numpy as np
import torch as T

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


def qmi_loss(data, targets, sigma=3, n_classes=2, eps=1e-8, use_cosine=True):

    if use_cosine:
        data = data / (T.sqrt(T.sum(data**2, dim=1, keepdim=True)) + eps)
        Y = T.mm(data, data.t())
        Y = 0.5 * (Y + 1)
    else:
        Y = squared_pairwise_distances(data)
        sigma = T.mean(Y.detach())
        Y = T.exp(-Y / (2 * sigma))
        # Y = T.exp(-Y / (2 * sigma**2))

    # Get the indicator matrix \Delta
    D = (targets.view(targets.shape[0],
                      1) == targets.view(1, targets.shape[0]))
    D = D.type_as(data)

    if n_classes == 0:
        n_classes = D.size(1)**2 / T.sum(D)

    Q_in = D * Y
    Q_btw = (1.0 / n_classes) * Y

    loss = T.sum(Q_in - Q_btw)

    return loss


def qmi_per_layer(model, loader, projector=None, layer=0):

    projector = projector.to(DEVICE)
    qmi_counter = 0
    qmi_loss_acc = 0

    with T.no_grad():
        for (inputs, targets) in loader:
            # Get the data
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Feed forward the network and update
            outputs = None
            outputs = model(inputs,
                            scaler=T.FloatTensor([1.0]).to(DEVICE),
                            layer=layer)
            outputs = T.flatten(outputs, start_dim=1)
            outputs = projector(outputs)

            qmi_counter += 1
            qmi_loss_acc += qmi_loss(outputs,
                                     targets,
                                     sigma=3,
                                     n_classes=64,
                                     eps=1e-8,
                                     use_cosine=False)

    return (qmi_loss_acc / qmi_counter * 1.0).item()


def squared_pairwise_distances(a, b=None):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = T.sum(a**2, dim=1)
    bb = T.sum(b**2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = T.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = T.clamp_min(dists, min=0)

    return dists
