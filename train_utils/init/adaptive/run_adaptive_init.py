import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")
from train_utils.losses.qmi_loss import qmi_per_layer

# transmitter_ploting = AdaptInitPloting(
#     n_layers=3,
#     log_dir=exp_dict['setup']['plt_path'],
#     outfile_name='A1_augm_mul_1_05_transmitter.png',
#     title=transmitter_p['plot_title'])

# transmitter_ploting.plot_adapt_init(
#     parameters['transmitter']['train_epochs'],
#     losses,
#     final_loss,
#     scalers,
#     projectors_norm,
#     layers_norm,
# )


class AdaptiveInitilization():
    def __init__(self, model, initializer, epoches, adapt_runner):
        self.model = model
        self.initializer = initializer
        self.epoches = epoches
        self.adapt_runner = adapt_runner

    def __call__(self):
        if self.initializer is not None:
            self.model.apply(self.initializer)

        scaling_factor = self._calc_models_scaling_factors()

        return scaling_factor, self.model

    def _calc_models_scaling_factors(self):

        scaling_factors = []

        # projectors_norm_list = np.empty((self.model.n_layers(), self.epoches))
        # scalers_list = np.empty((self.model.n_layers(), self.epoches))
        # losses_list = np.empty((self.model.n_layers(), self.epoches))
        # final_loss_list = np.empty((self.model.n_layers(), self.epoches))
        # layers_norm_list = np.empty(self.model.n_layers())

        for i in range(0, self.model.n_layers()):
            print("\nTraining for layer {}".format(i), end='\n')

            cur_scale = self._per_layer_calculation(
                i)  # , scalers, loss, projectors_norm, final_loss

            layer_norm = T.norm(self.model.weights[i])

            cur_std = T.std(
                self.model.weights[i]).cpu().detach().item() * cur_scale

            self.model.weights[i].data.normal_(0, np.abs(cur_std))
            scaling_factors.append(cur_scale)

            # projectors_norm_list[i] = projectors_norm
            # scalers_list[i] = scalers
            # losses_list[i] = loss
            # final_loss_list[i] = final_loss
            # layers_norm_list[i] = layer_norm.item()

        return scaling_factors  # , losses_list, scalers_list, projectors_norm_list, layers_norm_list, final_loss_list

    def _per_layer_calculation(self, layer):

        projector = nn.Linear(self.model.projector_sizes[layer][0],
                              self.model.projector_sizes[layer][1])

        scaler = nn.Parameter(T.FloatTensor([1.0]).to(DEVICE),
                              requires_grad=True)

        # projectors_norm = np.empty(self.epoches)
        # scalers = np.empty(self.epoches)
        # losses = np.empty(self.epoches)
        # final_losses = np.empty(self.epoches)
        final_loss_flag = False
        # print(
        #     "QMI Loss Before training:",
        #     qmi_per_layer(self.model,
        #                   self.adapt_runner.loader,
        #                   nn.Linear(self.model.projector_sizes[layer][0],
        #                             self.model.projector_sizes[layer][1]),
        #                   layer=layer))
        for epoch in range(self.epoches):
            actual_loss, scaler, projector_norm, final_loss = self.adapt_runner(
                model=self.model,
                scaler=scaler,
                projector=projector,
                layer=layer)

            # projectors_norm[epoch] = projector_norm
            # scalers[epoch] = scaler.item()
            # losses[epoch] = actual_loss
            if final_loss is not None:
                final_loss_flag = True
                # final_losses[epoch] = final_loss
                print(
                    "Epoch {}/{} | Scaler: {:.4f} - Actual loss: {:.4e} - Final loss: {:.4e}  "
                    .format(epoch, self.epoches, scaler.item(), actual_loss,
                            final_loss))
            else:
                print("Epoch {}/{} | Scaler: {:.4f} - Actual loss: {:.4e} ".
                      format(epoch, self.epoches, scaler.item(), actual_loss))
        # print(
        #     "QMI Loss After training:",
        #     qmi_per_layer(self.model,
        #                   self.adapt_runner.loader,
        #                   nn.Linear(self.model.projector_sizes[layer][0],
        #                             self.model.projector_sizes[layer][1]),
        #                   layer=layer))

        if final_loss_flag is False:
            final_losses = None

        return scaler.item(
        )  # , scalers, losses, projectors_norm, final_losses
