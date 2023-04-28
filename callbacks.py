import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import wandb
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linear_sum_assignment



class LogPredictionSamplesCallback(Callback):
    def __init__(self, wandb_logger, n_images=8,
                 frequency=1):
        self.n_images = n_images
        self.wandb_logger = wandb_logger
        self.frequency = frequency
        self.__call = 0

        super().__init__()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx < 1 and (self.__call % self.frequency == 0):
            n = self.n_images
            x = batch['image'][:n]

            x, indices_gt = pl_module.shuffle_image(x)

            batch_pred_index = pl_module.forward(x)

            indices = []
            for pred in batch_pred_index:
                pred = pred.detach().cpu().numpy()
                row_index, col_index = linear_sum_assignment(-pred + pred.max())
                indices.append(col_index)

            y = pl_module.unshuffle_image(x, indices)
            columns = ['input', 'prediction', 'score']
            data = []
            for x_i, y_i, v_i in zip(x, y, batch_pred_index):
                ax = plt.subplot()
                fig = plt.gcf()
                cm = ax.imshow(v_i.cpu().numpy(), cmap='RdYlGn')
                fig.colorbar(cm)
                data.append([wandb.Image(x_i), wandb.Image(y_i), wandb.Image(fig)])
                plt.close()
            self.wandb_logger.log_table(data=data, key=f'Validation Batch {batch_idx}', columns=columns)
        self.__call += 1
