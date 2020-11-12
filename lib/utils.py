import numpy as np
import torch
from torch.nn import Module
# from poutyne.framework.callbacks import Callback


def softmax(x, temp=10):
    """Computes softmax values for each sets of scores in x."""
    #e_x = np.exp(x - np.max(x))
    e_x = np.exp(temp*x)
    return e_x / e_x.sum()

def sigmoid(x):
    """Computes sigmoid values for each sets of scores in x."""
    return 1./(1. + np.exp(-x))

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# class SaveModel(Callback):
#     def __init__(self, model, filename, monitor='val_acc'):
#         self.base_model = model
#         self.filename = filename
#         self.best = 0.
#         self.monitor = monitor
#
#     def on_epoch_end(self, epoch_number, logs):
#         acc = logs[self.monitor]
#         if acc > self.best:
#             self.best = acc
#             weights = {"network": self.base_model.state_dict()}
#             torch.save(weights, self.filename.format(epoch=epoch_number, val_acc=acc))
