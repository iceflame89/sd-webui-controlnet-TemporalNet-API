# based on https://github.com/isl-org/MiDaS

import cv2
import torch
import torch.nn as nn
import os
from annotator.annotator_path import models_path
from .midas.model_loader import default_models, load_model

base_model_path = os.path.join(models_path, "midas")



def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MiDaSInference(nn.Module):

    def __init__(self, model_type, device):
        super().__init__()
        assert (model_type in default_models), f"Invalid model type: {model_type}! Available model types: {default_models.keys()}"
        model_path = os.path.join(base_model_path, default_models[model_type]) 
        model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=False, height=None, square=False)
        self.model = model
        self.model.train = disabled_train

    def forward(self, x):
        with torch.no_grad():
            prediction = self.model(x)
        return prediction

