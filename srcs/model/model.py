import torch
import torchvision


def get_convnexttiny_model(num_classes: int = 29):
    model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
