import torch
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg19_bn

from utils import get_state_dict_from_git_tag

class TissueTileNet(torch.nn.Module):
    def __init__(self, model, num_classes, activation=None):
        super(TissueTileNet, self).__init__()
        if type(model) in [torchvision.models.resnet.ResNet]:
            model.fc = torch.nn.Linear(512, num_classes)
        elif type(model) == torchvision.models.squeezenet.SqueezeNet:
            list(model.children())[1][1] = torch.nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        else:
            raise NotImplementedError
        self.model = model
        self.activation = activation

    def forward(self, x):
        y = self.model(x)

        if self.activation:
            y = self.activation(y)

        return y

def tissue_tile_net_transform ():
    """ Transformer which generates a torch tensor compatible with the model """
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def tissue_tile_net_model (activation, num_classes, weight_tag=None):
    model = TissueTileNet(resnet18(), num_classes, activation=activation)
    if weight_tag:
        state_dict = get_state_dict_from_git_tag(weight_tag)
        model.load_state_dict(state_dict)
    return model