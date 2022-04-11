import torch
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg19_bn

from luna.pathology.common.ml   import TorchTransformModel
from models.tissue_tile_net import TissueTileNet

from utils import get_state_dict_from_git_tag

class TissueTileNetTransformer(TorchTransformModel):

    def __init__(self, use_weights=False):
        # del kwargs['depth']
        self.model = TissueTileNet(resnet18(), 5, activation=torch.nn.Softmax())
        self.class_labels = {0:'Stroma', 1:'Tumor', 2:'Glass', 3:'Necrosis', 4:'TILs'}


    def get_preprocess(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def transform(self, X):
        out = self.model(X)
        return out