import torch
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg19_bn

from luna.pathology.analysis.ml import TorchTransformModel
from models.tissue_tile_net import TissueTileNet

from utils import get_state_dict_from_git_tag

class TissueTileNetTransformer(TorchTransformModel):

    def __init__(self, use_weights=False):
        # del kwargs['depth']
        self.model = TissueTileNet(resnet18(), 5, activation=torch.nn.Softmax())
        self.class_labels = {0:'Stroma', 1:'Tumor', 2:'Glass', 3:'Necrosis', 4:'TILs'}
        self.column_labels = {0:'Classification'}
        
        state_dict = get_state_dict_from_git_tag("main:tissue_net_2021-01-19_21.05.24-e17.pth")
        self.model.load_state_dict(state_dict)


    def get_preprocess(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def transform(self, X):
        out = self.model(X)
        labels = [self.class_labels[val] for val in torch.argmax(out)]]
        return labels
