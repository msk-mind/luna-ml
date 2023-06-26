import torch
import torchvision
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg19_bn

from luna.pathology.analysis.ml import TorchTransformModel
from models.tissue_tile_net import TissueTileNet

from utils import get_state_dict_from_git_tag
import numpy as np

class TissueTileNetTransformer(TorchTransformModel):

    def __init__(self, use_weights=False):
        # del kwargs['depth']
        self.model = TissueTileNet(resnet18(pretrained=True), 4, activation=torch.nn.Softmax(dim=1))
        self.class_labels = {
                0: 'Stroma',
                1: 'Tumor',
                2: 'Fat',
                3: 'Necrosis'
                }
        self.column_labels = {0:'Classification'}

        state_dict = get_state_dict_from_git_tag("1525-oncofusion-classifier:tissue_type_classifier_weights.torch")
        self.model.load_state_dict(state_dict)


    def get_preprocess(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def transform(self, X):
        out = self.model(X)
        preds = torch.argmax(out, dim=1)
        labels = np.array([self.class_labels[val.item()] for val in preds])
        labels = np.expand_dims(labels, axis=1)
        return labels
