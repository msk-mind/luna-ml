import torch
from torch import nn

from utils import get_state_dict_from_git_tag

class LogisticRegression(nn.Module):
    def __init__(self) -> None:
        super(LogisticRegression, self).__init__()
        self.lin1 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        return self.sigmoid(self.lin1(input))

def logistic_regression_model(weight_tag=None):
    model = LogisticRegression()
    if weight_tag:
        state_dict = get_state_dict_from_git_tag(weight_tag)
        model.load_state_dict(state_dict)
    return model