import torch
from torch import nn

from common.helpers import *


class LogisticRegression(nn.Module):
    def __init__(self) -> None:
        super(LogisticRegression, self).__init__()
        self.lin1 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        return self.sigmoid(self.lin1(input))

def logistic_regression():
    model = LogisticRegression()
    state_dict = get_state_dict_from_git("main", "logistic_regression_random.pth")
    print (state_dict)
    model.load_state_dict(state_dict)
    return model