import torch

print (torch.hub.list("msk-mind/luna-ml:main"))

model = torch.hub.load("msk-mind/luna-ml:main",  "logistic_regression", weight_tag="main:logistic_regression_ones.pth")
print (model.lin1.weight)

model = torch.hub.load("msk-mind/luna-ml:main",  "logistic_regression", weight_tag="main:logistic_regression_random.pth")
print (model.lin1.weight)