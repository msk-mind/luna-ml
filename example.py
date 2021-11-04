import torch

print (torch.hub.list("msk-mind/luna-ml:main"))

model = torch.hub.load("msk-mind/luna-ml:main",  "logistic_regression", weight_tag="main:logistic_regression_ones.pth")
print (model.lin1.weight)

model = torch.hub.load("msk-mind/luna-ml:main",  "logistic_regression", weight_tag="main:logistic_regression_random.pth")
print (model.lin1.weight)

model = torch.hub.load("msk-mind/luna-ml:main", "tissue_tile_net", weight_tag="main:tissue_net_2021-01-19_21.05.24-e17.pth")
print (model)