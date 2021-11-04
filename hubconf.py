# Optional list of dependencies required by the package
dependencies = ["torch", "torchvision"]

# classification
from models.logistic_regression import logistic_regression
from models.tissue_tile_net import tissue_tile_net, tissue_tile_net_transform