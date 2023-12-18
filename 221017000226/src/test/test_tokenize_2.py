import os
from io import open
import torch

ids = [1, 2, 3, 4, 5]

list_tensor = torch.tensor(ids)

print(type(list_tensor), list_tensor)

results = list_tensor.type(torch.int64)

print(type(results), results)
