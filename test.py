from torch import nn
import torch
import numpy as np

data = torch.rand(16, 3, 150, 25, 2)
print(data.shape)
print(data[0, 0, 0, :, 0])
half = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23]
del_half = [8, 9, 10, 11, 16, 17, 18, 19, 23, 24]

# data = data[:][:][:][1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23][:]
# print(data.shape)
# print(data[0, 0, 0, :, 0])

temp = np.random.random((16, 3, 150, 25, 2))
print(temp.shape)
print(temp[0, 0, 0, :, 0])
temp = np.delete(temp, del_half, axis=3)
print(temp.shape)
print(temp[0, 0, 0, :, 0])
