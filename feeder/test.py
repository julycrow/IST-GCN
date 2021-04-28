import pickle
import numpy as np
import torch

label_path = 'D:/database/after_kinetics_gendata/val_data.npy'
# with open(label_path, 'rb') as f:
#     sample_name, label = pickle.load(f)
# print(sample_name)
#arr = np.load(label_path)
#print(arr.shape)  # (222, 3, 300, 18, 2)(nctvm)
i = 0
j = 0
x = torch.randn(64, 3, 150, 18, 2)
#x = x.permute(0, 1, 2, 4, 3)
print(x[2][0][5])
print(x.shape)
#x.expand(64, 3, 150, 2, 19)
c = torch.sum(x, dim = 3)
print(c[2][0][5])
c /= 18
#c[1] /= 18
c.unsqueeze_(3)
x = torch.cat((x, c), 3)
print(x.shape)
print(c[2][0][5])
#temp.expand(64, 3, 150, 19, 2)
print(x[2][0][5])