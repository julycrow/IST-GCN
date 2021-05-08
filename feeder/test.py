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
x = np.random.random((2, 18, 3))
#x = x.permute(0, 1, 2, 4, 3)
print(x[0])
print(x.shape)
#x.expand(64, 3, 150, 2, 19)
c = np.sum(x, axis=1)
c /= 18
print(c[0])

c.expand_dims(c, 1)
print(c.shape)

x = np.concatenate((x, c), axis=1)
print(x.shape)
#temp.expand(64, 3, 150, 19, 2)
print(x[0])