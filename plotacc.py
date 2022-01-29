import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

data_dir = r"D:\OneDrive - mail.ustc.edu.cn\中期\aagcn_3gcn_A_acc.csv"
f = open(data_dir, encoding='UTF-8')
data = pd.read_csv(f)  # 将csv文件读入并转化为dataframe形式
fontsize = int(20)
model1 = 'AAGCN'
model2 = 'AAGCN_Cheby'
model3 = 'AAGCN_3dis'
# model1 = 'ST-GCN'
# model2 = 'ST-GCN_sum'
# model3 = 'ST-GCN_cat'
epoch = data['epoch'].values
AAGCN = data[model1].values
AAGCN_19 = data[model2].values
AAGCN_17 = data[model3].values
last = epoch[-2]
plt.figure(1)
plt.plot(epoch, AAGCN, 'g-', label=model1)
plt.plot(epoch, AAGCN_19, 'r-', label=model2)
plt.plot(epoch, AAGCN_17, 'b-', label=model3)
plt.text(last-1, AAGCN[-1]-6, s=str(AAGCN[-1]), fontsize=15)
plt.text(last-1, AAGCN_19[-1], s=str(AAGCN_19[-1]), fontsize=15)
plt.text(last-1, AAGCN_17[-1]-3, s=str(AAGCN_17[-1]), fontsize=15)

plt.title('NTU-RGB-D-xview', fontsize=fontsize)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('acc', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='best')
plt.savefig(data_dir + r'/../acc.jpg')
# plt.legend(loc='upper right', fontsize=15)
plt.show()

f.close()
