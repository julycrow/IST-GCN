import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

data_dir = r'D:\PycharmProject\st-gcn\work_dir\recognition\kinetics_skeleton\st_gcn_distance_agri5_32_60_75.26\loss-acc.csv'
f = open(data_dir, encoding='UTF-8')
data = pd.read_csv(f)  # 将csv文件读入并转化为dataframe形式
fontsize = int(20)
train_epoch = data['train_epoch'].values
train_loss = data['train_loss'].values
val_epoch = data['val_epoch'].values
val_epoch = val_epoch[~np.isnan(val_epoch)]
val_loss = data['val_loss'].values
val_loss = val_loss[~np.isnan(val_loss)]
top1 = data['acc'].values
top1 = top1[~np.isnan(top1)]
# plt.plot(train_epoch, train_loss, 'g-', label="train_loss")
# plt.plot(val_epoch, val_loss, 'r-', label="val_loss")
# plt.title('train_loss and val_loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='best')
# plt.show()
# plt.plot(val_epoch, top1, 'b-', label="acc")
# # plt.ylim((0, None))  # 纵坐标从0开始
# plt.ylim((0, 100))  # 纵坐标范围:0-100
# plt.title('acc')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.legend(loc='best')
# plt.show()
plt.figure(1)
plt.plot(train_epoch, train_loss, 'g-', label="train_loss")
plt.plot(val_epoch, val_loss, 'r-', label="val_loss")
plt.title('train_loss and val_loss', fontsize=fontsize)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='best')
plt.savefig(data_dir + r'/../loss1.jpg')
# plt.legend(loc='upper right', fontsize=15)
plt.show()

plt.figure(2)
plt.plot(val_epoch, top1, 'b-', label="acc")
acc_max = np.argmax(top1)
# acc_max = top1.index(max(top1))
show_max = '[' + str(val_epoch[acc_max]) + ' ' + str(top1[acc_max]) + ']'
plt.plot(val_epoch[acc_max], top1[acc_max], 'go')
plt.annotate(show_max, xy=(val_epoch[acc_max], top1[acc_max]), xytext=(val_epoch[acc_max], top1[acc_max]), fontsize=15)
# plt.plot(val_epoch[acc_max], top1[acc_max], 'gs')
# plt.ylim((0, None))  # 纵坐标从0开始
plt.ylim((0, 100))  # 纵坐标范围:0-100
plt.title('acc', fontsize=fontsize)
plt.xlabel('epoch', fontsize=15)
plt.ylabel('acc', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='best')
# plt.legend(loc='upper right', fontsize=15)
plt.savefig(data_dir + r'/../acc1.jpg')
plt.show()
f.close()
path = r"D:\PycharmProject\st-gcn\work_dir\recognition\kinetics_skeleton\st_gcn_distance_agri5_32_60_75.26"
new_path = path + '_' + str(top1[acc_max])
# os.rename(path, new_path)
