import matplotlib.pyplot as plt
import pandas as pd

data_dir = r"D:\PycharmProject\st-gcn\work_dir\recognition\kinetics_skeleton\ST_GCN_test\loss-acc.csv"
f = open(data_dir, encoding = 'UTF-8')
data=pd.read_csv(f) #将csv文件读入并转化为dataframe形式

train_epoch = data['train_epoch'].values
train_loss = data['train_loss'].values
val_epoch = data['val_epoch'].values
val_loss = data['val_loss'].values
top1 = data['acc'].values

plt.plot(train_epoch, train_loss, 'g-', label="train_loss")
plt.plot(val_epoch, val_loss, 'r-', label="val_loss")
plt.title('train_loss and val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
plt.plot(val_epoch, top1, 'b-', label="acc")
# plt.ylim((0, None))  # 纵坐标从0开始
plt.ylim((0, 100))  # 纵坐标范围:0-100
plt.title('acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(loc='best')
plt.show()

