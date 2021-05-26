import matplotlib.pyplot as plt
data_dir = "D:\\result.txt"
Train_Loss_list = []
Train_Accuracy_list = []
Valid_Loss_list = []
Valid_Accuracy_list = []

x1 = range(0, 30)

y1 = range(0, 30)
y2 = range(0, 60, 2)

#plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-',color='r')
plt.plot(x1, y1, 'b-',label="Train_Accuracy")
plt.plot(x1, y2, 'r-',label="val_Accuracy")
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.legend(loc='best')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-',label="Train_Loss")
# plt.xlabel('Test loss vs. epoches')
# plt.ylabel('Test loss')
# plt.legend(loc='best')
plt.show()