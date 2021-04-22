from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

seq = 10
x = np.arange(0, 6 * np.pi, 0.01)
y = np.sin(x) + np.cos(x) * x

fig = plt.figure(1)  # 第一幅图
plt.plot(y, 'r')  # 初始图

train = np.array(y).astype(float)  # 创建数组并转换为float类型
scaler = MinMaxScaler()  # 归一化
train = scaler.fit_transform(train.reshape(-1, 1))  # fit_transform对数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等进行处理
# 之后进行transform，从而实现数据的标准化、归一化等等和，reshape将train变成只有一列，行数不记，总shape相同
data = []
for i in range(len(train) - seq - 1):
    data.append(train[i: i + seq + 1])
data = np.array(data).astype('float64')

x = data[:, :-1]
y = data[:, -1]  # 对data这个二维的数据，逗号分隔开的前面的":"是说取全部的行，逗号后面的-1是说取最后一列。 如果换成一维数组会容易理解，比如list[:] 以及list[-1]。
split = int(data.shape[0] * 0.5)

train_x = x[: split]
train_y = y[: split]

test_x = x  # [split:]
test_y = y  # [split:]

train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

model = Sequential()
model.add(LSTM(input_dim=1, output_dim=6, return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))
model.summary()

model.compile(loss='mse', optimizer='rmsprop')

model.fit(train_x, train_y, batch_size=50, nb_epoch=100, validation_split=0.1)
predict_y = model.predict(test_x)
predict_y = np.reshape(predict_y, (predict_y.size,))

predict_y = scaler.inverse_transform([[i] for i in predict_y])
test_y = scaler.inverse_transform(test_y)
fig2 = plt.figure(2)
plt.plot(predict_y, 'g')
plt.plot(test_y, 'r')
plt.show()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
# 公式找到，怎样实现