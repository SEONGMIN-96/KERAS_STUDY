import numpy as np
from tensorflow.python.ops.control_flow_util import OpInContext

# 1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,4,7,5,7,8,9,10])

# 2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 훈련, 컴파일

from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# [lr = 0.01, 0.001, 0.0001]

# optimizer = Adam(lr=0.0001)
# default_lr = 0.001
# loss :  0.6516843438148499 결과물 :  [[11.388539]]
# loss :  0.7281934022903442 결과물 :  [[10.17423]]
# loss :  0.626919686794281 결과물 :  [[10.450676]]

# optimizer = Adagrad(lr=0.0001)
# default_lr = 0.001
# loss :  0.8990629315376282 결과물 :  [[12.019829]]
# loss :  0.5432530641555786 결과물 :  [[10.932756]]
# loss :  0.5591627359390259 결과물 :  [[11.101605]]

# optimizer = Adamax(lr=0.0001)
# default_lr = 0.001
# loss :  0.5743241906166077 결과물 :  [[10.622524]]
# loss :  0.5449070930480957 결과물 :  [[10.836326]]
# loss :  0.5466552972793579 결과물 :  [[10.978102]]

# optimizer = Adadelta(lr=0.0001)
# default_lr = 0.001
# loss :  0.5556269884109497 결과물 :  [[11.02103]]
# loss :  10.173304557800293 결과물 :  [[5.6283455]]
# loss :  43.43895721435547 결과물 :  [[-0.41141734]]

# optimizer = RMSprop(lr=0.0001)
# default_lr = 0.001
# loss :  1.2364429235458374 결과물 :  [[9.195055]]
# loss :  0.570138156414032 결과물 :  [[10.740248]]
# loss :  0.5559145212173462 결과물 :  [[11.195397]]

optimizer = SGD(lr=0.001, nesterov=False, momentum=0.0, decay=0)
# default_lr = 0.01

# optimizer = Nadam(lr=0.0001)
# loss :  331.01385498046875 결과물 :  [[-5.203874]]
# loss :  0.889053463935852 결과물 :  [[9.881237]]
# loss :  0.5424657464027405 결과물 :  [[11.022378]]

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측

loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)

'''
loss :  0.5771076083183289 결과물 :  [[11.264291]]
loss :  0.557800829410553 결과물 :  [[10.912322]]
loss :  0.5493611097335815 결과물 :  [[11.103882]]
loss :  0.5584396123886108 결과물 :  [[10.732864]]
'''