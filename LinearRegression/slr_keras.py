import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
np.set_printoptions(threshold=np.nan)
train_data=pd.read_csv('train(slr).csv')
test_data=pd.read_csv('test(slr).csv')
x_data=np.array(train_data.iloc[:,0:1])
y_data=np.array(train_data.iloc[:,1:])
xtest_data=np.array(test_data.iloc[:,0:1])
ytest_data=np.array(test_data.iloc[:,1:])


model=Sequential()
model.add(Dense(1,input_dim=1))
model.compile(loss='mean_squared_error',optimizer='rmsprop')


model.fit(x_data,y_data,epochs=1000,batch_size=50)

plt.scatter(x_data,y_data,color='g')
plt.plot(x_data,model.predict(x_data),color='r')
plt.show()

