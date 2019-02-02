import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=np.nan)
data=pd.read_csv('kc_house_data(lr).csv')
x_data=np.array(data.iloc[:,3:])
y_data=np.array(data.iloc[:,2:3])

train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2)
train_x=(train_x-np.mean(train_x))/np.std(train_x)
test_x=(test_x-np.mean(test_x))/np.std(test_x)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(1,input_dim=18))
model.compile(optimizer='rmsprop',loss='mean_squared_error')

model.fit(train_x,train_y,epochs=1000,batch_size=50)

v=np.zeros((17290))
for i in range(17290):
    v[i]=i
plt.scatter(v,train_y,color='g')
plt.scatter(v,model.predict(train_x),color='r')
plt.show()