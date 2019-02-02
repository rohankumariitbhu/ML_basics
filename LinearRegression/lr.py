import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=np.nan)

data=pd.read_csv('kc_house_data.csv')

x_data=np.array(data.iloc[:,3:])
x_data=(x_data-np.mean(x_data))/np.std(x_data)
y_data=np.array(data.iloc[:,2:3])
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2)

lr=LinearRegression().fit(train_x,train_y)
# print(lr.predict(test_x))
# print(lr.coef_)


j=(lr.predict(train_x)-train_y)**2/34580
print(j)