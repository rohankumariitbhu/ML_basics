import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data=pd.read_csv('train(slr).csv')
test_data=pd.read_csv('test(slr).csv')
x_data=np.array(train_data.iloc[:,0:1])
y_data=np.array(train_data.iloc[:,1:])
xtest_data=np.array(test_data.iloc[:,0:1])
ytest_data=np.array(test_data.iloc[:,1:])

m=np.random.rand()
b=np.random.rand()
lr=0.001

for _ in range(10000):
    y_pred=m*x_data+b
    cost=((y_pred-y_data)**2)/1398
    grad_cost_m=np.dot(x_data,(y_pred-y_data))/699
    grad_cost_b = (y_pred - y_data) / 699
    m-=lr*grad_cost_m
    b-=lr*grad_cost_b

Y_pred=m*x_data+b

plt.scatter(x_data,y_data)
plt.plot(x_data,Y_pred)
plt.show()