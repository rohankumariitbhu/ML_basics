import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv('kc_house_data(lr).csv')
x_data=np.array(data.iloc[:,3:])
y_data=np.array(data.iloc[:,2:3])

train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2)
train_x=(train_x-np.mean(train_x))/np.std(train_x)
test_x=(test_x-np.mean(test_x))/np.std(test_x)

w=np.zeros((18,1),dtype=np.float64)
b=0
lr=0.001
y=lambda x:np.dot(x,w)+b

# plt.subplot(9,1,1)
# plt.scatter(train_x[:,0:1],train_y,)
# # plt.show()
# plt.subplot(9,1,2)
# plt.scatter(train_x[:,1:2],train_y)
# # plt.show()
# plt.subplot(9,1,3)
# plt.scatter(train_x[:,2:3],train_y)
# plt.subplot(9,1,4)
# plt.scatter(train_x[:,3:4],train_y)
# plt.subplot(9,1,5)
# plt.scatter(train_x[:,4:5],train_y)
# plt.subplot(9,1,6)
# plt.scatter(train_x[:,5:6],train_y)
# plt.subplot(9,1,7)
# plt.scatter(train_x[:,6:7],train_y)
# plt.subplot(9,1,8)
# plt.scatter(train_x[:,7:8],train_y)
# plt.show()
# plt.subplot(8,1,1)
# plt.scatter(train_x[:,8:9],train_y)
# plt.subplot(8,1,2)
# plt.scatter(train_x[:,9:10],train_y)
# plt.subplot(8,1,3)
# plt.scatter(train_x[:,10:11],train_y)
# plt.subplot(8,1,4)
# plt.scatter(train_x[:,11:12],train_y)
# plt.subplot(8,1,5)
# plt.scatter(train_x[:,12:13],train_y)
# plt.subplot(8,1,6)
# plt.scatter(train_x[:,13:14],train_y)
#
#
# plt.show()

for _ in range(100000):
    u=(y(train_x) - train_y)
    for i in range(18):
        w[i,0]=w[i,0]-lr*(np.sum(u*train_x[:,i:i+1])/17290)
    b=b-lr*np.sum((u/17290))

v=np.zeros((17290))
for i in range(17290):
    v[i]=i
plt.scatter(v,train_y,color='g')
plt.scatter(v,y(train_x),color='r')

plt.show()