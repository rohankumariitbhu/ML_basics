import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.autograd import Variable
np.set_printoptions(threshold=np.nan)
from sklearn.model_selection import train_test_split
data=pd.read_csv('kc_house_data.csv')
x_data=np.array(data.iloc[:,3:])
y_data=np.array(data.iloc[:,2:3])

train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2)
train_x=(train_x-np.mean(train_x))/np.std(train_x)
train_y=(train_y-np.mean(train_y))/np.std(train_y)

# test_x=(test_x-np.mean(test_x))/np.std(test_x)
# test_y=(test_y-np.mean(test_y))/np.std(test_y)

class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(18,1)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred


our_model=LinearRegressionModel()
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(our_model.parameters(),lr=0.000001)

for epoch in range(50000):
    optimizer.zero_grad()
    y_pred=our_model(Variable(torch.Tensor(train_x)))
    loss=criterion(y_pred,Variable(torch.Tensor(train_y)))

    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.data))

v = np.zeros((17290))
for i in range(17290):
    v[i] = i
plt.scatter(v, train_y, color='g')
plt.scatter(v,our_model(Variable(torch.Tensor(train_x))).data.numpy(), color='r')
plt.show()