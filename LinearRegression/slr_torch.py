import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(threshold=np.nan)
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
x_data=np.array(train_data.iloc[:,0:1])
y_data=np.array(train_data.iloc[:,1:])
x_data=(x_data-np.mean(x_data))/np.std(x_data)
y_data=(y_data-np.mean(y_data))/np.std(y_data)
xtest_data=np.array(test_data.iloc[:,0:1])
ytest_data=np.array(test_data.iloc[:,1:])


class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    # our model
our_model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr=0.001)
for epoch in range(20):
    optimizer.zero_grad()
    pred_y=our_model(Variable(torch.Tensor(x_data)))
    loss=criterion(pred_y,Variable(torch.Tensor(y_data)))

    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.data))

predicted =our_model.forward(Variable(torch.Tensor(x_data))).data.numpy()
#
plt.scatter(x_data,y_data,color='g')
plt.plot(x_data,predicted,color='r')
# plt.plot(x_data,our_model(Variable(torch.Tensor(x_data))).numpy(),co lor='r')
plt.show()
