import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
np.set_printoptions(threshold=np.nan)
train_data=pd.read_csv('train(slr).csv')
test_data=pd.read_csv('test(slr).csv')
x_data=np.array(train_data.iloc[:,0:1])
y_data=np.array(train_data.iloc[:,1:])
xtest_data=np.array(test_data.iloc[:,0:1])
ytest_data=np.array(test_data.iloc[:,1:])
regression=LinearRegression().fit(x_data,y_data)

print(regression.coef_)
plt.subplot(2,1,1)
plt.scatter(x_data,y_data,color='g')
plt.plot(x_data,regression.predict(x_data),color='r')
plt.ylabel("Train data (label of X)")
plt.title("Linear regression plot training and testing phase")
plt.subplot(2,1,2)
plt.scatter(xtest_data,ytest_data,color='g')
plt.plot(xtest_data,regression.predict(xtest_data),color='r')
plt.xlabel("(feature X)")
plt.ylabel("Test data (label of X)")
plt.show()