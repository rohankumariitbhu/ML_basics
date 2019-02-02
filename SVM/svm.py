import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn import metrics
df = pd.read_csv('voice.csv')

#check relation between features
df.corr()
df.isnull().sum()
df.shape
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
gender_encode=LabelEncoder()
y=gender_encode.fit(y)

# Standardization (i.e. mean=0 and standard deviation =1 ) To make individual feature look like standard noramally distributed data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("Accuracy score for SVM with Default hyperparameter :")
print(metrics.accuracy_score(y_test,y_pred))


#Default Linear Kernel

svc=SVC(kernel='linear')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print("Accuracy score for linear Kernel:")
print(metrics.accuracy_score(y_test,y_pred))

#Default RBF kernel

svc=SVC(kernel='rbf')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print("Accuracy score for RBF Kernel:")
print(metrics.accuracy_score(y_test,y_pred))

#Default Polynomial kernel
svc=SVC(kernel='poly')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print('Accuracy Score for polynomial kernel :')
print(metrics.accuracy_score(y_test,y_pred))

# Performing K-fold cross validation on different kernels

#CV on linear Kernel
svc=SVC(kernel='linear')
score=cross_val_score(svc,x_train,y_train,cv=10,scoring='accuracy')
print(score)
print(score.mean())

#CV on rbf kernel
svc=SVC(kernel='rbf')
score=cross_val_score(svc,x_train,y_train,cv=10,scoring='accuracy')
print(score)
print(score.mean())

#CV on Polynomial kernel
svc=SVC(kernel='poly')
score=cross_val_score(svc,x_train,y_train,cv=10,scoring='accuracy')
print(score)
print(score.mean())



