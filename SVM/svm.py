import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn import metrics


df = pd.read_csv('voice.csv')

#check relation between features
df.corr()
print(df.isnull().sum())
X=df.iloc[1:,:-1]
y=df.iloc[1:,-1]
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)

# Standardization (i.e. mean=0 and standard deviation =1 ) To make individual feature look like standard noramally distributed data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

svc=SVC(gamma='auto')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("Accuracy score for SVM with Default hyperparameter :")
print(metrics.accuracy_score(y_test,y_pred))


#Default Linear Kernel

svc=SVC(kernel='linear',gamma='auto')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print("Accuracy score for linear Kernel:")
print(metrics.accuracy_score(y_test,y_pred))

#Default RBF kernel

svc=SVC(kernel='rbf',gamma='auto')
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print("Accuracy score for RBF Kernel:")
print(metrics.accuracy_score(y_test,y_pred))

#Default Polynomial kernel
svc=SVC(kernel='poly',gamma='auto')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print('Accuracy Score for polynomial kernel :')
print(metrics.accuracy_score(y_test,y_pred))

# Performing K-fold cross validation on different kernels

#CV on linear Kernel
svc=SVC(kernel='linear',gamma='auto')
score=cross_val_score(svc,x_train,y_train,cv=10,scoring='accuracy')
print(score)
print(score.mean())

#CV on rbf kernel
svc=SVC(kernel='rbf',gamma='auto')
score=cross_val_score(svc,x_train,y_train,cv=10,scoring='accuracy')
print(score)
print(score.mean())

#CV on Polynomial kernel
svc=SVC(kernel='poly',gamma='auto')
score=cross_val_score(svc,x_train,y_train,cv=10,scoring='accuracy')
print(score)
print(score.mean())

# toggling with c parameter of SVC (more is the value of C leads to overfitting and less is the value of C leads to underfitting )
C_range=list(range(1,26))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c,gamma='auto')
    scores = cross_val_score(svc, x_train, y_train, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)

C_values=list(range(1,26))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0,27,2))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# More closer Look towards accuracy of model
C_range=list(np.arange(0.1,6,0.1))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='linear', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)

# Visualize
C_values=list(np.arange(0.1,6,0.1))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0.0,6,0.3))
plt.xlabel('Value of C for SVC ')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# toggling with gamma parameter inverse of standar deviation of RBF kernel (gaussian function)
# i.e similarity measure between two points
gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)

gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.xticks(np.arange(0.0001,100,5))
plt.ylabel('Cross-Validated Accuracy')
plt.show()
# poor in 10 and 100 also slightly dip at gamma=1 hence more details for the range in 0.0001 to 0.1
gamma_range=[0.0001,0.001,0.01,0.1]
acc_score=[]
for g in gamma_range:
    svc = SVC(kernel='rbf', gamma=g)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)

gamma_range=[0.0001,0.001,0.01,0.1]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(gamma_range,acc_score)
plt.xlabel('Value of gamma for SVC ')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# Polynomial kernel with different degree
degree=[2,3,4,5,6]
acc_score=[]
for d in degree:
    svc = SVC(kernel='poly', degree=d)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)


degree=[2,3,4,5,6]

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(degree,acc_score,color='r')
plt.xlabel('degrees for SVC ')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# Grid search to find best parameter
svm_model=SVC()
tuned_parameter={
    'C': (np.arange(0.1,1,0.1)), 'kernel':['linear'],
    'C' : (np.arange(0.1,1,0.1)), 'gamma':[0.01,0.02,0.03,0.04,0.05], 'kernel':['rbf'],
    'degree': [2, 3, 4], 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05], 'C': (np.arange(0.1, 1, 0.1)), 'kernel': ['poly']

}

model_svm=GridSearchCV(svm_model,tuned_parameter,scoring='accuracy')
model_svm.fit(x_train,y_train)
print(model_svm.best_score_)
print(model_svm.best_params_)
y_pred= model_svm.predict(x_test)
print(metrics.accuracy_score(y_pred,y_test))