import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv('IRIS.csv')

x=df.iloc[:,:-1]
y=df.iloc[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
svc=SVC()
svc.fit(x,y)
pred=svc.predict(x_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))