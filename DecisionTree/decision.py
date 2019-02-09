import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import graphviz

df=pd.read_csv('mushrooms.csv')
print(df[df.isnull().any(axis=1)])
y=df['class']
x=df.drop(['class'],axis=1)
y=pd.get_dummies(y)
x=pd.get_dummies(x)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)
param={'criterion':('gini','entropy'),
       'min_samples_split':[2,3,4,5],
       'max_depth':[9,10,11,12],
       'class_weight':('balanced',None),
       'presort':(False,True),
       }
tr=DecisionTreeClassifier()
gsearch=GridSearchCV(tr,param)
gsearch.fit(X_train,Y_train)
model=gsearch.best_estimator_
scores=cross_val_score(model,X_train,Y_train,cv=5,scoring='f1_macro')
print(scores)
score=model.score(X_test,Y_test)
print(score)

dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=X_test.columns,
                               class_names=Y_test.columns,
                               filled=True, rounded=True,
                               special_characters=True)

graph=graphviz.Source(dot_data)
print(graph)

