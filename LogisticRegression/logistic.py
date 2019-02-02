import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import  preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import  RFE
from sklearn.metrics import accuracy_score, log_loss, auc, roc_curve
from sklearn.model_selection import train_test_split

train_data=pd.read_csv('train(slr).csv')
test_data=pd.read_csv('test(slr).csv')
# X=train_data.iloc[:,:-1]
# Y=train_data.iloc[:,-1]
# X_test=test_data.iloc[:,:-1]
# Y_test=test_data.iloc[:,-1]


print('The number of samples into the train data is {}.'.format(train_data.shape[0]))


# check missing values in train data
train_data.isnull().sum()
# age density histogram for visualization
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_data["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
#Missing value of age will replace with median (i.e 28)
# High percentage of missing value hence not the good feature to include in the model
print('Percent of missing "Cabin" records is %.2f%%' %((train_data['Cabin'].isnull().sum()/train_data.shape[0])*100))

print('The most common boarding port of embarkation is %s.' %train_data['Embarked'].value_counts().idxmax())
# As most of common bording port of embarkation is S hence missing will also be replaced by S

train_data["Age"].fillna(train_data["Age"].median(skipna=True),inplace=True)
train_data["Embarked"].fillna(train_data["Embarked"].value_counts().idxmax(),inplace=True)
train_data.drop("Cabin",axis=1,inplace=True)

# SibSp and Parch both columns account for person is travelling individual or with family so we combine the effects of these two variables into single categorical predictor "TravelAlone"
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)

#create categorical variables and drop some variables
training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])
training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)

final_train = training

# apply same data preprocessing for test data
test_data["Age"].fillna(train_data["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(train_data["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()

# Visualize the age (surviving and decrease in population)
plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()

# By visulaizing the age for survival passanger it is found that most them were children hence include a categorical feature "Minor" for age<16

final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)


cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]
X = final_train[cols]
Y=final_train["Survived"]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

logistic=LogisticRegression()
rfe=RFE(logistic,8)
rfe.fit(x_train,y_train)

# summarize the selection of the attributes
print('Selected features: %s' % list(X.columns[rfe.support_]))

print("=========================")
y_pred=rfe.predict(x_test)
print("========================")
y_pred_proba = rfe.predict_proba(x_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(rfe.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(rfe.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(rfe.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

