import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
# X=train_data.iloc[:,:-1]
# Y=train_data.iloc[:,-1]




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

# By visulaizing the age for survival passanger it is found that most them were children hence include a categorical feature "Minor" for age<16

final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)

cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]
X = final_train[cols]
Y=final_train["Survived"]
#split data into training and dev set for tunning hyperparameters
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
w=np.zeros((X.shape[1],1))
y_train=y_train.values.reshape(x_train.shape[0],1)
b=0
lr=0.000000001
m=X.shape[0]
x_train=np.array(x_train,dtype=np.float64)
y_train=np.array(y_train,dtype=np.float64)

def forward(w,X):
    return np.dot(X,w)+b

def sigmoid(x):
    return 1/(1+np.exp(-x))

def probability(w,X):
    return sigmoid(forward(w,X))

def cost(w,X,Y):
    total_cost=np.sum(-Y*np.log(probability(w,X)))+np.sum(-(1-Y)*np.log(1-probability(w,X)))
    total_cost/=m
    return total_cost
grad_w=np.zeros((X.shape[1],1))
Cost=0
for i in range(1000):
        Cost=cost(w,x_train,y_train)
        h=forward(w,x_train)
        grad_b =np.sum(sigmoid(h)-y_train)
        for j in range(x_train.shape[1]):
            grad_w[j] = np.sum((sigmoid(h) - y_train) *x_train[:,j:j+1].T)
            w[j]-=lr*grad_w[j]
            b-=lr*grad_b

print("Final Cost of Training",Cost)
print("Final weight", w)
print("Final biased term for this model",b)












