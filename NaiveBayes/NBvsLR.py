#Credit Card Fraud Detection using Naive Bayes Classifer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,auc,accuracy_score,f1_score,roc_auc_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


df=pd.read_csv('creditcard.csv')
print("Pie chart for genuine(class 0) and fraud transactions:")
fig,ax=plt.subplots(1,1)
ax.pie(df.Class.value_counts(),autopct='%1.1f%%',labels=['Genuine','Fraud'],colors=['yellow','r'])
plt.axis('equal')
plt.ylabel('')
plt.show()

#plot time of transaction to see any trend in fraud and genuine transactions
print('Time Variable')
df['Time_Hr']=df['Time']/3600
print(df['Time_Hr'].tail(5))
fig ,(ax1,ax2)=plt.subplots(2,1, sharex=True,figsize=(6,3))
ax1.hist(df.Time_Hr[df.Class==0],bins=48,color='g',alpha=0.5)
ax2.hist(df.Time_Hr[df.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Time (hrs)')
plt.ylabel('Transactions')
plt.show()
#check amount Feature
fig, (ax3,ax4) = plt.subplots(2,1, figsize = (6,3), sharex = True)
ax3.hist(df.Amount[df.Class==0],bins=50,color='g',alpha=0.5)
ax3.set_yscale('log') # to see the tails
ax3.set_title('Genuine') # to see the tails
ax3.set_ylabel('# transactions')
ax4.hist(df.Amount[df.Class==1],bins=50,color='r',alpha=0.5)
ax4.set_yscale('log') # to see the tails
ax4.set_title('Fraud') # to see the tails
ax4.set_xlabel('Amount ($)')
ax4.set_ylabel('# transactions')
plt.show()

#standard scaling of amount feature
from sklearn.preprocessing import StandardScaler
df['scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'],axis=1)

gs=gridspec.GridSpec(28,1)
plt.figure(figsize=(6,28*4))

#Try to visualize it in jupyter Notebook
for i, col in enumerate(df[df.iloc[:,0:28].columns]):
    ax5=plt.subplot(gs[i])
    sns.distplot(df[col][df.Class==1],bins=50,color='r')
    sns.distplot(df[col][df.Class==0],bins=50,color='g')
    ax5.set_xlabel('')
    ax5.set_title('feature:' + str(col))
plt.show()

def split_data(df, drop_list):
    df = df.drop(drop_list,axis=1)
    print(df.columns)
    y = df['Class'].values  # target
    X = df.drop(['Class'], axis=1).values  # features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)


def get_predictions(clf, X_train, y_train, X_test):
    # create classifier
    clf = clf
    # fit it to training data
    clf.fit(X_train,y_train)
    # predict using test data
    y_pred = clf.predict(X_test)
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    #for fun: train-set predictions
    train_pred = clf.predict(X_train)
    print('train-set confusion matrix:\n', confusion_matrix(y_train,train_pred))
    return y_pred, y_pred_prob

def print_scores(y_test,y_pred,y_pred_prob):
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred))
    print("recall score: ", recall_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    print("f1 score: ", f1_score(y_test,y_pred))
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))

# Case-1 (Naive bayes without droping any feature)
drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)


# Case-NB-2 : drop some of principle components that have similar distributions in above plots
drop_list = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)

# Case-NB-3 : drop some of principle components + Time
drop_list = ['Time_Hr','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)

#Time_Hr is not helping much as a feature in classification

# Case-NB-4 : drop some of principle components + Time + 'scaled_Amount'
drop_list = ['scaled_Amount','Time_Hr','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)

#case-4 gives better model sensitivity (recall) and precision hence it is better to remove some redundant features to gain fast calculation and sensitivity for our model and it will also reduce model complexity and hence minimize the chances of overfitting
df=df.drop(drop_list,axis=1)
print(df.columns)

# let us check recall score for logistic regression
# Case-LR-1
y_pred, y_pred_prob = get_predictions(LogisticRegression(C = 0.01, penalty = 'l1'), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)


# full imbalanced dataset this default logistic regressor performs very poorly. So let us try to train it in tradional way i.e. from under-sampled data
# consider genuine-class cases which is equal to all fraud-classes i.e. consider 50/50 ratio of both classes.
# get indices for fraud and genuine classes
fraud_ind = np.array(df[df.Class == 1].index)
gen_ind = df[df.Class == 0].index
n_fraud = len(df[df.Class == 1])
# random selection from genuine class
random_gen_ind = np.random.choice(gen_ind, n_fraud, replace = False)
random_gen_ind = np.array(random_gen_ind)
# merge two class indices: random genuine + original fraud
under_sample_ind = np.concatenate([fraud_ind,random_gen_ind])
# Under sample dataset
undersample_df = df.iloc[under_sample_ind,:]
y_undersample  = undersample_df['Class'].values #target
X_undersample = undersample_df.drop(['Class'],axis=1).values #features

print("# transactions in undersampled data: ", len(undersample_df))
print("% genuine transactions: ",len(undersample_df[undersample_df.Class == 0])/len(undersample_df))
print("% fraud transactions: ", sum(y_undersample)/len(undersample_df))


# let us train logistic regression with undersamples data
# Case-LR-2
# split undersampled data into 80/20 train-test datasets.
# - Train model from this 80% fraction of undersampled data, get predictions from left over i.e. 20%.
drop_list = []
X_und_train, X_und_test, y_und_train, y_und_test = split_data(undersample_df, drop_list)
y_und_pred, y_und_pred_prob = get_predictions(LogisticRegression(C = 0.01, penalty = 'l1'), X_und_train, y_und_train, X_und_test)
print_scores(y_und_test,y_und_pred,y_und_pred_prob)

#Now, let us check its performance for the full skewed dataset. Just to mention: "train" from undersampled data, and "test" on full data.
# Case-LR-3
# "train" with undersamples, "test" with full data
# call classifier
lr = LogisticRegression(C = 0.01, penalty = 'l1')
# fit it to complete undersampled data
lr.fit(X_undersample, y_undersample)
# predict on full data
y_full = df['Class'].values #target
X_full = df.drop(['Class'],axis=1).values #features
y_full_pred = lr.predict(X_full)
# Compute predicted probabilities: y_pred_prob
y_full_pred_prob = lr.predict_proba(X_full)
print("scores for Full set")
print('test-set confusion matrix:\n', confusion_matrix(y_full,y_full_pred))
print("recall score: ", recall_score(y_full,y_full_pred))
print("precision score: ", precision_score(y_full,y_full_pred))

# Case-LR-4
y_p20_pred = lr.predict(X_test)
y_p20_pred_prob = lr.predict_proba(X_test)
print("scores for test (20% of full) set")
print('test-set confusion matrix:\n', confusion_matrix(y_test,y_p20_pred))
print("recall score: ", recall_score(y_test,y_p20_pred))
print("precision score: ", precision_score(y_test,y_p20_pred))


#NB vs LR recall score

#B confusion matrix

#LR confusion matrix

