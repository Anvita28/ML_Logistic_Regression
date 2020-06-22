# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:41:43 2019

@author: Anvita Bansal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('training_titanic.csv')
test_data=pd.read_csv('test_titanic.csv')
#print(train_data.columns)



count=train_data.count()
print(count)
train_data.drop('Cabin',axis=1,inplace=True)
print(train_data.head())
# giving average age values to null age values
train_data.Age.fillna(train_data.Age.mean(),inplace=True)
print(train_data.Age.count())

train_data.dropna(inplace=True)
train_data.drop('Name',axis=1,inplace=True)
train_data.drop('Ticket',axis=1,inplace=True)
count=train_data.count()
print(count)
#cleaning of train data done
train_data['gender']=train_data['Sex']

def f(s):
    if(s=='Male'):
        return 0
    else:
        return 1
    
train_data['Sex']=train_data.gender.apply(f)
train_data.drop('gender',axis=1,inplace=True)

train_data['Emb']=train_data['Embarked']

def g(s):
    if(s=='S'):
        return 0
    elif(s=='Q'):
        return 1
    else:
        return 2
train_data['Embarked']=train_data.Emb.apply(g)
train_data.drop('Emb',axis=1,inplace=True)
print(train_data.tail())

X_train=train_data.drop('Survived',axis=1)
print(X_train)
Y_train=train_data['Survived']
print(Y_train)

#.....................
#Same cleaning process for test data

test_data.drop('Cabin',axis=1,inplace=True)
test_data.Age.fillna(test_data.Age.mean(),inplace=True)
print(test_data.Age.count())

test_data.dropna(inplace=True)
test_data.drop('Name',axis=1,inplace=True)
test_data.drop('Ticket',axis=1,inplace=True)
count=test_data.count()
print(count)
#cleaning of data done
test_data['gender']=test_data['Sex']

def f(s):
    if(s=='Male'):
        return 0
    else:
        return 1
    
test_data['Sex']=test_data.gender.apply(f)
test_data.drop('gender',axis=1,inplace=True)

test_data['Emb']=test_data['Embarked']

def g(s):
    if(s=='S'):
        return 0
    elif(s=='Q'):
        return 1
    else:
        return 2
test_data['Embarked']=test_data.Emb.apply(g)
test_data.drop('Emb',axis=1,inplace=True)
print(test_data.tail())


from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(X_train,Y_train)
pred=lr.predict(test_data)
print(pred)
np.savetxt('predTitanic.csv',pred,delimiter=',')

#from sklearn inmport decisio
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
y_pred=clf.predict(test_data)
print(clf.score(X_train,Y_train)),print(clf.predict(test_data,y_pred))

