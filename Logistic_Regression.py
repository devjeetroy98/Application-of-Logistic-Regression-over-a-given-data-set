import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
%matplotlib inline
df=pd.read_csv("C:\\Users\\abc\\Desktop\\project 2\\train.csv")
df.drop("Cabin",axis=1,inplace=True)
G=pd.get_dummies(df['Sex'])
Embark=pd.get_dummies(df['Embarked'])
Pcl=pd.get_dummies(df['Pclass'])
df=pd.concat([df,G,Embark,Pcl],axis=1)
df.drop(['Sex','Embarked','PassengerId','Ticket','Fare'],axis=1,inplace=True)
mean_value=df['Age'].mean()
df['Age']=df['Age'].fillna(mean_value)
df.drop(['Name'],axis=1,inplace=True)
X=df.drop("Survived",axis=1)
y=df["Survived"]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.15,random_state=38)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(X_train,y_train)
pred=reg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)