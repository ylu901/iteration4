# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import inline as inline
import matplotlib
import os
import matplotlib.pyplot as plt  # this is used for the plot the graph
import seaborn as sns  # used for plot interactive graph.
import numpy as np
import scipy.stats as ss

import pandas as pd
from matplotlib import pyplot

df = pd.read_csv('heart.csv')
pd.set_option('display.width',400)
pd.set_option('display.max_columns',14)
df= df.loc[df.thalach >=80]
df= df.loc[df.trestbps <=180]
df= df.loc[df.oldpeak <=5]
df= df.loc[df.chol <=400]
df.duplicated()
df=df.drop_duplicates()
from sklearn.model_selection import train_test_split
X=df.drop('target',axis=1)
Y=df.target
X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.3)



from sklearn import tree
from sklearn.metrics import accuracy_score
dto=tree.DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3)
dt=dto.fit(X,Y)
importance =dt.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
opt=dto.fit(X_train, y_train)

predtr=opt.predict(X_train)
predtt=opt.predict(X_test)

print('Decision Tree train accuracy score of: ',
      round(accuracy_score(y_train,predtr),4)*100,'%')
print('Decision Tree test accuracy score of: ',
      round(accuracy_score(y_test,predtt),4)*100,'%')
from sklearn.tree import export_graphviz
dotfile=open('tree.dot','w')
export_graphviz(opt, dotfile,feature_names=X.columns, filled=True, rounded=True)
from graphviz import Source
dotfile=open('tree.dot','r')
text=dotfile.read()
Source(text)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 6, p=1)

opt=knn.fit(X_train, y_train)
predtr=opt.predict(X_train)
predtt=opt.predict(X_test)

print('KNN training accuracy score of: ',
      round(accuracy_score(y_train,predtr),4)*100,'%')
print('KNN test accuracy score of: ',
      round(accuracy_score(y_test,predtt),4)*100,'%')


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

features_names = ['age', 'sex','cp','trestbps','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

from sklearn import svm
svm = svm.SVC( kernel='linear', probability=True)
svm.fit(X, Y)
f_importances(svm.coef_[0], features_names)
opt=svm.fit(X_train, y_train)
predtr=opt.predict(X_train)
predtt=opt.predict(X_test)

print('SVM training accuracy score of: ',
      round(accuracy_score(y_train,predtr),4)*100,'%')
print('SVM test accuracy score of: ',
      round(accuracy_score(y_test,predtt),4)*100,'%')



from sklearn.neural_network import MLPClassifier
nn= MLPClassifier(activation='identity')

opt=nn.fit(X_train, y_train)
predtr=opt.predict(X_train)
predtt=opt.predict(X_test)

print('NN training accuracy score of: ',
      round(accuracy_score(y_train,predtr),4)*100,'%')
print('NN test accuracy score of: ',
      round(accuracy_score(y_test,predtt),4)*100,'%')