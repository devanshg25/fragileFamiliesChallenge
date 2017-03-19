from __future__ import division
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import timeit
import math
from sklearn.ensemble import RandomForestClassifier

X = np.genfromtxt('bg_train.csv', delimiter=',', dtype=float)
print X.shape
X_test = np.genfromtxt('bg_test.csv', delimiter=',', dtype=float)
print X_test.shape

y = np.genfromtxt('train_labels_filled.csv', delimiter=',')
y = y[:,4] #eviction column
print y.shape

cols = []
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if math.isnan(X[i,j]):
            if j not in cols:
                cols.append(j)
                #print X[i,j]
                #print ('{},{}'.format(i, j))

for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if math.isnan(X_test[i,j]):
            if j not in cols:
                cols.append(j)
                #print X[i,j]
                #print ('{},{}'.format(i, j))
print cols
print len(cols)
X = np.delete(X, cols, axis=1)
X_test = np.delete(X_test, cols, axis=1)
print X.shape
print X_test.shape

start = timeit.default_timer()
rf = RandomForestClassifier()
rf.fit(X, y)
print('{} : {}'.format("Elapsed Time", timeit.default_timer()-start))

print('{} : {}'.format("Training Accuracy", rf.fit(X, y).score(X,y)))

y_test = rf.predict(X_test)

#check = (y_test == y_true)
#num_correct = sum(check)
#test_accuracy = num_correct/len(check)
#print('{} : {}'.format("Test Accuracy", test_accuracy))

