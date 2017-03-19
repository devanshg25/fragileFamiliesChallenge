from __future__ import division
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import timeit

X = np.genfromtxt('bg_train.csv', delimiter=',', dtype=float)
X = X[:, 1:6]
print X.shape
y = np.genfromtxt('train_labels_filled.csv', delimiter=',')
print y.shape
y = y[:,4] #eviction column
#print y

start = timeit.default_timer()
mnb = MultinomialNB()
mnb.fit(X, y)
print('{} : {}'.format("Elapsed Time", timeit.default_timer()-start))

print('{} : {}'.format("Training Accuracy", mnb.fit(X, y).score(X,y)))

X_test = np.genfromtxt('output.csv', delimiter=',', dtype=float)
X_test = X_test[:,1:6]
#y_true = np.genfromtxt('test_classes_0.txt')
y_test = mnb.predict(X_test)
print y_test

#check = (y_test == y_true)
#num_correct = sum(check)

#test_accuracy = num_correct/len(check)

#print('{} : {}'.format("Test Accuracy", test_accuracy))

