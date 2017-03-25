from __future__ import division
import argparse
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, f1_score, precision_score, confusion_matrix, precision_recall_curve, mean_squared_error
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, mutual_info_classif
import time
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def feature_selection_imp(train_matrix, test_matrix, train_targets, dictionary_file):
	clf = ExtraTreesClassifier()
	clf = clf.fit(train_matrix, train_targets)
	imp = clf.feature_importances_
	# for i in xrange(len(imp)):
	#  	print imp[i]

	num = xrange(len(imp))
	vocab = []
	with open(dictionary_file, "r") as f:
		for line in f:
			vocab.append(line)
	numberedImp = zip(imp, num, vocab)

	# Change threshold to vary number of features selected
	threshold = 0.0000000000000000

	numsToDel = []
	wordsToDel = []
	for i in xrange(len(numberedImp)):
		val, no, word = numberedImp[i]
		if val < threshold:
			numsToDel.append(no)
			wordsToDel.append(word)

	print wordsToDel
	print len(wordsToDel)

	train_matrix = SelectFromModel(clf, threshold= threshold, prefit=True).transform(train_matrix)
	test_matrix = SelectFromModel(clf, threshold= threshold, prefit=True).transform(test_matrix)

	num_features = len(train_matrix[0])
	# print len(train_matrix[0])
	# print len(test_matrix[0])

	return train_matrix, test_matrix, num_features

def feature_selection_chi2(train_matrix, test_matrix, train_targets, k_best):
	k = SelectKBest(chi2, k=k_best)
	train_matrix = k.fit_transform(train_matrix, train_targets)
	test_matrix = k.transform(test_matrix)
	num_features = len(test_matrix[0])
	return train_matrix, test_matrix, num_features

def feature_selection_mutualinformation(train_matrix, test_matrix, train_targets, dictionary_file):
	mi = SelectKBest(mutual_info_classif, k=200)
	train_matrix = mi.fit_transform(train_matrix, train_targets)
	test_matrix = mi.transform(test_matrix)
	num_features = len(train_matrix[0])
	return train_matrix, test_matrix, num_features

def run_classifications(X, y, X_test, labelname):

    X, X_test, n = feature_selection_chi2(X, X_test, y, 2000)

    # gnb = GaussianNB()
    # start_time = time.time()
    # gnb.fit(train_matrix, train_targets)
    # runtime = str(time.time() - start_time)
    # y_train = gnb.predict(train_matrix)
    # y_test = gnb.predict(test_matrix)

    # print_stats("Gaussian Naive Bayes", train_targets, test_targets, y_train, y_test, runtime, num_features)

    # y_prob = gnb.predict_proba(test_matrix)
    # fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
    # per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
    # # pr_auc = auc(per, rec)
    # roc_auc = auc(fpr, tpr)

    # plt.plot(fpr, tpr, lw=2, color='#83b2d0',label='Gaussian Naive Bayes ROC (area = %0.2f)' % (roc_auc))

    # pers.append(per)
    # recalls.append(rec)
    # threshs.append(thresh)
    # # pr_aucs.append(pr_auc)

    # ######

    rf = RandomForestClassifier()
    start_time = time.time()
    rf.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = rf.predict(X)
    y_test = rf.predict(X_test)

    print_classification_stats("Random Forest " + labelname, y, y_train, y_test, runtime)


def run_regressions(X, y, X_test, labelname):

    y_new = np.multiply(y, 100).astype(int)
    X, X_test, n = feature_selection_chi2(X, X_test, y_new, 2000)
    #X, X_test, n = feature_selection_chi2(X, X_test, y, 2000)

    # gnb = GaussianNB()
    # start_time = time.time()
    # gnb.fit(train_matrix, train_targets)
    # runtime = str(time.time() - start_time)
    # y_train = gnb.predict(train_matrix)
    # y_test = gnb.predict(test_matrix)

    # print_stats("Gaussian Naive Bayes", train_targets, test_targets, y_train, y_test, runtime, num_features)

    # y_prob = gnb.predict_proba(test_matrix)
    # fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
    # per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
    # # pr_auc = auc(per, rec)
    # roc_auc = auc(fpr, tpr)

    # plt.plot(fpr, tpr, lw=2, color='#83b2d0',label='Gaussian Naive Bayes ROC (area = %0.2f)' % (roc_auc))

    # pers.append(per)
    # recalls.append(rec)
    # threshs.append(thresh)
    # # pr_aucs.append(pr_auc)

    # ######
    lr = LinearRegression()
    start_time = time.time()
    lr.fit(X, y)
    runtime = str(time.time() - start_time)
    y_train = lr.predict(X)
    print y_train[:100]
    y_test = lr.predict(X_test)

    print_regression_stats("Random Forest " + labelname, y, y_train, y_test, runtime)

def print_classification_stats(name, y, y_train, y_test, runtime, num_features=None):

    if num_features is not None:
        print "Number of features selected: " + str(num_features)

    # Accuracy score
    print name + " Train Accuracy: " + str(accuracy_score(y, y_train))
    # print name + " Test Accuracy: " + str(accuracy_score(test_targets, y_test))

    # Precision
    print name + " Train Precision Score: " + str(precision_score(y, y_train))
    # print name + " Test Precision Score: " + str(precision_score(test_targets, y_test))

    # Recall
    print name + " Train Recall Score: " + str(recall_score(y, y_train))
    # print name + " Test Recall Score: " + str(recall_score(test_targets, y_test))

    # F1
    print name + " Train F1 Score: " + str(f1_score(y, y_train))
    # print name + " Test F1 Score: " + str(f1_score(test_targets, y_test))

    # Runtime
    print name + " Fitting Runtime: " + runtime

def print_regression_stats(name, y, y_train, y_test, runtime, num_features=None):

    if num_features is not None:
        print "Number of features selected: " + str(num_features)

    # Accuracy score
    print name + " Mean Squared Error: " + str(mean_squared_error(y, y_train))

    # Runtime
    print name + " Fitting Runtime: " + runtime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', default=0, type=int)
    parser.add_argument('--r', default=0, type=int)
    args = parser.parse_args()

    classify = args.c
    regress = args.r

    try:
        X = np.load('X.npy')
        X_test = np.load('X_test.npy')
        y = np.load('y.npy')
        print "Loaded Train/Test Data from memory..."
    except IOError:
        X = np.genfromtxt('bg_train.csv', delimiter=',', dtype=float)
        X_test = np.genfromtxt('bg_test.csv', delimiter=',', dtype=float)
        y = np.genfromtxt('train_labels_filled.csv', delimiter=',')

        print('{} : {}'.format("Shape of X", X.shape))
        print('{} : {}'.format("Shape of y", y.shape))
        # print('{} : {}'.format("Shape of y_label", y_grit.shape))
        print('{} : {}'.format("Shape of X_test", X_test.shape))

        print('Removing bad columns')
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
        X = np.delete(X, cols, axis=1)
        X_test = np.delete(X_test, cols, axis=1)
        print('Removed ' + str(len(cols)) + ' cols')
        print('{} : {}'.format("Final shape of X", X.shape))
        print('{} : {}'.format("Final shape of X_test", X_test.shape))

        np.save('X.npy', X)
        np.save('X_test.npy', X_test)
        np.save('y.npy', y)
        print "Saved Train/Test Data to memory..." 
    
    
    y_grit = y[:,1]
    y_gpa = y[:,2]
    y_mhardship = y[:,3]
    y_eviction = y[:,4]
    y_jobloss = y[:,5]
    y_jobtraining = y[:,6]
    

    if classify:
        print("-----------------Eviction-----------------")
        run_classifications(X, y_eviction, X_test, "Eviction")
        print("-----------------Job Loss-----------------")
        run_classifications(X, y_jobloss, X_test, "Job Loss")
        print("---------------Job Training---------------")
        run_classifications(X, y_jobtraining, X_test, "Job Training")
    if regress:
        print("----------------------Grit---------------------")
        run_regressions(X, y_grit, X_test, "Grit")
        print("----------------------GPA----------------------")
        run_regressions(X, y_gpa, X_test, "GPA")
        print("---------------Material Hardship---------------")
        run_regressions(X, y_mhardship, X_test, "Material Hardship")

main()
