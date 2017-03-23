from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, f1_score, precision_score, confusion_matrix, precision_recall_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

def build_feature_matrix(filename):
	matrix = []
	f = open(filename, 'rb')
	reader = csv.reader(f)
	for row in reader:	
		features = []
		for item in row:
			features.append(int(item))
		matrix.append(features)
	f.close()
	return matrix

def build_target_vector(filename):
	target = []
	with open(filename, "r") as f:
		for line in f:
			target.append(int(line))
	return target

def get_accuracy(prediction, actual):
	i = 0
	success = 0.0
	samples = len(actual)
	for val in prediction:
		if val == actual[i]:
			success = success + 1
		i = i + 1
	return str(success/samples * 100) + "%"

def parse_csv_train(filename, output):
	outfile = open(output, "w")
	with open(filename, 'rb') as file:
		reader = csv.reader(file)
		row_number = 0
		sample_number = 0
		positives = 0
		negatives = 0

		# Get 100,000 positive sentiments, and 100,000 negative sentiments
		begin_positives = 800001
		end_positives = 900000
		begin_negatives = 1
		end_negatives = 100000
		for row in reader:
			row_number = row_number + 1
			sample = ""
			if row_number >= begin_negatives and row_number <= end_negatives:
				sample_number = sample_number + 1
				classification = int(row[0]) 
				if classification == 4:
					sample = str(sample_number) + " " + row[5] + "	1" + '\n'
					positives = positives + 1
				elif classification == 0:
					sample = str(sample_number) + " " + row[5] + "	0" + '\n'
					negatives = negatives + 1
			if row_number >= begin_positives and row_number <= end_positives:
				sample_number = sample_number + 1
				classification = int(row[0]) 
				if classification == 4:
					sample = str(sample_number) + " " + row[5] + "	1" + '\n'
					positives = positives + 1
				elif classification == 0:
					sample = str(sample_number) + " " + row[5] + "	0" + '\n'
			outfile.write(sample)
		outfile.close()

def parse_csv_test(filename, output):
	outfile = open(output, "w")
	with open(filename, 'rb') as file:
		reader = csv.reader(file)
		row_number = 0
		sample_number = 0
		positives = 0
		negatives = 0

		# Get 10,000 positive sentiments, and 10,000 negative sentiments
		begin_positives = 900001
		end_positives = 910000
		begin_negatives = 100001
		end_negatives = 110000
		for row in reader:
			row_number = row_number + 1
			sample = ""
			if row_number >= begin_negatives and row_number <= end_negatives:
				sample_number = sample_number + 1
				classification = int(row[0]) 
				if classification == 4:
					sample = str(sample_number) + " " + row[5] + "	1" + '\n'
					positives = positives + 1
				elif classification == 0:
					sample = str(sample_number) + " " + row[5] + "	0" + '\n'
					negatives = negatives + 1
			if row_number >= begin_positives and row_number <= end_positives:
				sample_number = sample_number + 1
				classification = int(row[0]) 
				if classification == 4:
					sample = str(sample_number) + " " + row[5] + "	1" + '\n'
					positives = positives + 1
				elif classification == 0:
					sample = str(sample_number) + " " + row[5] + "	0" + '\n'
			outfile.write(sample)
		outfile.close()	

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

def feature_selection_chi2(train_matrix, test_matrix, train_targets, dictionary_file, k_best):
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

def run_tests(train_bow_filename, train_tv_filename, test_bow_filename, test_tv_filename, dictionary_file):
	train_matrix = build_feature_matrix(train_bow_filename)
	train_targets = build_target_vector(train_tv_filename)

	test_matrix = build_feature_matrix(test_bow_filename)
	test_targets = build_target_vector(test_tv_filename)
	train_matrix, test_matrix, num_features = feature_selection_chi2(train_matrix, test_matrix, train_targets, dictionary_file, 500)

	pers = []
	recalls = []
	threshs = []
	pr_aucs = []

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

	bnb = BernoulliNB()
	start_time = time.time()
	bnb.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = bnb.predict(train_matrix)
	y_test = bnb.predict(test_matrix)

	print_stats("Bernoulli Naive Bayes", train_targets, test_targets, y_train, y_test, runtime, num_features)

	y_prob = bnb.predict_proba(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)

	plt.plot(fpr, tpr, lw=2, color='#0f8880',label='Bernoulli Naive Bayes ROC (area = %0.2f)' % (roc_auc))

	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	###### 
	mnb = MultinomialNB()
	start_time = time.time()
	mnb.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = mnb.predict(train_matrix)
	y_test = mnb.predict(test_matrix)

	print_stats("Multinomial Naive Bayes", train_targets, test_targets, y_train, y_test, runtime)

	y_prob = mnb.predict_proba(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)

	plt.plot(fpr, tpr, lw=2, color='#f69a56',label='Multinomial Naive Bayes ROC (area = %0.2f)' % (roc_auc))

	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	###### 
	svml = SVC(kernel='linear')
	start_time = time.time()
	svml.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = svml.predict(train_matrix) 
	y_test = svml.predict(test_matrix) 

	print_stats("SVM Linear Kernel", train_targets, test_targets, y_train, y_test, runtime)

	y_score = svml.decision_function(test_matrix) 
	fpr, tpr, thresholds = roc_curve(test_targets, y_score)
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#a4a4d0',label='SVM Linear Kernel ROC (area = %0.2f)' % (roc_auc))


	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	###### 
 
	svmp = SVC(kernel='poly')
	start_time = time.time()
	svmp.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = svmp.predict(train_matrix) 
	y_test = svmp.predict(test_matrix) 

	print_stats("SVM Poly Kernel", train_targets, test_targets, y_train, y_test, runtime)

	y_score = svmp.decision_function(test_matrix) 
	fpr, tpr, thresholds = roc_curve(test_targets, y_score)
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#848484',label='SVM Poly Kernel ROC (area = %0.2f)' % (roc_auc))

	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	###### 

	knn3 = KNeighborsClassifier(n_neighbors=3)
	start_time = time.time()
	knn3.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = knn3.predict(train_matrix) 
	y_test = knn3.predict(test_matrix) 

	print_stats("10 Nearest Neighbors", train_targets, test_targets, y_train, y_test, runtime)

	y_prob = knn3.predict_proba(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#94a2fa',label='10-Nearest Neighbors ROC (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	#######
	boost = AdaBoostClassifier(base_estimator=MultinomialNB())
	start_time = time.time()
	boost.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = boost.predict(train_matrix) 
	y_test = boost.predict(test_matrix) 

	print_stats("AdaBoost Multinomial Naive Bayes", train_targets, test_targets, y_train, y_test, runtime)

	y_score = boost.decision_function(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_score)
	per, rec, thresh = precision_recall_curve(test_targets, y_score)
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#c25975',label='AdaBoost Multinomial Naive Bayes ROC (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	#######
	lr = LogisticRegression()
	start_time = time.time()
	lr.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = lr.predict(train_matrix) 
	y_test = lr.predict(test_matrix) 

	print_stats("Logistic Regression", train_targets, test_targets, y_train, y_test, runtime)

	y_score = lr.decision_function(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_score)
	per, rec, thresh = precision_recall_curve(test_targets, y_score)
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#d26bff',label='Logistic Regression ROC (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)


	#######
	dtc = DecisionTreeClassifier()
	start_time = time.time()
	dtc.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = dtc.predict(train_matrix) 
	y_test = dtc.predict(test_matrix) 

	print_stats("Decision Tree", train_targets, test_targets, y_train, y_test, runtime)

	y_prob = dtc.predict_proba(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#a4a4d0',label='Decision Tree (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	#######
	rf = RandomForestClassifier()
	start_time = time.time()
	rf.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = rf.predict(train_matrix) 
	y_test = rf.predict(test_matrix) 

	print_stats("Random Forest", train_targets, test_targets, y_train, y_test, runtime)

	y_prob = rf.predict_proba(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#fef56f',label='Random Forest (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	#######
	bc = BaggingClassifier(base_estimator=SVC(kernel='linear'), n_estimators=50)
	start_time = time.time()
	bc.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = bc.predict(train_matrix) 
	y_test = bc.predict(test_matrix) 

	print_stats("Bagging", train_targets, test_targets, y_train, y_test, runtime)

	y_prob = bc.predict_proba(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#fcd0bd',label='Bagging (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	#######
	mlp = MLPClassifier(max_iter=500)
	start_time = time.time()
	mlp.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = mlp.predict(train_matrix) 
	y_test = mlp.predict(test_matrix) 

	print_stats("MLP", train_targets, test_targets, y_train, y_test, runtime)

	y_prob = mlp.predict_proba(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_prob[:,1])
	per, rec, thresh = precision_recall_curve(test_targets, y_prob[:,1])
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#ebf3de',label='MLP (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	#######
	p = Perceptron()
	start_time = time.time()
	p.fit(train_matrix, train_targets)
	runtime = str(time.time() - start_time)
	y_train = p.predict(train_matrix) 
	y_test = p.predict(test_matrix) 

	print_stats("Perceptron", train_targets, test_targets, y_train, y_test, runtime)

	y_score = p.decision_function(test_matrix)
	fpr, tpr, thresholds = roc_curve(test_targets, y_score)
	per, rec, thresh = precision_recall_curve(test_targets, y_score)
	# pr_auc = auc(per, rec)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, lw=2, color='#fecdca',label='Perceptron (area = %0.2f)' % (roc_auc))
	
	pers.append(per)
	recalls.append(rec)
	threshs.append(thresh)
	# pr_aucs.append(pr_auc)

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	plt.show()
	# plt.savefig("roc_curves")

def print_stats(name, train_targets, test_targets, y_train, y_test, runtime, num_features=None):
	
	if num_features is not None:
		print "Number of features selected: " + str(num_features)

	# Accuracy score
	# print name + " Train Accuracy: " + str(accuracy_score(train_targets, y_train))
	print name + " Test Accuracy: " + str(accuracy_score(test_targets, y_test))

	# Precision
	# print name + " Train Precision Score: " + str(precision_score(train_targets, y_train))
	print name + " Test Precision Score: " + str(precision_score(test_targets, y_test))

	# Recall
	# print name + " Train Recall Score: " + str(recall_score(train_targets, y_train))
	print name + " Test Recall Score: " + str(recall_score(test_targets, y_test))

	# F1
	# print name + " Train F1 Score: " + str(f1_score(train_targets, y_train))
	print name + " Test F1 Score: " + str(f1_score(test_targets, y_test))

	# Runtime
	print name + " Fitting Runtime: " + runtime


def main():

	# # Unigrams
	# print "-------------------------------Unigrams----------------------------------------"
	# train_bog_file = "./unigrams_0/uni_train_bag_of_words_0.csv"
	# train_classes_file = "./unigrams_0/uni_train_classes_0.txt"
	# test_bog_file = "./unigrams_0/uni_test_bag_of_words_0.csv"
	# test_classes_file = "./unigrams_0/uni_test_classes_0.txt"
	# dictionary_file = "./unigrams_0/uni_train_vocab_0.txt"
	# run_tests(train_bog_file, train_classes_file, test_bog_file, test_classes_file, dictionary_file)

	# # Bigrams
	# print "-------------------------------Bigrams----------------------------------------"
	# train_bog_file = "./bigrams_0/bi_train_bag_of_words_0.csv"
	# train_classes_file = "./bigrams_0/bi_train_classes_0.txt"
	# test_bog_file = "./bigrams_0/bi_test_bag_of_words_0.csv"
	# test_classes_file = "./bigrams_0/bi_test_classes_0.txt"
	# run_tests(train_bog_file, train_classes_file, test_bog_file, test_classes_file)

	# Unigrams + Bigrams
	# print "---------------------------Unigrams + Bigrams---------------------------------"
	# train_bog_file = "./unigrams_bigrams_0/ub_train_bag_of_words_0.csv"
	# train_classes_file = "./unigrams_bigrams_0/ub_train_classes_0.txt"
	# test_bog_file = "./unigrams_bigrams_0/ub_test_bag_of_words_0.csv"
	# test_classes_file = "./unigrams_bigrams_0/ub_test_classes_0.txt"
	# dictionary_file = "./unigrams_bigrams_0/ub_train_vocab_0.txt"
	# run_tests(train_bog_file, train_classes_file, test_bog_file, test_classes_file, dictionary_file)

	# # Twitter Dataset
	# train_bog_file = "./twitter_datasets/twitter_train_bag_of_words_15.csv"
	# train_classes_file = "./twitter_datasets/twitter_train_classes_15.txt"
	# test_bog_file = "./twitter_datasets/twitter_test_bag_of_words_0.csv"
	# test_classes_file = "./twitter_datasets/twitter_test_classes_0.txt"
	# run_tests(train_bog_file, train_classes_file, test_bog_file, test_classes_file)

	# Twitter Dataset
	train_bog_file = "./twitter_train_100_bag_of_words_100.csv"
	train_classes_file = "./twitter_train_100_classes_100.txt"
	test_bog_file = "./twitter_test_100_bag_of_words_0.csv"
	test_classes_file = "./twitter_test_100_classes_0.txt"
	dictionary_file = "./twitter_train_100_vocab_100.txt"
	run_tests(train_bog_file, train_classes_file, test_bog_file, test_classes_file, dictionary_file)
	
	# Parse bigger dataset into text file
	# parse_csv_train("tweets_2.csv", "twitter_train_200000.txt")
	# parse_csv_test("tweets_2.csv", "twitter_test_20000.txt")

	# k-fold quantifies generalization error

main()