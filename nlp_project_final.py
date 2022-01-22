# -*- coding: utf-8 -*-
"""
Introduction to A.I(COMP237-001)
Group 3: Ari Cerrahyan, Alex Green, Maziar Hassanzadeh, Jaeuk Kim, Gaeun Kim
"""
##############################################################################
''' Load the data into a pandas data frame '''
import pandas as pd
import os

# path = r"C:\Users\USER\Desktop\Jack\Centennial College\COMP237_001_Introduction to A.I\Assignments\NLP_Group_Project"
filename = "Youtube03-LMFAO.csv"
relative_path = os.getcwd()
# fullpath = os.path.join(path, filename)
fullpath = os.path.join(relative_path, filename)
original_file = pd.read_csv(fullpath)
# original_file = pd.read_csv(filename)
##############################################################################

''' Data Exploration '''
print('='*70)
print(original_file.info())
print('\nOriginal_file Shape:', original_file.shape)
print('\nFirst 4 Rows',original_file.head(4),'\n'+'='*70 + "\n")

# we need only two columns for the prjects
# content ===> each comment (document)
# class ===> 0 means normal comment, 1 means it is a spam

raw_data = original_file.iloc[:, 3:]
print(raw_data.info())
print('\nRaw Data(Content and Class columns) Shape:\n', raw_data.shape)
print('\nCheck if there is any missing value\n' ,raw_data.isna().sum())
print('\n\nRaw_Data:\n', raw_data, '\n'+ '='*70 + "\n")

##############################################################################
''' Shuffle the data '''
# before shufflling
print('Before Shuffling\n', raw_data)

# shuffling
raw_data = raw_data.sample(frac=1)

#after shufflling
print('\nAfter Shuffling\n', raw_data, '\n' + '='*70 + '\n')

##############################################################################

'''  Data Split'''
# each item in X is a comment(a document)
# each item in Y is the classification  |  0 is normal, 1 is spam
raw_docs = raw_data.iloc[:, 0]
raw_class = raw_data.iloc[:, 1]

#Replace numeric values in raw_class into Noraml/Spam | Numeric values to Strings(0 -> Normal, 1 -> Spam)
raw_class = raw_class.replace([0, 1], ["Normal", "Spam"])
print('Raw classes \n\n', raw_class,'\n' +'='*70 + '\n')

##############################################################################
''' Count Vectorization '''
#Convert documents into vectors | 
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
count_vec_X = count_vectorizer.fit_transform(raw_docs)

print("After Vectorization && Before TF-IDF\n")
print("Number of Documents: ", count_vec_X.shape[0])
print("Number of Feature Words: ", count_vec_X.shape[1])

# print("Feature Words")
# print(count_vectorizer.get_feature_names())

##############################################################################

''' tf-idf transformation'''
# Create the tf-idf transformer
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

total_tfidf = tfidf.fit_transform(count_vec_X)

print("After TF-IDF transformation\n")
print("Number of Documents:", total_tfidf.shape[0])
print("Number of Feature Words:", total_tfidf.shape[1], '\n')


print("Some of Total TF-IDF\n",total_tfidf[:, :3], '\n' + '='*70 + '\n')

# Using pandas split your dataset into 75% for training and 25% for testing (Do not use test_train_ split)
ratio = int(0.75 * len(raw_data))
X_train, X_test = total_tfidf[:ratio], total_tfidf[ratio:]
y_train, y_test = raw_class[:ratio], raw_class[ratio:]

##############################################################################

# Train a Multinomial Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB().fit(X_train, y_train)

##############################################################################

# Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=5)
print('--------Mean of Model Accuracy--------\n:', accuracy.mean() )


# Test the model on the test data, print the confusion matrix and the accuracy of the model.

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

y_pred = classifier.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() # .ravel() is for a contiguous flattened array


print('\n--------The accuracy of the model on the test data--------\n:', accuracy_score(y_test, y_pred)) 
print('\n\n--------Confusion Matrix--------\n', confusion_matrix(y_test, y_pred))
print('\n\n --------Classification Report-----------\n', classification_report(y_test, y_pred), '\n'+ '='*70 + '\n')



##############################################################################
# As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results.
# Define test data 
input_data = [
    'Made my day', 
    'Check this link www.skynet.com You can find John Corner',
    'I listen to this song when I go to bed',
    'It\'s been yeras since the realse but still daxx good ',
    'I used to dance to this. So good for a party',    
    'Elon Musk will show you how to make money with Doge Coin. Visit this channel'
]


# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(input_data)

# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

# Print the outputs
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', category)


