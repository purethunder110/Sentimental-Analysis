import os
#import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

#loading dataset
positive_training = 'dataset/train/pos'
positive_test = 'dataset/test/pos'
negative_training = 'dataset/train/neg'
negative_test = 'dataset/test/neg'

print("loading dataset....")
positive_set_train = []
for filename in os.listdir(positive_training):
    with open(os.path.join(positive_training, filename), 'r') as file:
        review = file.read()
        positive_set_train.append(review)

negative_set_train = []
for filename in os.listdir(negative_training):
    with open(os.path.join(negative_training, filename), 'r') as file:
        review = file.read()
        negative_set_train.append(review)

positive_set_test = []
for filename in os.listdir(positive_test):
    with open(os.path.join(positive_test, filename), 'r') as file:
        review = file.read()
        positive_set_test.append(review)

negative_set_test = []
for filename in os.listdir(negative_test):
    with open(os.path.join(negative_test, filename), 'r') as file:
        review = file.read()
        negative_set_test.append(review)

# combine positive and negative reviews for training
all_reviews = positive_set_train + negative_set_train
labels = [1] * len(positive_set_train) + [0] * len(negative_set_train)

# create a tf-idf vectorizer to convert text to numerical features
print("making tf-Vectorizer of data...")
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(all_reviews)

# train a SVM classifier
print("Training Classifier..")
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train_tfidf, labels)

# load and process test data
print("testing model...")
X_test = positive_set_test + negative_set_test
y_test = [1] * len(positive_set_test) + [0] * len(negative_set_test)

# transform test data using the same TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# predict sentiment for test data
y_pred = svm_classifier.predict(X_test_tfidf)

# evaluate the model on the test data
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy on Test Data: {accuracy}")
print("Classification Report on Test Data:\n", report)

# save the processed training data
"""
processed_train_df = pd.DataFrame({'text': all_reviews, 'label': labels})
processed_train_df.to_csv('data_processed.csv', index=False)
"""
joblib.dump(tfidf_vectorizer,open('vector.bin','wb'))
joblib.dump(svm_classifier,open('classifier.bin','wb'))
print("model saved....")
