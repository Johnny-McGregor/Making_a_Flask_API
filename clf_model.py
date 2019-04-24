#import ML model
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

import pickle

class NLPClassifier(object):

    def __init__(self):
        """
        Attributes:
            clf: sklearn classifier model
            vecotrizer: CountVectorizer or similar
        """

        self.clf = LogisticRegression(C=3, random_state=42)
        self.vectorizer = CountVectorizer(max_df=.1, max_features=5000)

    def vectorizer_fit(self, X):
        """Fits the vectorizer to the text
        """
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transforms the text data into a dense matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """Trains the classifier to associate the label with the
        dense matrix
        """
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='./lib/models/CountVectorizer.pkl'):
        """save the trained vectorizer
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Vectorizer saved at {}".format(path))

    def pickle_clf(self, path='./lib/models/ScienceClassifier.pkl'):
        """
        saves the trained classifer models
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Classifer saved at {}".format(path))

    def classification_report(self, X, y, target_names):
        """get a classification report for metrics
        """
        classification_report(X, y, target_names = target_names)
