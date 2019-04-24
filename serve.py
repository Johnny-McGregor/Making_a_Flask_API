from clf_model import NLPClassifier

import pandas as pd
import numpy as np
import regex as re

from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

#define a function to clean up posts
def post_to_words(post):
    #return only text from each post
    post_text = BeautifulSoup(post).get_text()

    #use regex to eliminate punctuation
    letters_only = re.sub("[^a-zA-Z]", " ", post_text)

    #lowercase everything
    words = letters_only.lower().split()

    #assign set of all stop words ("a", "and", etc) to a variable
    stops = set(stopwords.words('english'))

    #keep only words that aren't in stops
    meaningful_words = [w for w in words if not w in stops]

    return(" ".join(meaningful_words))

#build and save a fit model
def build_model():
    model = NLPClassifier()

    with open('./lib/data/allposts.csv') as f:
        data = pd.read_csv(f)

    data.posts = data.posts.apply(post_to_words)

    model.vectorizer_fit(data.posts)
    print('Vectorizer has been fit')

    X = model.vectorizer_transform(data.posts)
    print('Transform is completed')

    y = data.science

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y)
    model.train(X_train, y_train)
    print('Model has been trained')

    model.pickle_clf()
    model.pickle_vectorizer()

    preds = model.predict(X_test)

    model.classification_report(y_test, preds,
                               target_names = ['Science', 'Not Science'])

if __name__ == "__main__":
    build_model()
