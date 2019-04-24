from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from clf_model import NLPClassifier

app = Flask(__name__)
api = Api(app)

#instantiate new model
model = NLPClassifier()

#load trained classifier
clf_path = 'lib/models/ScienceClassifier.pkl'
with open(path, 'rb') as f:
    model.clf = pickle.load(f)

#load trained vectorizer
vect_path = 'lib/models/CountVectorizer.pkl'
with open(vect_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

#parse arguments
parser = reqparse.RequestParser()
parser.add_argument('query')

class SciencePredictor(Resource):
    def get(self):
        #find the user query
        args = parser.parse_args()
        user_query = args['query']

        #vectorize user's query to make prediction
        uq_vectorized = model.vectorizer_transform(
            np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        #output 'Science' or 'Not Science' and probability
        if prediction == 0:
            pred_text = 'Not Science'
        else:
            pred_text = 'Science'

        #round off the probability to three decimal places
        confidence = round(pred_proba[0], 3)

        #create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}

        return output

    #setup Api resource routing here
    #route the url to the resource
    api.add_resource(SciencePredictor, '/')


    if __name__ == '__main__':
        app.run(debug=True)
