## Making a Simple Flask API
So far, I've made a few different types of machine learning models.  However, all these models exist in notebooks and aren't available to be used out in the wild.  I know software engineers are the ones we rely on to deploy and maintain APIs and models in production, but I think it's important for data scientists to have a basic understanding of the process.  Basic is probably a good description for this API.

In my "NLP Classification Project" repository (https://github.com/Johnny-McGregor/NLP_Classification_Project) I played around with a handful of machine learning models to classify text as to whether it is science related.  I took the Logistic Regression model from that repository for this API.  The goal is to be able to provide the API with a JSON file that includes some text, and return a classification ("Science" or "Not Science") as well as the probability associated with the prediction.

I was helped in the construction of this little project by a tutorial given by Rachael Tatman of Kaggle during CareerCon 2019 tutorials (https://www.kaggle.com/rtatman/careercon-intro-to-apis) as well as a helpful article on Towards Data Science by Nguyen Ngo (https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166).

The files in this repository are what Heroku requires when deploying an API:

# Serve.py
This is the script containing the trained model

# Script.py
This is the script that applies the trained model

# openapi.yaml
This is the file to guide the API.  It describes that it will take a user provided JSON of text and return a JSON of the classification and probability of the text being science related.

# requirements.txt
This lists all of the Python libraries that need to be imported in order for this API to run

# runtime.txt
This is the version of Python that the API scripts were created with.
