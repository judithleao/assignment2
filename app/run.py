import json
import plotly
import pandas as pd
import plotly.express as px
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
#from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# Script to import figures and the needed pca module
from sklearn.decomposition import PCA

from langdetect import detect
# This helped: https://knowledge.udacity.com/questions/49726, cite this
# Consider creating a package: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder, answer by joey
import sys
sys.path.append("../models")
from length_estimator import LengthOfMessage
from train_classifier import return_figures



#-------------------------------------------------------------------------------------------------------#

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages_Table', engine)

# This import is an interim solution >> Should save the categories names as file or db and import same as df
category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', 
       'medical_products', 'search_and_rescue', 'security', 'military',
       'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
       'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport',
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',
       'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
       'storm', 'fire', 'earthquake', 'cold', 'other_weather',
       'direct_report']

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():     
    
    figures = return_figures(df, category_names) # as defined via main() function in train_classifier.py

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()