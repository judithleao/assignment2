import json
import plotly
import pandas as pd
import plotly.express as px
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.decomposition import PCA

from langdetect import detect

# Source: Udacity Knowledge Question Thread
# Question by Lara Q
# Answer by Christophe B
# https://knowledge.udacity.com/questions/49726, cite this
import sys
sys.path.append("../models")
from length_estimator import LengthOfMessage
from train_classifier import return_figures
from train_classifier import tokenize

#-------------------------------------------------------------------------------------------------------#

app = Flask(__name__)

# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages_Table', engine)
category_names = pd.read_sql("SELECT * FROM Categories_Table", engine)
category_names = category_names['0'].to_list()


# Load model
model = joblib.load("../models/classifier.pkl")


# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():     
    
    figures = return_figures(df, category_names) # as defined via main() function in train_classifier.py

    # Plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()