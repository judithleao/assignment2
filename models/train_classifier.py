'''
>> Potentially need to install <<
pip install plotly_express
pandas must be updated
'''

# Import libraries
import sys
import nltk
nltk.download(['punkt', 
               'wordnet', 
               'averaged_perceptron_tagger', 
               'stopwords'])

import re
import numpy as np
import pandas as pd
import plotly.express as px
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
# my own module
from length_estimator import LengthOfMessage


################################################################
############################ FUNCTIONS #########################
################################################################

def load_data(database_filepath):
    '''
    INPUT: path leading to database
    OUTPUT: 
        X and Y as independent and dependent variables for training model. 
        df contains all data. 
        category_names (list) contains labels for dependent variables.
    PURPOSE: Load data and format for further use.
    '''
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql("SELECT * FROM Messages_Table", engine)
    category_names = pd.read_sql("SELECT * FROM Categories_Table", engine)
    category_names = category_names['0'].to_list()
    # Split the data in X and Y variables
    X = df['message']
    Y = df[category_names]
    return X, Y, category_names, df


def imbalance(df, category_names):
    '''
    INPUT: 
        dataframe with all data.
        category_names (list) contains labels for dependent variables.
    OUTPUT: dataframe that shows % of messages that had specific label as tag.
    PURPOSE: Analyse imbalance in dependent variables. Smaller % means more imbalanced.
    '''
    dict_value_counts = {}
    for i in category_names:
        class_not_chosen = df[i].value_counts()[0]
        class_chosen = df[i].value_counts()[1]
        dict_value_counts[i] = [class_chosen/(class_chosen + class_not_chosen)]
    df_imbalance_per_label = pd.DataFrame(dict_value_counts).transpose()
    df_imbalance_per_label = df_imbalance_per_label.rename(
        columns={0:'imbalance'}).sort_values(by=['imbalance'], ascending=False)
    return df_imbalance_per_label


def return_figures(df, category_names): 
    '''
    INPUT: 
        dataframe with all data.
        category_names (list) contains labels for dependent variables.
    OUTPUT: 4 charts.
    PURPOSE: Charting for website.
    '''
    ######################################################
    ############# FIGURE 1: PCA SCATTER PLOT #############
    ######################################################
    # Create co-occurence matrix
    # Source: Stack Overflow
    # Question by user3084006. Profile: https://stackoverflow.com/users/3084006/user3084006
    # Answer by alko. Profile: https://stackoverflow.com/users/1265154/alko
    # https://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas
    df_cooc = df[category_names].T.dot(df[category_names])
    np.fill_diagonal(df_cooc.values, 0)
    
    # Get category sizes
    category_size_dict = df[category_names].sum(axis=0).to_dict()
    
    # Create row percentages
    # Source: Stack Overflow
    # Question by yourselvs. Profile: https://stackoverflow.com/users/4593740/yourselvs
    # Answer by jpp. Profile: https://stackoverflow.com/users/9209546/jpp
    # https://stackoverflow.com/questions/50820659/compute-row-percentages-in-pandas-dataframe
    df_cooc = df_cooc.div(df_cooc.sum(axis=1), axis=0)
    
    # Create array
    X = df_cooc.to_numpy()
    
    # Run PCA and compile dataframe with vectors, labels and size
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df_components = pd.DataFrame(components)
    df_components['category'] = category_names
    # Source: Stack Overflow
    # Question by Fabio Lamanna. Profile: https://stackoverflow.com/users/2699288/fabio-lamanna
    # Answer by EdChum. Profile: https://stackoverflow.com/users/704848/edchum
    # https://stackoverflow.com/questions/29794959/pandas-add-new-column-to-dataframe-from-dictionary
    df_components['category_size'] = df_components['category'].map(category_size_dict)
    # Source: Stack Overflow
    # Question by UserYmY. Profile: https://stackoverflow.com/users/2058811/userymy
    # Answer by EdChum. Profile: https://stackoverflow.com/users/704848/edchum
    # https://stackoverflow.com/questions/28986489/how-to-replace-text-in-a-column-of-a-pandas-dataframe
    df_components['category'] = df_components['category'].str.replace("_"," ")
    
    # Retrieve total variance explained - see https://plotly.com/python/pca-visualization/
    total_var = pca.explained_variance_ratio_.sum() * 100
    
    # Plot
    fig1 = px.scatter(df_components, 
                 x=0, 
                 y=1, 
                 title=f'Total Explained Variance: {total_var:.2f}%', 
                 text='category', 
                 size='category_size',
                 labels={'0': 'PCA component 1', '1': 'PCA component 2'},
                 height=700
                 )
    
    ######################################################
    ############## FIGURE 2: IMBALANCE PLOT ##############
    ######################################################
    fig2 = px.bar(imbalance(df, category_names), 
                  labels={
                      'index': 'Topic', 
                      'value': 'Share of messages tagged with Topic'}, 
                  height=650)
    # Source: plotly community
    # Answer by vitaminc. Profile: https://community.plotly.com/u/vitaminc
    # https://community.plotly.com/t/plotly-express-scatter-mapbox-hide-legend/36306
    fig2.layout.update(showlegend=False)
                
    ######################################################
    ################## FIGURE 3: GENRES ##################
    ######################################################
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    fig3 = px.bar(x=genre_names, y=genre_counts, 
                  labels={
                      'x': 'Genre', 
                      'y': 'Number of messages'})
    
    ######################################################
    ################ FIGURE 4: Languages #################
    ######################################################
    language_df = pd.DataFrame(df['language'].value_counts())
    language_df = language_df.drop(['n/a'], axis=0)
    fig4 = px.bar(language_df, 
                  labels={
                      'index': 'Language', 
                      'value': 'Number of messages'})
    fig4.layout.update(showlegend=False)
    #----------------------------------------------------#
    
    figures = [
               fig1, 
               fig2, 
               fig3, 
               fig4
              ]
    return figures   
    
    
def tokenize(text):
    '''
    INPUT: text as instance of 'message' from X.
    OUTPUT: lemmed as list of separate words.
    PURPOSE: transforms text into list of words ready for modelling.
    '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed


def build_model():
    '''
    INPUT: None.
    OUTPUT: cv as GridSearchCV object.
    PURPOSE: set up pipeline and parameters for model.
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)), 
                ('tfidf', TfidfTransformer())
            ])),
            ('length_ex', LengthOfMessage())
        ])),
        ('moc', MultiOutputClassifier(RandomForestClassifier(class_weight="balanced")))
    ])
    parameters = {
        'moc__estimator__max_depth': (50,
                                      100),
        'moc__estimator__n_estimators': (10,
                                         20)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3, n_jobs=4)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT: 
        model as GridSearchCV object.
        X_test as testing data for independent variables.
        Y_test as testing data for dependent variables.
        category_names contains labels for dependent variables.
    OUTPUT: None
    PURPOSE: prints metrics to evaluate model performance.
    '''
    Y_pred = model.predict(X_test)
    # List best PARAMETERS
    print("These are the best parameters: {}".format(model.best_params_))
    
    # List ACCURACY per Y-value
    accuracy_dict = {}
    for i, j in enumerate(category_names):
        accuracy_dict[j] = accuracy_score(Y_test.iloc[:,i], Y_pred[:,i])
    df_accuracy = pd.DataFrame.from_dict(accuracy_dict, orient='index')
    df_accuracy = df_accuracy.rename(columns={0:'accuracy'})
    print("This is the accuracy per label:")
    print(df_accuracy)
    
    # List CLASSIFICATION REPORTS per Y-value
    for i, j in enumerate(category_names):
        print("Classification report for label {}".format(j))
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
        
# Source: Machine Learning mastery
# By Jason Brownlee, 08/06/2016 in Python Machine Learning
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
def save_model(model, model_filepath):
    '''
    INPUT: 
       model as GridSearchCV object.
       model_filepath as path to store model outputs.
    OUTPUT: None
    PURPOSE: Comverts model outputs into pickle file and saves to filepath.
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    '''
    INPUT: None
    OUTPUT: None
    PURPOSE: Calls other functions in logical order and fits model.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names, df = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    


if __name__ == '__main__':
    main()