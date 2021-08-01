'''
>> Potentially need to install <<
pip install plotly_express

pandas must be updated
'''

# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
    INPUT: 
    OUTPUT: 
    PURPOSE: 
    '''
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql("SELECT * FROM Messages_Table", engine)
    category_names = pd.read_sql("SELECT * FROM Categories_Table", engine)
    category_names = category_names['0'].to_list()
    # Splitting the data in X and Y variables
    X = df['message']
    #category_names = [col for col in list(df.columns) if col not in ['id', 'message', 'original', 'genre']]
    Y = df[category_names]
    return X, Y, category_names, df


def imbalance(df, category_names):
    '''
    INPUT: 
    OUTPUT: 
    PURPOSE: 
    '''
    dict_value_counts = {}
    for i in category_names: ## have not yet removed child_alone
        class_not_chosen = df[i].value_counts()[0]
        class_chosen = df[i].value_counts()[1]
        dict_value_counts[i] = [class_chosen/(class_chosen + class_not_chosen)]
    df_imbalance_per_label = pd.DataFrame(dict_value_counts).transpose()
    df_imbalance_per_label = df_imbalance_per_label.rename(columns={0:'imbalance'}).sort_values(by=['imbalance'], ascending=False)
    return df_imbalance_per_label


def return_figures(df, category_names): 
    '''
    INPUT: 
    OUTPUT: 
    PURPOSE: 
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
    df_cooc = df_cooc.div(df_cooc.sum(axis=1), axis=0)
    
    # Create array
    X = df_cooc.to_numpy()
    
    # Run PCA and compile dataframe with vectors, labels and size
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df_components = pd.DataFrame(components)
    df_components['category'] = category_names
    df_components['category_size'] = df_components['category'].map(category_size_dict)
    df_components['category'] = df_components['category'].str.replace("_"," ")
    
    # Retrieve total variance explained
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
    fig2 = px.bar(imbalance(df, category_names), labels={'index': 'Topic', 'value': 'Share of messages tagged with Topic'}, height=650)
    # Source: plotly community
    # Answer by vitaminc. Profile: https://community.plotly.com/u/vitaminc
    # https://community.plotly.com/t/plotly-express-scatter-mapbox-hide-legend/36306
    fig2.layout.update(showlegend=False)
                
    ######################################################
    ################## FIGURE 3: GENRES ##################
    ######################################################
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    fig3 = px.bar(x=genre_names, y=genre_counts, labels={'x': 'Genre', 'y': 'Number of messages'})
    
    ######################################################
    ################ FIGURE 4: Languages #################
    ######################################################
    language_df = pd.DataFrame(df['language'].value_counts())
    language_df = language_df.drop(['n/a'], axis=0)
    fig4 = px.bar(language_df, labels={'index': 'Language', 'value': 'Number of messages'})
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
    INPUT: 
    OUTPUT: 
    PURPOSE: 
    '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    stemmed = [PorterStemmer().stem(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    return lemmed


def build_model():
    '''
    INPUT: 
    OUTPUT: 
    PURPOSE: 
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)), 
                ('tfidf', TfidfTransformer())
            ])),
            ('length_ex', LengthOfMessage())
        ])),
        ('moc', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        #'vect__': (try, try, try),
        #'tfidf__': (try, try, try),
        #'moc__':  (try, try, try),
        'moc__estimator__class_weight':  ("balanced", 
                                          None)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT: 
    OUTPUT: 
    PURPOSE: 
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
        

def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
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