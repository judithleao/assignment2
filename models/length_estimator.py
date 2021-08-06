################################################################
############################## CLASS ###########################
################################################################
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

class LengthOfMessage(BaseEstimator, TransformerMixin):
    '''
    Class Purpose: Create bespoke estimator to add to Pipeline.
    Estimator Purpose: Estimator creates variables that measures length of a message.
    '''
    def length_extractor(self, text):
        '''
        INPUT: text (string) where each text passed is a separate message.
        OUTPUT: length of the text passed in.
        PURPOSE: create length as additional independent variables for model.
        '''
        word_list = nltk.word_tokenize(text)
        length = len(word_list)
        return length

    def fit(self, x, y=None):
        '''
        INPUT: x, y as arrays of independent and dependent variables.
        OUTPUT: self, the class instance.
        PURPOSE: required as part of the estimator construction to chain methods.
        '''
        return self

    def transform(self, X):
        '''
        INPUT: X, 2D-array, as 'message' variable.
        OUTPUT: dataframe containing lengths for all records.
        PURPOSE: transform data - add length to each record by using length_extractor function.
        '''
        X_length = pd.Series(X).apply(self.length_extractor)
        return pd.DataFrame(X_length)