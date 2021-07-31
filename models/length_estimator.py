################################################################
############################## CLASS ###########################
################################################################
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

class LengthOfMessage(BaseEstimator, TransformerMixin):

    def length_extractor(self, text):
        word_list = nltk.word_tokenize(text)
        length = len(word_list)
        return length

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_length = pd.Series(X).apply(self.length_extractor)
        return pd.DataFrame(X_length)