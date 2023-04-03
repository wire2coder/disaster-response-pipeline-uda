"""
Sample Script Syntax:
> python train_classifier.py <path to sqllite  destination db> <path to the pickle file>

Sample Script Execution:
> python train_classifier.py ../data/disaster_response_db.db classifier.pkl

Arguments:
    1) Path to SQLite destination database (e.g. disaster_response_db.db)
    2) Path to pickle file name where ML model needs to be saved (e.g. classifier.pkl)
"""

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

# import libraries
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from scipy.stats import gmean
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin



def load_data(database_filepath):
    """
    Load 'database file'
    
    Arguments:
        database_filepath -> Path to SQLite database file

    Output:
        X -> a dataframe containing 'features/columns'
        Y -> a dataframe containing 'labels/answer'

        category_names -> List of categories name
    """
    
    engine1 = create_engine('sqlite:///'+database_filepath)    
    df = pd.read_sql_table('ETL_Preparation', engine1)
    
     # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    y = df.iloc[:,4:]

    # checking stuff
    # print(X)
    # print(y.columns)

    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    pass


def build_pipeline():
     """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """

    pipeline2 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline2


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        # > python train_classifier.py ../Piplines/ETL_Preparation.db model1.pkl

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        # print(f"X_train.shape: {X_train.shape} X_test.shape: {X_test.shape} Y_train.shape: {Y_train.shape} Y_test.shape: {Y_test.shape}")
        # X_train.shape: (20972,) X_test.shape: (5244,) Y_train.shape: (20972, 36) Y_test.shape: (5244, 36)
        
        print('Building model...')
        model = build_pipeline()
        
        # print('Training model...')
        # model.fit(X_train, Y_train)
        
        # print('Evaluating model...')
        # evaluate_model(model, X_test, Y_test, category_names)

        # print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # save_model(model, model_filepath)

        # print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()