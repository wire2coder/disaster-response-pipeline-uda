import sys
import re
import pickle
import numpy as np
import pandas as pd
import shutil

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin



def load_data(database_filepath):
    # what does this fuction do?

    engine1 = create_engine('sqlite:///'+database_filepath)    

    try:
        df = pd.read_sql_table('ETL_Preparation', engine1)
    except ValueError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    
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


def tokenize(text, url_place_holder_string="urlplaceholder"):
    # what does this fuction do?
    # 1. Replace all urls with a urlplaceholder string
    # 2. Extract all the urls from the provided text
    # 3. Replace url with a url placeholder string
    # 4. Extract the word tokens from the provided text
    # 5. Lemmanitizer to remove inflectional and derivationally related forms of a word
    # 6. List of clean tokens

    
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


## From /Piplines >> ML Pipeline Preparation.ipynb
# The InitialVerbDetector class is a custom scikit-learn transformer that identifies if any sentence
# within a given text starts with a verb. It inherits from BaseEstimator and TransformerMixin and
# implements the fit() and transform() methods.
class InitialVerbDetector(BaseEstimator, TransformerMixin):

    def detect_initial_verb(self, input_text):
        sentences = nltk.sent_tokenize(input_text)

        for sentence in sentences:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            initial_word, initial_tag = pos_tags[0]

            if initial_tag in ['VB', 'VBP'] or initial_word == 'RT':
                return True

        return False

    def transform(self, input_data):
        tagged_data = pd.Series(input_data).apply(self.detect_initial_verb)
        return pd.DataFrame(tagged_data)

    # Implements the scikit-learn transformer interface by returning 'self'
    def fit(self, input_data, target_data=None):
        return self


def build_pipeline():
    # using the improved version of 'pipe1' with GridSerchCV
    # from 'ML Pipeline Preparation.ipynb'
    # with these 'hyperparameters':
    # learning_rate=0.03
    # n_estimators=13

    pipe1 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier(learning_rate=0.03, n_estimators=13)))
    ])

    # 0526_Fri_16June2023
    # using 'GridSearchCV' to find the best parameters from a 'dict' of parameters
    # have to prefix the parameter names with clf__estimator__, 
    # use clf__estimator__learning_rate and clf__estimator__n_estimators in the parameter_dict1.
    parameter_dict1 = {
        'classifier__estimator__learning_rate': [0.01, 0.03],
        'classifier__estimator__n_estimators': [10, 13]
    }

    from sklearn.model_selection import GridSearchCV
    betterPipe1 = GridSearchCV(pipe1, param_grid=parameter_dict1, cv=5, n_jobs=-1, verbose=3)

   
    # [CV 3/5] END classifier__estimator__learning_rate=0.03, classifier__estimator__n_estimators=13;, score=0.185 total time= 1.6min
    # [CV 4/5] END classifier__estimator__learning_rate=0.03, classifier__estimator__n_estimators=13;, score=0.182 total time= 1.6min
    # [CV 5/5] END classifier__estimator__learning_rate=0.03, classifier__estimator__n_estimators=13;, score=0.195 total time= 1.4min

    return betterPipe1


def evaluate_pipeline(pipeline, X_test, Y_test, category_names):

    Y_pred = pipeline.predict(X_test)
    
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    
    # Print classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))


def save_model(model1, filepath1):
    import os

    # check if the file already exists, if so, delete it
    if os.path.exists(filepath1):
        os.remove(filepath1)

    with open(filepath1, 'wb') as file1:
        pickle.dump(model1, file1)
    

def copyfile1():
    source = 'Piplines/ETL_Preparation.db'
    destination = 'models'

    try:
        shutil.copy(source, destination)
        print(f'File copied successfully.')
    except Exception as e:
        print(f'An error occurred while copying the file: {e}')
        sys.exit(1)


def main():

    if len(sys.argv) == 3:
        # make sure to use a different 'pickle' file name for each run
        # python train_classifier.py ../Piplines/ETL_Preparation.db new1.pkl 

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        # print(f"X_train.shape: {X_train.shape} X_test.shape: {X_test.shape} Y_train.shape: {Y_train.shape} Y_test.shape: {Y_test.shape}")
        # X_train.shape: (20972,) X_test.shape: (5244,) Y_train.shape: (20972, 36) Y_test.shape: (5244, 36)
        
        print('Building model...')
        model = build_pipeline()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_pipeline(model, X_test, Y_test, category_names)

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