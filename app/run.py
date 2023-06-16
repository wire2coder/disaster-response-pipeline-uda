import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib << error
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


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


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
# fix1
engine = create_engine('sqlite:///./ETL_Preparation.db')
df = pd.read_sql_table('ETL_Preparation', engine)

# load model
# fix1
model = joblib.load("./mo3.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    word_name = df.iloc[ : , 4 : ].columns
    word_values = (df.iloc[ : , 4 : ] != 0).sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
            
        {  # graph 1, already here
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        { # graph 2, extra  
            'data': [
                Bar(
                    x = word_name,
                    y = word_values
                )
            ],

            'layout': {
                'title': 'Distribution of Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category (Words)",
                    'tickangle': 25
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls = plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
        query = query,
        classification_result = classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()