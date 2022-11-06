import sys
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
sys.path.append( '..' )
from lib.nlp_lib import tokenize

app = Flask(__name__)

def mean_for_gouped_df(df):
    """
    Helper function to use mean on grouped data frame (workaround)

    Args:
    df: The data frame to calculate the mean of its columns.

    Returns:
    Data frame that holda the mean of the columns
    of the input data frame
    """
    return df.mean(axis=0)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Classified_Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl", ).set_params(n_jobs=1)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # extract the genre counts and names
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #extract the mean of each category
    df_categories = df.drop(['id', 'message', 'original', 'genre'],axis=1)
    categories_mean = df_categories.mean(axis=0)
    # use 
    cat_mean_groupedby_genre = df.drop(['id', 'message', 'original'],axis=1).groupby('genre').agg(mean_for_gouped_df)
    cat_mean_direct = cat_mean_groupedby_genre.filter(items=['direct'], axis=0)
    cat_mean_news = cat_mean_groupedby_genre.filter(items=['news'], axis=0)
    cat_mean_social = cat_mean_groupedby_genre.filter(items=['social'], axis=0)
    
    
    # create visuals
    graphs = [
        {
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
        {
            'data': [
                Bar(
                    x=categories_mean.index,
                    y=categories_mean.values
                )
            ],

            'layout': {
                'title': 'The categories mean values for the whole dataset',
                'yaxis': {
                    'title': "mean"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_mean_direct.columns,
                    y=cat_mean_direct.values[0]
                )
            ],

            'layout': {
                'title': 'The categories mean values for the genre direct',
                'yaxis': {
                    'title': "mean"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_mean_news.columns,
                    y=cat_mean_news.values[0]
                )
            ],

            'layout': {
                'title': 'The categories mean values for the genre news',
                'yaxis': {
                    'title': "mean"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_mean_social.columns,
                    y=cat_mean_social.values[0]
                )
            ],

            'layout': {
                'title': 'The categories mean values for the genre social',
                'yaxis': {
                    'title': "mean"
                },
                'xaxis': {
                    'title': "category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
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
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()