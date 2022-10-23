

import sys
import pickle
import logging
import pandas as pd
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

sys.path.append( '..' )
from lib.nlp_lib import tokenize

def load_data(database_filepath):
    """
    Load data from the given data base.
    Return the message data and the corresponding categories.

    Args:
    database_filepath: The database file.

    Returns:
    X: Data frame that holds the messages.
    Y: Data frame that holds the corresponding categories.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Classified_Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'],axis=1)
    return X, Y

def build_model():
    """
    Build and return the model. The grid search is used to determine
    the best parameter set.

    Args:
    None

    Returns:
    pipeline: The ML pipeline for the classifier.
    """
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
        'tfidf__norm':['l1', 'l2'],
        'tfidf__use_idf':[True],
        'tfidf__sublinear_tf':[False],
        'clf__estimator__random_state': [0],
        'clf__estimator__min_samples_split':[5, 7]}
    
    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the given ML model based on the given test data.
    The evaluation is written in a log file.

    Args:
    model: The model to be tested.
    X_test: The messages to be classified
    Y_test: The corresponding categories.

    Returns:
    None
    """
    logger = logging.getLogger('mylogger')
    handler = logging.FileHandler('model_eval_result.log', mode='w')
    logger.addHandler(handler)
    logger.setLevel('DEBUG')
    
    # Predict the categories of the given texts
    Y_pred = pd.DataFrame(model.predict(X_test), columns = Y_test.columns)
    # Write the evaluation to the logging file
    for col in Y_test.columns:
        logger.info('Evaluation of: ' + col)
        logger.info(classification_report(Y_test[col].values, Y_pred[col].values))
    
    handler.close()
    logger.removeHandler(handler) 


def save_model(model, model_filepath):
    """
    Save the machine learning model to a file.

    Args:
    model: The model to be saved.
    model_filepath: The file path to save the model to.

    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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