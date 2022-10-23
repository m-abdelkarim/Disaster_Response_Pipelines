
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

class TextClassifier:
    def __init__(self, database_filepath, model_filepath, table_name, col_name_text, list_categories):
        self.database_filepath =  database_filepath
        self.table_name = table_name
        self.model_filepath = model_filepath
        self.col_name_text = col_name_text
        self.list_categories = list_categories


    def load_data(self):
        """
        Load data from the given data base.
        Return the message data and the corresponding categories.

        Args:
        None

        Returns:
        X: Data frame that holds the messages.
        Y: Data frame that holds the corresponding categories.
        """
        engine = create_engine('sqlite:///' + self.database_filepath)
        df = pd.read_sql_table(self.table_name, engine)
        X = df[self.col_name_text]
        Y = df[self.list_categories]
        return X, Y

    def build_model(self):
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


    def evaluate_model(self, model, X_test, Y_test):
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


    def save_model(self, model):
        """
        Save the machine learning model to a file.

        Args:
        model: The model to be saved.

        Returns:
        None
        """
        pickle.dump(model, open(self.model_filepath, 'wb'))

    def train_save(self):
        print('Loading data...\n    DATABASE: {}'.format(self.database_filepath))
        X, Y = self.load_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = self.build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        self.evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(self.model_filepath))
        self.save_model(model)

        print('Trained model saved!')
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        table_name = 'Classified_Messages'
        col_name_message = 'message'
        engine = create_engine('sqlite:///' + database_filepath)
        df = pd.read_sql_table(table_name, engine)
        categories_list = df.drop(['id', 'message', 'original', 'genre'],axis=1).columns.values

        disaster_clf = TextClassifier(database_filepath, model_filepath,
            table_name, col_name_message, categories_list)
        
        disaster_clf.train_save()

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()