import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

class DataPreProcessing:
    def __init__(self, messages_filepath, categories_filepath, database_filepath):
        """
        The class init function. 

        Args:
        messages_filepath: The path of the csv file that holds the messages.
        categories_filepath: The path of the csv file that holds the categories of the messages.
        database_filepath: The database file path, in which the dataframe shall be saved.

        Returns:
        The data frame that holds the merged data.
        """
        self.messages_filepath = messages_filepath
        self.categories_filepath = categories_filepath
        self.database_filepath = database_filepath

    def load_data(self):
        """
        Load, merge and return the data given. 

        Args:
        None

        Returns:
        The data frame that holds the merged data.
        """
        messages = pd.read_csv(self.messages_filepath)
        categories = pd.read_csv(self.categories_filepath)
        df = pd.merge(messages, categories, on='id', how='inner')
        return df


    def clean_data(self, df):
        """
        This function cleans the given dataframe.
        The message categories are split into columns, and the duplicates are dropped.

        Args:
        df: The dataframe to be cleaned.

        Returns:
        df: The cleaned dataframe.
        """
        categories = df['categories'].str.split(pat = ';', expand = True)
        row = categories.iloc[0]
        category_colnames = row.apply(lambda x:x.split('-')[0])
        categories.columns = category_colnames
    
        # For each column, extract the last char of string and cast it to int
        for column in categories:
            categories[column] = categories[column].str[-1:]
            categories[column] = categories[column].astype('int')
        # Replace the categories column with the separated coloumns
        df.drop('categories', axis = 1, inplace = True)
        df = pd.concat([df, categories], axis = 1)
        # Drop duplicated rows and reset the index afterwards
        df.drop_duplicates(inplace = True)
        df.reset_index(drop = True, inplace = True)
    
        # remove rows that conatins invalid values
        for col in categories:
            indx = df.index[(df[col] != 0) & (df[col] != 1)].tolist()
            if len(indx):
                df.drop(index = indx, axis = 0, inplace = True)
        
        return df
    
    def save_data(self, df):
        """
        This function saves the given dataframe as table with
        name "Classified_Messages" in the given database file.

        Args:
        df: The dataframe to be saved.

        Returns:
        None
        """
        engine = create_engine('sqlite:///' + self.database_filepath)
        df.to_sql('Classified_Messages', engine, index = False, if_exists = 'replace')

    def run(self):
        """
        This is the run function of the class, in which the data is loaded,
        cleaned and saved to database.

        Args:
        None

        Returns:
        None
        """

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(self.messages_filepath, self.categories_filepath))
        df = self.load_data()

        print('Cleaning data...')
        df = self.clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(self.database_filepath))
        self.save_data(df)

        print('Cleaned data saved to database!')



def main():

    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        dataProc = DataPreProcessing(messages_filepath, categories_filepath, database_filepath)
        dataProc.run()
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
    