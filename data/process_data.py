import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    Load, merge and return the data given. 

    Args:
    messages_filepath: The path of the csv file that holds the messages.
    categories_filepath: The path of the csv file that holds the categories of the messages. 

    Returns:
    The data frame that holds the merged data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='inner')
    return df


def clean_data(df):
    """
    This function cleans the given dataframe.
    The message categories are split into columns, and
    the doublicates are dropped.

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



def save_data(df, database_filename):
    """
    This function saves the given dataframe as table with
    name "Classified_Messages" in the given database file.

    Args:
    df: The dataframe to be saved.
    database_filename: The database file name, 
    in which the dataframe shall be saved.

    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Classified_Messages', engine, index = False, if_exists = 'replace')


def main():
    """
    The main function.

    Args:
    None

    Returns:
    None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()