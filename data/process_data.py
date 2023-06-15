# this file is for making a 'database' from 2 csv files
# disaster_categories.csv
# disaster_messages.csv

# these are the stuff you did in >> ETL Pipeline Preparation.ipynb

'''
Sample Script Syntax:

python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite destination db>

cd data
python process_data.py disaster_messages.csv disaster_categories.csv ETL_Preparation.db
'''

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# you're merging 2 'dataframes' into 1
def load_messages_with_categories(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # df = pd.merge(messages, categories,on = 'id')
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')

    return df 


# these are the stuff you did in >> ETL Pipeline Preparation.ipynb
def clean_categories_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe, not inclduing 'row 1'
    row = categories.iloc[[1]]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        # categories[column] = categories[column].str[-1]
        # print(categories[column].str[-1])
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    # df.columns # Index(['id', 'message', 'original', 'genre', 'categories'], dtype='object')
    df = df.drop('categories', axis=1, inplace=False)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)

    # drop duplicates
    df= df.drop_duplicates() #0 duplicates

    return df


def save_data_to_db(df, database_filename):
    # check if file exist, if exist, delete it
    import os

    if os.path.exists(database_filename):
        os.remove(database_filename)

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('ETL_Preparation', engine, index=False)
    # df.to_sql('ETL_Preparation', engine, index=False, if_exists='replace')


# function for making a 'database' from 2 csv files
def main():

    # check that we have 4 'inputs'
    if len(sys.argv) == 4:

        # starting from 'position 1' to everything else
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] 

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
                .format(messages_filepath, categories_filepath))
        
        df = load_messages_with_categories(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_categories_data(df)
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data_to_db(df, database_filepath)
        
        print('Cleaned data has been saved to database!')
        
    else: 
        print("Please provide the arguments correctly: \nSample Script Execution:\n \
        > python process_data.py disaster_messages.csv disaster_categories.csv ETL_Preparation.db \n \
        Arguments Description: \n \
        1) Path to the CSV file containing messages (e.g. disaster_messages.csv)\n \
        2) Path to the CSV file containing categories (e.g. disaster_categories.csv)\n \
        3) Path to SQLite destination database (e.g. disaster_response_db.db)")


if __name__ == '__main__':
    main()