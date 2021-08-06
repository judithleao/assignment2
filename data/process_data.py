'''
>> Potentially need to install <<
pip install plotly_express
'''

# Import relevant libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
from langdetect import detect

################################################################
############################ FUNCTIONS #########################
################################################################

def load_data(messages_filepath, categories_filepath):    
    '''
    INPUT: 
        messages_filepath as path to messages.csv.
        categories_filepath as path to categories.csv.
    OUTPUT: dataframe that combines both messages.csv and categories.csv.
    PURPOSE: Merge datasets and return dataframe.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def extract_language(snippet):    
    '''
    INPUT: snippet (string) as instance of 'original' message from df.
    OUTPUT: language (string) defining language of snippet
    PURPOSE: Define language for an original version of a message
    '''
    try:
        language = detect(snippet)
    except:
        language = 'n/a'
    return language
    
def clean_data(df):
    '''
    INPUT: dataframe with all data.
    OUTPUT: 
        dataframe with all data.
        category_names (dataframe) contains labels for dependent variables.
    PURPOSE: Expand categories and clean data (missing, duplicates, recodes, add language column).
    '''
    # Create one numeric column per answer category
    ## Split column into multiple columns at separator ;
    categories = df['categories'].str.split(';', expand=True)
    ## Set clean column names
    row = categories.iloc[0]
    category_colnames = [colname[:-2] for colname in row]
    categories.columns = category_colnames
    ## Clean cells and turn to numeric
    for column in categories:    
        # Source: Stackoverflow
        # Question by: prp. Profile: https://stackoverflow.com/users/9153261/prp
        # Answer by: jezrael. Profile: https://stackoverflow.com/users/2901002/jezrael
        # https://stackoverflow.com/questions/52850192/python-extract-last-letter-of-a-string-from-a-pandas-column
        categories[column] = categories[column].str.strip().str[-1]
        categories[column] = pd.to_numeric(categories[column]) 
    ## Remove if label has only one class, recode column if more than 2 classes where any code >1 
    removal_list = []
    reduction_list = []
    for col in category_colnames:
        if categories[col].nunique() == 1:
            removal_list.append(col)
        if categories[col].nunique() > 2:
            reduction_list.append(col)      
    categories = categories.drop(columns=removal_list)
    for col in reduction_list:
        categories.loc[categories[col] > 1, col] = 0
    ## Save the category_names
    category_names = list(categories.columns)
    category_names = pd.DataFrame(category_names)
    
    # Drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Missings in x-variable: Drop if no message
    df = df.dropna(axis=0, subset=['message'])
    
    # Add the language for each original snippet
    # Source: StackOverflow
    # Question by Michael. Profile: https://stackoverflow.com/users/2327821/michael
    # https://stackoverflow.com/questions/19914937/applying-function-with-multiple-arguments-to-create-a-new-pandas-column

    df['language'] = df['original'].apply(extract_language)
    
    return df, category_names
           
       
def save_data(df, category_names, database_filename):
    '''
    INPUT: 
        dataframe with all data.
        category_names (dataframe) contains labels for dependent variables.
        database_filename as path to database.
    OUTPUT: None
    PURPOSE: Save files to database.
    '''
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('Messages_Table', engine, index=False, if_exists='replace') 
    category_names.to_sql('Categories_Table', engine, index=False, if_exists='replace') 


def main():
    '''
    INPUT: None
    OUTPUT: None
    PURPOSE: Calls other functions in logical order to clean data and save to database.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df, category_names = clean_data(df)
                
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, category_names, database_filepath)
        
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