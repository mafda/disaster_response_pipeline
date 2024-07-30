"""
Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

Process Data: python process_data.py <path to messages csv file> <path to
categories csv file> <path to sqllite  destination db>

Arguments Description:
    1. File path to the messages CSV file.
    2. File path to the categories CSV file.
    3. File path to the SQLite database to save the cleaned data.
"""

import os
import re
import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(data):
    """
    Loads a CSV file into a pandas DataFrame and prints its shape.

    Args:
        data (str): Path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(data)
    print(f"Load {data} with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def merge_data(df1, df2, column):
    """
    Merges two pandas DataFrames on a specified column and prints the shape of
    the resulting DataFrame.

    Args:
        df1 (pd.DataFrame): The first DataFrame to be merged. df2
        (pd.DataFrame): The second DataFrame to be merged. column (str): The
        column name on which to merge the DataFrames.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    df = pd.merge(df1, df2, on=column)
    print(f"Merge data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_data(df, column):
    """
    Cleans the data in the specified DataFrame by splitting the specified
    column into individual category columns, converting values to numeric,
    and removing duplicates.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be cleaned.
        column (str): The name of the column to be split into individual
        categories.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # create a dataframe of the 36 individual category columns
    categories = df[column].str.split(pat=";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    print(f"{len(category_colnames)} categories")

    # rename the columns of `categories`
    categories.columns = category_colnames

    for category in categories:
        # set each value to be the last character of the string
        categories[category] = categories[category].astype(str).str[-1:]

        # convert category from string to numeric
        categories[category] = categories[category].astype(int)

    # drop not binary columns
    categories = categories.drop(categories[categories['related'] == 2].index)

    # drop the original categories column from `df`
    df.drop([column], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join="inner", axis=1)

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Drop {df.duplicated().sum()} duplicated rows")
        # drop duplicates
        df.drop_duplicates(inplace=True)

    print(f"Clean data with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def check_unique_zero_columns(df):
    """
    Checks for columns in the DataFrame that have unique values all equal to
    zero, and drops these columns from the DataFrame if found.

    Args:
        df (pd.DataFrame): The input DataFrame to check.

    Returns:
        pd.DataFrame: The DataFrame with zero-value columns dropped, if any were
        found.
    """
    zero_columns = []
    for column in df.columns:
        if df[column].nunique() == 1 and df[column].unique()[0] == 0:
            zero_columns.append(column)

    if len(zero_columns) > 0:
        print(
            f"{len(zero_columns)} columns have unique values all equal to zero"
        )
        df = df.drop(zero_columns, axis=1)
        print(f"Drop {len(zero_columns)} columns")

    else:
        print(
            f"{len(zero_columns)} columns have unique values all equal to zero"
        )
    return df


def replace_urls(message, replacement="urlplaceholder"):
    """
    Replace URLs in a given message with a specified replacement string.

    Args:
        message (str): The input message containing URLs to be replaced.
        replacement (str, optional): The string to replace URLs with. Default is
        'urlplaceholder'.

    Returns:
        str: The message with URLs replaced by the specified replacement string.
    """
    url_pattern = r"(https?://\S+|www\.\S+)"
    return re.sub(url_pattern, replacement, message)


def save_sqlite(df, filepath):
    """
    Saves the given DataFrame to an SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        filepath (str): The file path where the SQLite database will be created.

    Returns:
        None
    """
    engine = create_engine("sqlite:///" + filepath)
    table_name = os.path.basename(filepath).replace(".db", "") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists="replace")


def main():
    """
    Main function to load data, clean it, and save it to an SQLite database.

    It expects three arguments:
    1. File path to the messages CSV file.
    2. File path to the categories CSV file.
    3. File path to the SQLite database to save the cleaned data.

    Example usage:
    python process_data.py data/messages.csv data/categories.csv
        data/DisasterResponse.db
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[
            1:
        ]

        # Loads the messages and categories datasets
        print(
            "Loading data...\n"
            f"    MESSAGES: {messages_filepath}\n"
            f"    CATEGORIES: {categories_filepath}\n"
        )

        messages = load_data(messages_filepath)
        categories = load_data(categories_filepath)

        # Merge datasets
        print("\nMerging data...")
        df = merge_data(messages, categories, "id")

        # Cleans the data
        print("\nCleaning data...")

        # Split `categories` into separate category columns.
        df = clean_data(df, "categories")

        # Drop columns with unique values equal to zero
        df = check_unique_zero_columns(df)

        # Replace URLs in a given message
        print("Search and replace URLs in a column message")
        df["message"] = df["message"].apply(replace_urls)

        # Stores it in a SQLite database
        print("\nSaving data...")
        save_sqlite(df, database_filepath)

        print("\nCleaned data saved to database!\n")

    else:
        print(
            "Please provide the arguments correctly\n\n"
            "Arguments Description: \n"
            "- first and second arguments are the datasets,\n"
            "- third argument is the filepath of the database to save the"
            "cleaned data."
            "\n\nExample: python process_data.py "
            "src/process_data.py data/messages.csv data/categories.csv"
            "data/DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
