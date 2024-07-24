import os
import pickle
import re
import sys
import warnings

import nltk
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine

warnings.simplefilter("ignore")


def load_data(db_filepath):
    """
    Load data from a SQLite database into a Pandas DataFrame.

    Args:
        db_filepath (str): The file path to the SQLite database.

    Returns:
        pd.DataFrame: The data loaded from the SQLite database.
    """
    engine = create_engine("sqlite:///" + db_filepath)
    table_name = os.path.basename(db_filepath).replace(".db", "") + "_table"
    db = pd.read_sql_table(table_name, engine)
    return db


def get_xy(db, x_column, y_column):
    """
    Extract feature and target variables from a DataFrame.

    Args:
        db (pd.DataFrame): The DataFrame containing the data.
        x_column (str): The name of the column to be used as the feature
        variable (X).
        y_column (str): The name of the column to be excluded from
        the target variables (y).

    Returns:
        tuple: A tuple containing:
            - X (pd.Series): The feature variable.
            - y (pd.DataFrame): The target variables with y_column dropped.
            - col_names (list): A list of the names of the target columns.
    """
    X = db[x_column]
    y = db.drop(y_column, axis=1)
    col_names = list(y.columns.values)
    print(f"    X shape: {X.shape} \n    y shape: {y.shape}\n")
    return X, y, col_names


def tokenize(text):
    """
    Process and tokenize text for Natural Language Processing (NLP) tasks.

    Args:
        text (str): The input text string to be tokenized.

    Returns:
        list: A list of processed and tokenized words from the input text.
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    words = nltk.word_tokenize(text)

    # remove stop words
    stopwords_ = nltk.corpus.stopwords.words("english")
    words = [word for word in words if word not in stopwords_]

    # extract root form of words
    words = [
        nltk.stem.WordNetLemmatizer().lemmatize(word, pos="v")
        for word in words
    ]

    return words


def build_model():
    """
    Build a machine learning pipeline for text classification.

    This function creates a pipeline that processes text data and trains a
    multi-output classifier using a RandomForestClassifier. The pipeline
    includes text tokenization, vectorization, TF-IDF transformation, and
    classification steps.

    Returns:
        model (Pipeline): A scikit-learn Pipeline object ready for training and
        prediction.
    """
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    (
                                        "vect",
                                        CountVectorizer(tokenizer=tokenize),
                                    ),
                                    ("tfidf", TfidfTransformer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    return model


def get_evaluate_multioutput(actual, predicted, col_names):
    """
    Evaluate a multi-output classification model using various metrics.

    This function calculates accuracy, precision, recall, and F1 score for
    each output label in a multi-output classification task. It then returns
    these metrics in a DataFrame.

    Args:
        actual (numpy.ndarray): Array of actual labels.
        predicted (numpy.ndarray): Array of predicted labels.
        col_names (list of str): List of column names corresponding to the
        output labels.

    Returns:
        metrics_df (pd.DataFrame): DataFrame containing accuracy, precision,
                                   recall, and F1 score for each output label.
    """
    metrics = []

    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(
            actual[:, i], predicted[:, i], average="weighted"
        )
        recall = recall_score(
            actual[:, i], predicted[:, i], average="weighted"
        )
        f1 = f1_score(actual[:, i], predicted[:, i], average="weighted")

        metrics.append([accuracy, precision, recall, f1])

    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(
        data=metrics,
        index=col_names,
        columns=["Accuracy", "Precision", "Recall", "F1"],
    )

    return metrics_df


def get_evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate a multi-output classification model and print the evaluation metrics.

    This function uses a trained model to make predictions on a test dataset,
    evaluates the predictions using various metrics (accuracy, precision,
    recall, F1 score) for each category, and prints the evaluation results.

    Args:
        model: Trained multi-output classification model.
        X_test (pd.DataFrame or np.ndarray): Test dataset features.
        y_test (pd.DataFrame or np.ndarray): Actual labels for the test dataset.
        category_names (list of str): List of category names corresponding to
        the output labels.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    evaluate = get_evaluate_multioutput(
        np.array(y_test), y_pred, category_names
    )
    print(evaluate)


def save_model(model, path):
    """
    Save a trained model to a specified file path using pickle.

    This function serializes a trained machine learning model and saves it to a
    file using Python's pickle module, allowing the model to be loaded and used
    later.

    Args:
        model: Trained machine learning model to be saved.
        path (str): File path where the model will be saved.

    Returns:
        None
    """
    pickle.dump(model, open(path, "wb"))


def main():
    """
    Train and save a machine learning model for text classification.

    This function performs the following steps:
    1. Loads data from a SQLite database.
    2. Extracts features (X) and target variables (y) from the dataset.
    3. Splits the data into training and test sets.
    4. Builds a machine learning pipeline model.
    5. Trains the model on the training data.
    6. Evaluates the model on the test data.
    7. Saves the trained model to a specified file path.

    The function expects two command-line arguments:
    1. The file path of the SQLite database containing the data.
    2. The file path where the trained model should be saved.

    Example:
    python src/train_classifier.py data/DisasterResponse.db classifier_model.pkl
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        # Load database
        print(f"Loading data...\n    DATABASE: {database_filepath}\n")
        db = load_data(database_filepath)

        # Extract X and y variables
        print("Extract X and y variables from the DATABASE...")
        y_drop = ["id", "message", "original", "genre"]
        X, y, category_names = get_xy(db, "message", y_drop)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        print("Building model...\n")
        model = build_model()

        print("Training model...\n")
        model_fitted = model.fit(X_train, y_train)

        print("Evaluating model...")
        get_evaluate_model(model_fitted, X_test, y_test, category_names)

        print(f"\nSaving model...\n    MODEL: {model_filepath}\n")
        save_model(model, model_filepath)

        print("Trained model saved!\n")

    else:
        print(
            "Please provide the arguments correctly\n\n"
            "Arguments Description: \n"
            "- first argument is the database filepath,\n"
            "- second argument is the filepath to save the model"
            "\n\nExample: python src/train_classifier.py"
            "data/DisasterResponse.db classifier_model.pkl\n"
        )


if __name__ == "__main__":
    main()
