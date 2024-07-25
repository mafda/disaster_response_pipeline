"""
Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

Run the web application.
"""

import json
import os
import re
from collections import Counter

import joblib
import nltk
import pandas as pd
import plotly
from flask import Flask, render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

app = Flask(__name__)


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


# filepaths
database_filepath = "../data/DisasterResponse.db"
model_filepath = "../models/classifier_model.pkl"

# load data
engine = create_engine("sqlite:///" + database_filepath)
table_name = os.path.basename(database_filepath).replace(".db", "") + "_table"
df = pd.read_sql_table(table_name, engine)

# load model
model = joblib.load(model_filepath)

# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # Distribution of Message Genres
    genre_counts = (
        df.groupby("genre").count()["message"].sort_values(ascending=False)
    )
    genre_names = list(genre_counts.index)

    # Distribution of Message Categories
    category_names = df.iloc[:, 4:].columns
    category_boolean = (df.iloc[:, 4:] != 0).sum()
    category_sorted = category_boolean.sort_values(ascending=False)

    # graph - Words frequency
    all_words = []
    for text in df["message"]:
        all_words.extend(tokenize(text))
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(10)
    top_words_df = pd.DataFrame(top_words, columns=["word", "count"])

    # create visuals
    graphs = [
        # graph - Distribution of Message Genres
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        # graph - Distribution of Message Categories
        {
            "data": [Bar(x=category_names, y=category_sorted)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category", "tickangle": 35},
            },
        },
        # graph - Words frequency
        {
            "data": [Bar(x=top_words_df["word"], y=top_words_df["count"])],
            "layout": {
                "title": "Distribution of Most Frequency Words",
                "yaxis": {"title": "Frequency"},
                "xaxis": {"title": "Words", "tickangle": 35},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    """
    Run the web application.

    This function starts the web server for the application, making it
    accessible on all network interfaces (host="0.0.0.0") and listening on port
    3000. The server will be run in debug mode, which provides detailed error
    messages and auto-reload functionality for development purposes.
    """
    app.run(host="0.0.0.0", port=3000, debug=True)


if __name__ == "__main__":
    main()
