# Disaster Response Project

This project involves developing an ETL pipeline integrated with supervised learning and grid search to classify text messages sent during disaster events. 

It includes an ML pipeline and a web app designed to categorize disaster response messages in real time using Natural Language Processing (NLP) techniques.

## Results


| home page                          | results page                            |
| ---------------------------------- | --------------------------------------- |
| ![](assets/Disasters_graphics.jpg) | ![](assets/Disasters_model_results.jpg) |

## Project Setup

### Clone this repository

```shell
(base)$: git clone git@github.com:mafda/disaster_response_project.git
(base)$: cd disaster_response_project
```

### Configure environment

- Create the conda environment

    ```shell
    (base)$: conda env create -f environment.yml
    ```

- Activate the environment

    ```shell
    (base)$: conda activate disaster_response
    ```

- Download the dataset and model from [this
  link](https://drive.google.com/drive/folders/1uNqCHmE__m9tEV-pgvkBcu4iJFfkaSbN?usp=share_link),
  create `data` folder and copy the `categories.csv` and `messages.csv` datasets
  here.

    ```shell
    (disaster_response)$: mkdir data
    ```

## Project Structure

```shell
.
├── README.md
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db
│   ├── categories.csv
│   └── messages.csv
├── environment.yml
├── models
│   └── classifier_model.pkl
├── notebooks
│   ├── ETL_Pipeline_Preparation.ipynb
│   └── ML_Pipeline_Preparation.ipynb
└── src
    ├── process_data.py
    └── train_classifier.py
```

## Project Components

**1. ETL Pipeline**

File `src/process_data.py`, contains a data cleaning pipeline that:
* Loads the `messages.csv` and `categoriescsv` datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

**2. ML Pipeline**

File `src/train_classifier.py`, contains a machine learning pipeline that:
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

**3. Flask Web App**

File `app/run.py` will start the web app where users can enter their query and the machine learning model will categorize this event.
* Show data visualizations using Plotly in the web app.

## Executing Program

There are 3 steps to follow, from cleaning the data, through training the model, and ending with the web app.

### 1. Cleaning data

In the project directory, run:

```shell
(disaster_response)$: python src/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
```

### 2. Training model

After the data cleaning process, in the project directory, run:

```shell
(disaster_response)$: python src/train_classifier.py data/DisasterResponse.db models/classifier_model.pkl
```

### 3. Web App

Go the `app` directory and run:

```shell
(disaster_response)$: cd app
(disaster_response) app $: python app/run.py
```

And go to http://localhost:3000

## References

- [Data Scientist Nanodegree
  Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)

---

made with 💙 by [mafda](https://mafda.github.io/)
