import sys
import os
import time
# import libraries
import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import re

# Natural Langauge Toolkit
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize


from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

# mitigate target/label imbalance - coud'nt use this, not designed for multi-class imbalance.
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import SMOTE

# models to consider for model building phase
models = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(verbose=True, class_weight='balanced'),
            "hyperparameters":
                {
                    "classifier__estimator__solver": ["newton-cg", "lbfgs", "liblinear"]
                }
        },
        {
            "name": "AdaBoostClassifier",
            "estimator": AdaBoostClassifier(),
            "hyperparameters":
                {
                    'classifier__estimator__learning_rate': [0.01, 0.02, 0.05],
                    'classifier__estimator__n_estimators': [20, 50, 80]
                }
        },
        {
            "name": "RandomForestClassifier",
            "estimator": RandomForestClassifier(random_state=1, verbose=True, class_weight='balanced'),
            "hyperparameters":
                {
                    "classifier__estimator__n_estimators": [20],
                    "classifier__estimator__criterion": ["entropy", "gini"],
                    "classifier__estimator__max_depth": [2, 5, 10],
                    "classifier__estimator__max_features": ["log2", "sqrt"],
                    "classifier__estimator__max_features": [1, 5, 8],
                    "classifier__estimator__min_samples_split": [2, 3, 5]

                }
        }
    ]

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence)) # using our defined token() above after pos.
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    Parameters
    -----------
    data_base_filepath: str
        Input sql database path to be loaded.
    """
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Disasters', con=engine) # is table always called this? 
    
    def replace_with_majority(row):
        if row == 2.0:
            return 1.0
        return row

    df['related'] = df['related'].apply(replace_with_majority)
    # the 'child_alone' feature has all zeroes, remove it.
    df.drop(['child_alone'], axis=1, inplace=True)
    
    X = df['message']
    y = df.iloc[:,4:]
    labels = y.columns.values
    return X, y, labels


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Parameters
    -----------
    models: List
        List of dictionaries for models for model building.
    tune_hyperparameters: Boolean
        Flag for GridSearch hyperparameter tuning.
    """
    tune_hyperparameters=False
    pipes, grids = [], []
    # create a pipeline + GridSearch obj for each model.
    for model in models:
        # import pdb; pdb.set_trace()
        pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()) # adding addtional features.
                ])),

                ('starting_verb', StartingVerbExtractor())
            ])),

            ('classifier', MultiOutputClassifier(model['estimator']))
        ])
        # --------------------------------------------
        # Hyperameter tuning
        # --------------------------------------------
        cv = GridSearchCV(pipeline, param_grid=model['hyperparameters'], scoring='f1_micro', n_jobs=-1)
        grids.append(cv)
        pipes.append(pipeline)
    # return grids if tune_hyperparameters else pipes
    return pipes[1] # adaboost did the best


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Parameters
    -----------
    
    """
    # Get results and add them to a dataframe.
    # import pdb; pdb.set_trace()
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    Parameters
    -----------
    model: sklearn model obj
        Model to pickle and save.
    model_filepath: str
        Path to save pickle.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start_time = time.time()
        print("="*10, "Started timer for training", "="*10)
        model.fit(X_train, Y_train)
        print(f"Completed training in: {(time.time() - start_time) / 60:.2f} mins.")
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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