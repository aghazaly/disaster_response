import sys
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

def load_data(database_filepath):
    """
    Loading data from a database file
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns.values
    return X, y, category_names

def tokenize(text):
    """
    Takes a string and returns a tokenized list
    """

    text = re.sub("[^a-zA-Z0-9]", " ", text) #retain alphanumeric only
    tokens = word_tokenize(text) #like split but it takes care of punctuation, hasthags, tweethandlers
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]#reduce words to their source (plurals)
    tokens = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens]#reduce words to their source (verbs)

    return tokens


def build_model():
    """
    Returns trained model given a pipeline of transformers and estimators
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [40, 50]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=6, verbose=100)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Printing Accuracy and Best Params
    """

    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """
    Saving trained model to a pickle file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: '\
              'python train_classifier.py ../data/DisasterResponse.db classifier.pickle ')


if __name__ == '__main__':
    main()
