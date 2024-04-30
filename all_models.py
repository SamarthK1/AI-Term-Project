from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import pandas as pd
import ast

"""
Script contains all the models to and return their predictions or loss in order to evaluate the results in Overall Results Notebook.

These are the optimal models for each Machine Learning Algorithm, for detailed information check each of algorithms notebook.
"""


# Multi-Layer-Perceptron
def mlp():
    return


def svm():
    return None


# Logistic Regression
def logistic_regression():

    # Load the train and test data (already preprocessed).
    print("Logistic Regression: Loading data")
    train_data = pd.read_csv("train_data.csv")
    X_train_raw = train_data["X_train"].tolist()
    y_train = train_data["y_train"].tolist()
    test_data = pd.read_csv("test_data.csv")
    X_test_raw = test_data["X_test"].tolist()
    y_test_raw = test_data["y_test"].tolist()

    # Format the X values into lists.
    X_train_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item
        for item in X_train_raw
    ]
    X_test_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item for item in X_test_raw
    ]

    # Utilize the Bag of Words approach using a CountVectorizer.
    # Convert the train and test data into strings for the CountVectorizer.
    train_str = []
    test_str = []

    for text in X_train_raw:
        train_str.append(" ".join(text))

    for text in X_test_raw:
        test_str.append(" ".join(text))

    # Initialize the CountVectorizer, limiting max_features to the top 1000 words.
    vectorizer = CountVectorizer(max_features=1000)

    # Fit the vectorizer on train and test data. Transform the data into BoW matrices.
    X_train = vectorizer.fit_transform(train_str).toarray()
    test_bow = vectorizer.transform(test_str).toarray()

    print("Logistic Regression: Initializing model")
    # Initialize the optimal Logistic Regression model using the following hyperparameters.
    # Using max_iter = 500, C = 100, penalty = l1, and solver = liblinear.
    logReg = LogisticRegression(max_iter=500, C=100, penalty="l1", solver="liblinear")
    logReg.fit(X_train, y_train)

    print("Logistic Regression: Making predictions")
    y_pred = logReg.predict(test_bow)

    # Losses
    loss_model = SGDClassifier(
        loss="log",
        learning_rate="constant",
        eta0=0.01,
        max_iter=1,
        warm_start=True,
        penalty=None,
    )
    losses = []

    print("Logistic Regression: Calculating losses")
    # Train the model for 10 epochs.
    for _ in range(10):
        loss_model.partial_fit(X_train, y_train, classes=np.unique(y_train))
        probabilities = loss_model.predict_proba(X_train)
        loss = log_loss(y_train, probabilities)
        losses.append(loss)

    print(
        f"Accuracy of the Logistic Regression Model: {accuracy_score(y_test_raw, y_pred):.4f}"
    )

    # Return the predictions and losses.
    return y_pred, losses


def naive_bayes():
    return None


def decision_tree():
    return None
