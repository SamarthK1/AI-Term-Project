from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC


"""
Script contains all the models to and return their predictions or loss in order to evaluate the results in Overall Results Notebook.

These are the optimal models for each Machine Learning Algorithm, for detailed information check each of algorithms notebook.
"""


# Multi-Layer-Perceptron
def mlp():
    """Multi-Layer Perceptron Algorithm Implementation

    Returns:
        y_pred: predictions
    """

    # Load training data
    train_data = pd.read_csv("train_data.csv")
    X_train_raw = train_data["X_train"].tolist()
    y_train_raw = train_data["y_train"].tolist()

    # Load test data
    test_data = pd.read_csv("test_data.csv")
    X_test_raw = test_data["X_test"].tolist()
    y_test_raw = test_data["y_test"].tolist()

    # Make it a list
    X_train_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item
        for item in X_train_raw
    ]
    X_test_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item for item in X_test_raw
    ]

    # Join into one big string
    corpus_train = [" ".join(item) for item in X_train_raw]
    corpus_test = [" ".join(item) for item in X_test_raw]

    # Initialize the CountVectorizer, limiting max_features to the top 1000 words.
    vectorizer = CountVectorizer(max_features=1000)

    # Fit the vectorizer on train and test data. Transform the data into BoW matrices.
    X_train = vectorizer.fit_transform(corpus_train).toarray()
    X_test = vectorizer.transform(corpus_test).toarray()

    # Encode the taget variables
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.fit_transform(y_test_raw)

    # Build the Structure of the Neural Network
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(len(set(y_train)), activation="softmax"),
        ]
    )

    # Compile and Train the Model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        X_train, y_train, epochs=10, batch_size=32, validation_split=0.2
    )

    # Make the Predictions
    y_pred = model.predict(X_test, batch_size=32)

    return y_pred, history


def svm():
    """SVM algorithm implementation.

    Returns:
        y_pred: predictions
    """

    # Load training data
    train_data = pd.read_csv("train_data.csv")
    X_train_raw = train_data["X_train"].tolist()
    y_train_raw = train_data["y_train"].tolist()

    # Load test data
    test_data = pd.read_csv("test_data.csv")
    X_test_raw = test_data["X_test"].tolist()
    y_test_raw = test_data["y_test"].tolist()

    # Subsets for faster training
    X_train_subset = X_train_raw[:10000]
    y_train_subset = y_train_raw[:10000]

    X_test_subset = X_test_raw[:2000]
    y_test_subset = y_test_raw[:2000]

    # Make the subset a list
    X_train_subset = [
        ast.literal_eval(item) if isinstance(item, str) else item
        for item in X_train_subset
    ]
    X_test_subset = [
        ast.literal_eval(item) if isinstance(item, str) else item
        for item in X_test_subset
    ]

    # Subset into a string
    corpus_train_subset = [" ".join(item) for item in X_train_subset]
    corpus_test_subset = [" ".join(item) for item in X_test_subset]

    # Initialize the CountVectorizer, limiting max_features to the top 500 words.
    vectorizer = CountVectorizer(max_features=500)

    # Vectorize the Subset
    X_train_subset = vectorizer.fit_transform(corpus_train_subset).toarray()
    X_test_subset = vectorizer.transform(corpus_test_subset).toarray()

    # Encode the taget variables
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.fit_transform(y_test_raw)

    y_train_subset = encoder.fit_transform(y_train_subset)
    y_test_subset = encoder.fit_transform(y_test_subset)

    # Optimized Hyperparameter
    model = SVC(class_weight="balanced", C=0.01, gamma="scale", kernel="linear")
    model.fit(X_train_subset, y_train_subset)

    y_pred = model.predict(X_test_subset)

    return y_pred


# Logistic Regression
def logistic_regression():
    """Logistic Regression Algorithm Implementation

    Returns:
        y_pred: predictions
    """

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

    return y_pred


def naive_bayes():
    # Load training data
    train_data = pd.read_csv("train_data.csv")
    X_train = train_data["X_train"].tolist()
    y_train = train_data["y_train"].tolist()

    # Load test data
    test_data = pd.read_csv("test_data.csv")
    X_test = test_data["X_test"].tolist()
    y_test = test_data["y_test"].tolist()

    # Tokenization
    X_train_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item for item in X_train
    ]
    X_test_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item for item in X_test
    ]

    # Convert tokenized texts to strings for CountVectorizer
    X_train_texts = [" ".join(tokens) for tokens in X_train_raw]
    X_test_texts = [" ".join(tokens) for tokens in X_test_raw]

    # Create CountVectorizer and limit vocabulary to the top 1000 words
    vectorizer = CountVectorizer(max_features=1000)
    X_train_bow = vectorizer.fit_transform(X_train_texts)
    X_test_bow = vectorizer.transform(X_test_texts)

    class NaiveBayesClassifier:
        def __init__(self):
            self.class_counts = defaultdict(int)
            self.word_counts = defaultdict(lambda: defaultdict(int))
            self.vocab = set()

        def train(self, X, y):
            for i in range(len(X)):
                tokens = X[i]
                emotion = y[i]
                self.class_counts[emotion] += 1
                for token in tokens:
                    self.word_counts[emotion][token] += 1
                    self.vocab.add(token)

        def predict(self, X):
            predictions = []
            for i, tokens in enumerate(X):
                emotion_log_probs = []
                for emotion in self.class_counts:
                    log_prob = np.log(
                        self.class_counts[emotion] / sum(self.class_counts.values())
                    )
                    denominator = sum(self.word_counts[emotion].values()) + len(
                        self.vocab
                    )
                    log_prob += sum(
                        np.log((self.word_counts[emotion][token] + 1) / denominator)
                        for token in tokens
                    )
                    emotion_log_probs.append(log_prob)
                max_idx = np.argmax(emotion_log_probs)
                pred_emotion = list(self.class_counts.keys())[max_idx]
                predictions.append(pred_emotion)
            return predictions

    # Create Naive Bayes Classifier instance
    classifier = NaiveBayesClassifier()

    # Train the classifier
    classifier.train(X_train_raw, y_train)

    # Make predictions on the testing data
    predictions = classifier.predict(X_test_raw)

    return predictions


def decision_tree():

    # Load training data
    train_data = pd.read_csv("train_data.csv")
    X_train = train_data["X_train"].tolist()
    y_train = train_data["y_train"].tolist()

    # Load test data
    test_data = pd.read_csv("test_data.csv")
    X_test = test_data["X_test"].tolist()
    y_test = test_data["y_test"].tolist()

    # Tokenization
    X_train_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item for item in X_train
    ]
    X_test_raw = [
        ast.literal_eval(item) if isinstance(item, str) else item for item in X_test
    ]

    # Convert tokenized texts to strings for CountVectorizer
    X_train_texts = [" ".join(tokens) for tokens in X_train_raw]
    X_test_texts = [" ".join(tokens) for tokens in X_test_raw]

    # Create CountVectorizer and limit vocabulary to the top 1000 words
    vectorizer = CountVectorizer(max_features=1000)
    X_train_bow = vectorizer.fit_transform(X_train_texts)
    X_test_bow = vectorizer.transform(X_test_texts)

    class DecisionTreeClassifier:
        def __init__(self, max_depth=None):
            self.tree = None
            self.max_depth = max_depth

        def train(self, X, y):
            self.tree = self.build_tree(X, y)

        def build_tree(self, X, y, depth=0):
            # Check for base cases
            if depth == self.max_depth or len(np.unique(y)) == 1:
                return np.argmax(np.bincount(y))

            num_samples, num_features = X.shape
            best_feature = None
            best_gain = -1

            # Calculate the information gain for each feature
            for feature_idx in range(num_features):
                values = np.unique(X[:, feature_idx])
                for value in values:
                    left_indices = np.where(X[:, feature_idx] == value)[0].astype(
                        np.int64
                    )
                    right_indices = np.where(X[:, feature_idx] != value)[0].astype(
                        np.int64
                    )

                    left_labels = np.array(y)[left_indices]
                    right_labels = np.array(y)[right_indices]
                    gain = self.information_gain(y, left_labels, right_labels)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = (feature_idx, value)

            if best_feature is None:
                return np.argmax(np.bincount(y))

            feature_idx, value = best_feature
            left_indices = np.where(X[:, feature_idx] == value)[0].astype(np.int64)
            right_indices = np.where(X[:, feature_idx] != value)[0].astype(int)

            left_mask = np.isin(np.arange(X.shape[0]), left_indices)
            right_mask = np.isin(np.arange(X.shape[0]), right_indices)

            left_tree = self.build_tree(
                X[np.nonzero(left_mask)[0]], y[np.nonzero(left_mask)[0]], depth + 1
            )
            right_tree = self.build_tree(
                X[np.nonzero(right_mask)[0]], y[np.nonzero(right_mask)[0]], depth + 1
            )

            return {
                "feature_idx": feature_idx,
                "value": value,
                "left": left_tree,
                "right": right_tree,
            }

        def information_gain(self, parent_labels, left_labels, right_labels):
            parent_entropy = self.calculate_entropy(parent_labels)
            left_entropy = self.calculate_entropy(left_labels)
            right_entropy = self.calculate_entropy(right_labels)

            num_parent = len(parent_labels)
            num_left = len(left_labels)
            num_right = len(right_labels)

            weighted_entropy = (num_left / num_parent) * left_entropy + (
                num_right / num_parent
            ) * right_entropy

            return parent_entropy - weighted_entropy

        def calculate_entropy(self, labels):
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            probabilities = label_counts / len(labels)
            entropy = -np.sum(probabilities * np.log2(probabilities))

            return entropy

        def predict(self, X):
            predictions = []
            for instance in X:
                prediction = self.traverse_tree(instance, self.tree)
                predictions.append(prediction)

            return predictions

        def traverse_tree(self, instance, tree):
            if isinstance(tree, dict):
                feature_idx = tree["feature_idx"]
                value = tree["value"]

                if instance[feature_idx] == value:
                    return self.traverse_tree(instance, tree["left"])
                else:
                    return self.traverse_tree(instance, tree["right"])
            else:
                return tree

    # Create Decision Tree Classifier instance
    classifier = DecisionTreeClassifier(max_depth=3)

    # Train the classifier
    classifier.train(X_train_bow.toarray(), np.array(y_train))

    # Make predictions on the testing data
    predictions = classifier.predict(X_test_bow.toarray())

    return predictions
