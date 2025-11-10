# Training model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
from prepare_ds import generate_dataset
from utils import DEFAULT_PATH

from joblib import dump, load


def select_method(method='RF'):
    """With teacher: (RF, Bayers...), without teacher (segmentation)"""
    if method == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Method not recognized: {method}")

    print("Selected model / method:", method)

    return model


def save_model(model, out=None):
    out = out if out is not None else "model.joblib"
    out = DEFAULT_PATH['output'] + out
    dump(model, out)
    print("Model was saved: " + out)


def train_model(data, labels, model='RF', save=True):

    print("Split X, y -> X_train, y_train...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=0.2, random_state=42)

    # TODO: develop it.
    print("Start training model...")
    m = select_method(model)
    m.fit(X_train, y_train)
    print("Model was trained. Start validate it...")
    y_pred = m.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if save:
        print("Saving model...")
        n = f"tmp_weights_{model}.joblib"
        save_model(m, n)

    return m

