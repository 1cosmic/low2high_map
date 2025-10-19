# Training model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
from prepare_ds import preparation_dataset


def select_method(method='RF'):
    """With teacher: (RF, Bayers...), without teacher (segmentation)"""
    if method == 'RF':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Method not recognized: {method}")

    print("Selected model / method:", method)

    return model


def train_model(data=None, labels=None, model='RF'):

    if data is None or labels is None:
        print("Loading and preparing dataset...")
        data, labels = preparation_dataset(mask_mode='random', resize='by_label', percent=0.01)


    X_train, X_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=0.2, random_state=42)

    # TODO: develop it.
    model = select_method(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))