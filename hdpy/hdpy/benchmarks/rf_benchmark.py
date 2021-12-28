import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("--input-data", help="path to training dataset")

parser.add_argument("--train-data", help="path to training dataset")
parser.add_argument("--test-data", help="path to test dataset")
args = parser.parse_args()


def train(model, x_train, y_train):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    rf_random = RandomizedSearchCV(
        estimator=model,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    rf_random.fit(x_train, y_train)

    return rf_random.best_estimator_


def test(model, x_test):

    preds = model.predict(x_test)
    return preds


def load_data(data_file):
    with open(data_file, "rb") as handle:
        data = pickle.load(handle)

    return data

def main(): 

    x_train, y_train = load_data(args.train_data)
    x_test, y_test = load_data(args.test_data)

    # need to mask out the labels of 2 which mean ambiguous and make our problem harder

    #x_train = x_train[y_train != 2]
    #y_train = y_train[y_train != 2]
    #x_test = x_test[y_test != 2]
    #y_test = y_test[y_test != 2]


    model = RandomForestClassifier()

    best_model = train(model, x_train, y_train)
    y_pred = test(best_model, x_test)
    y_pred_proba = best_model.predict_proba(x_test)

    from sklearn.metrics import classification_report, roc_auc_score

    print(classification_report(y_true=y_test, y_pred=y_pred))

    print(roc_auc_score(y_true=y_test, y_score=y_pred_proba[:,1]))

if __name__ == "__main__":

    main()
