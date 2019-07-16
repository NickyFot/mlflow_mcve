from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model

from package import transform


import mlflow


EXPERIMENT_ID = "3161289561806886"


def workflow():
    with mlflow.start_run(experiment_id=EXPERIMENT_ID) as run:
        digits = datasets.load_digits()
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            digits.data,
            digits.target,
            test_size=0.25,
            random_state=0
        )
        logistic_regr = linear_model.LogisticRegression()
        logistic_regr.fit(x_train, y_train)
        transform.log_model(logistic_regr, 'package')


if __name__ == '__main__':
    workflow()