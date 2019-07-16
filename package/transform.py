import os
import pandas as pd

from mlflow.utils.file_utils import TempDir
from mlflow import pyfunc
from mlflow import sklearn
import mlflow


def log_model(model, path):
    with TempDir() as tmp:
        data_path = tmp.path('model')
        os.mkdir(data_path)
        xg_path = os.path.join(data_path, 'model')
        mlflow.sklearn.save_model(
            model,
            path=xg_path
        )
        mlflow.pyfunc.log_model(
            artifact_path=path,
            data_path=data_path,
            conda_env='environment.yml',
            code_path=['package/'],
            loader_module=__name__
        )


def _load_pyfunc(path):
    model_path = os.path.join(path, 'model')
    model = mlflow.pyfunc.load_model(model_path)
    return ClassifierWrapperPyfunc(model)


class ClassifierWrapperPyfunc(object):
    def __init__(self, model):
        self._model = model

    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict(input)
