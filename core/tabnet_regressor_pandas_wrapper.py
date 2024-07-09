from typing import Union
import joblib
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor


class TabNetRegressorPandasWrapper(TabNetRegressor):

    def __init__(self, categorical_features: list[str], **tabnet_parameters) -> None:
        self.categorical_features = categorical_features
        self.tabnet_params = tabnet_parameters

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        load_model_path: Union[str, None] = None,
        **fit_params
    ):
        if load_model_path:
            self.model = joblib.load(load_model_path)
            fit_params["warm_start"] = True
        else:
            self.model = self.__define_model(X)

        self.model.fit(X.values, y.values.reshape(-1, 1), **fit_params)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values).flatten()

    def __define_model(self, X: pd.DataFrame) -> TabNetRegressor:
        cat_idxs = [
            i for i, f in enumerate(X.columns) if f in self.categorical_features
        ]
        cat_dims = [X[feature].nunique() for feature in self.categorical_features]

        self.tabnet_params["cat_idxs"] = cat_idxs
        self.tabnet_params["cat_dims"] = cat_dims

        return TabNetRegressor(**self.tabnet_params)
