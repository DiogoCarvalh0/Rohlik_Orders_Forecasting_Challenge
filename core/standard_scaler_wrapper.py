from typing import Union
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler


class StandardScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features: Union[list[str], None] = None) -> None:
        super().__init__()

        self.features = features

    def fit(self, X: pd.DataFrame, y: None = None):
        self.features = self.features if self.features is not None else X.columns

        self.encoders_ = {var: StandardScaler().fit(X[[var]]) for var in self.features}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "encoders_")

        _X = X.copy()

        for var, encoder in self.encoders_.items():
            _X[var] = encoder.transform(_X[[var]])

        return _X
