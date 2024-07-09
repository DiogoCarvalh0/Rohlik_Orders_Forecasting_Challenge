import pandas as pd

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features: list[str]) -> None:
        super().__init__()

        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: None = None):
        self.encoders_ = {
            var: LabelEncoder().fit(X[var]) for var in self.categorical_features
        }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "encoders_")

        _X = X.copy()

        for var, encoder in self.encoders_.items():
            _X[var] = encoder.transform(_X[var])

        return _X
