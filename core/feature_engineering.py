import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FE(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        _X = X.copy()

        _X["month"] = _X["date"].dt.month
        _X["week"] = _X["date"].dt.isocalendar().week

        self.week_information = self.__get_week_information(_X)
        self.month_information = self.__get_month_information(_X)

        self.is_fit_ = True

        return self

    def transform(self, X):
        check_is_fitted(self, "is_fit_")

        X = self.__get_date_features(X)

        X = X.merge(self.week_information, on=["warehouse", "week"], how="left")
        X = X.merge(self.month_information, on=["warehouse", "month"], how="left")

        X = self.__add_special_dates(X)
        X = self.__variable_interaction(X)
        X = self.__get_next_holiday_information(X)
        X = self.__get_past_holiday_information(X)

        return X

    def __get_month_information(self, X):
        cols = ["snow", "precipitation"]
        rename = {col: f"monthly_avg_{col}" for col in cols}

        month_information = (
            X.groupby(["warehouse", "month"])[cols]
            .mean()
            .reset_index()
            .rename(columns=rename)
        )

        return month_information

    def __get_week_information(self, X):
        cols = ["mov_change", "user_activity_1", "user_activity_2"]
        rename = {col: f"weekly_avg_{col}" for col in cols}

        week_information = (
            X.groupby(["warehouse", "week"])[cols]
            .mean()
            .reset_index()
            .rename(columns=rename)
        )

        return week_information

    def __get_date_features(self, X):
        X["year"] = X["date"].dt.year
        X["month"] = X["date"].dt.month
        X["week"] = X["date"].dt.isocalendar().week
        X["day"] = X["date"].dt.day

        X["dayofyear"] = X["date"].dt.dayofyear
        X["dayofweek"] = X["date"].dt.dayofweek

        X["is_weekend"] = X["dayofweek"].isin([5, 6]).astype(int)

        X["sin_dayofyear"] = np.sin(2 * np.pi * X["dayofyear"] / 365)
        X["cos_dayofyear"] = np.cos(2 * np.pi * X["dayofyear"] / 365)
        X["sin_month"] = np.sin(2 * np.pi * X["month"] / 12)
        X["cos_month"] = np.cos(2 * np.pi * X["month"] / 12)
        X["sin_dayofweek"] = np.sin(2 * np.pi * X["dayofweek"] / 7)
        X["cos_dayofweek"] = np.cos(2 * np.pi * X["dayofweek"] / 7)
        X["sin_day"] = np.sin(2 * np.pi * X["day"] / 30)
        X["cos_day"] = np.cos(2 * np.pi * X["day"] / 30)

        X["continuous_week_representation"] = (
            (X["year"] - 2020) * 48 + X["month"] * 4 + X["day"] // 7
        )

        return X

    def __variable_interaction(self, X):
        cols = ["holiday", "shops_closed", "winter_school_holidays", "school_holidays"]

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                X[f"{cols[i]}+{cols[j]}"] = X[cols[i]] + X[cols[j]]
                X[f"{cols[i]}-{cols[j]}"] = X[cols[i]] - X[cols[j]]
                X[f"{cols[i]}*{cols[j]}"] = X[cols[i]] * X[cols[j]]

        return X

    def __add_special_dates(self, X):
        valentines_day = (X["day"] == 14) & (X["month"] == 2)
        X.loc[valentines_day, "holiday"] = 1
        X.loc[valentines_day, "holiday_name"] = "Valentines"

        black_friday = (  # day after the fourth Thursday in November
            (X["day"] >= 23)
            & (X["day"] <= 29)
            & (X["month"] == 11)
            & (X["dayofweek"] == 4)
        )
        X.loc[black_friday, "holiday"] = 1
        X.loc[black_friday, "holiday_name"] = "Black Friday"

        return X

    def __get_next_holiday_information(self, X):
        X["next_holiday_date"] = pd.NaT
        X["next_holiday_name"] = None

        X.loc[X["holiday"] == 1, "next_holiday_date"] = X.loc[X["holiday"] == 1, "date"]
        X.loc[X["holiday"] == 1, "next_holiday_name"] = X.loc[
            X["holiday"] == 1, "holiday_name"
        ]

        X.loc[
            (X["holiday"] == 1) & (X["next_holiday_name"].isna()),
            "next_holiday_name",
        ] = "MISSING_NAME"

        X["next_holiday_date"] = X.groupby("warehouse")["next_holiday_date"].bfill()
        X["next_holiday_name"] = X.groupby("warehouse")["next_holiday_name"].bfill()

        X["days_until_next_holiday"] = (X["next_holiday_date"] - X["date"]).dt.days

        return X

    def __get_past_holiday_information(self, X):
        X["past_holiday_date"] = pd.NaT
        X["past_holiday_name"] = None

        X.loc[X["holiday"] == 1, "past_holiday_date"] = X.loc[X["holiday"] == 1, "date"]
        X.loc[X["holiday"] == 1, "past_holiday_name"] = X.loc[
            X["holiday"] == 1, "holiday_name"
        ]
        X.loc[
            (X["holiday"] == 1) & (X["past_holiday_name"].isna()),
            "past_holiday_name",
        ] = "MISSING_NAME"

        X["past_holiday_date"] = X.groupby("warehouse")["past_holiday_date"].ffill()
        X["past_holiday_name"] = X.groupby("warehouse")["past_holiday_name"].ffill()

        X["days_after_past_holiday"] = (X["date"] - X["past_holiday_date"]).dt.days

        return X
