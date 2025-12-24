# File: Model/ForecastSubjectModel.py

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from scipy.stats import norm


# =========================
# Helper metrics
# =========================
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =========================
# DataFrame builder (giữ nguyên logic từ Forecast.ipynb)
# =========================
def build_ml_df(df: pd.DataFrame, df_rates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Chuẩn hoá dataframe dùng cho mô hình dự báo điểm trung bình theo môn.

    Tham số:
        df       : bảng thống kê theo môn-năm, có cột ['year', 'subject', 'mean_score', ...]
        df_rates : bảng tỉ lệ >=5, >=8 theo môn-năm (year, subject, rate_ge_5, rate_ge_8)

    Trả về:
        df_ml    : dataframe đã bổ sung feature post_2025, delta_score, mean_prev, is_first_year, ...
    """
    df = df.sort_values(["subject", "year"]).copy()

    df["post_2025"] = (df["year"] >= 2025).astype(int)

    df["delta_score"] = (
        df.groupby("subject")["mean_score"]
          .diff()
          .fillna(0)
    )

    df["mean_prev"] = (
        df.groupby("subject")["mean_score"]
          .transform(lambda x: x.shift().expanding().mean())
    )

    df["is_first_year"] = (df.groupby("subject").cumcount() == 0).astype(int)

    df["mean_prev"] = df["mean_prev"].fillna(0)

    # merge tỉ lệ nếu có
    if df_rates is not None:
        df = df.merge(df_rates, on=["year", "subject"], how="left")
        df[["rate_ge_5", "rate_ge_8"]] = df[["rate_ge_5", "rate_ge_8"]].fillna(0)

    return df


# =========================
# ARIMA / ARIMAX theo từng môn
# =========================
class ArimaSubjectModel:
    """
    ARIMA / ARIMAX theo từng môn học
    - scenario = "raw"  : ARIMA (không dùng biến post_2025)
    - scenario = "post" : ARIMAX với biến ngoại sinh post_2025

    evaluate_year() trả về dataframe chứa:
      subject | mean_true | mean_pred | MAE | RMSE | MAPE
    """

    def __init__(self, min_points_for_ar1: int = 4):
        self.min_points_for_ar1 = min_points_for_ar1

    # -------- Utils --------
    @staticmethod
    def _to_series(df: pd.DataFrame, value_col: str) -> pd.Series:
        years = df["year"].astype(int)
        values = pd.to_numeric(df[value_col], errors="coerce")
        return pd.Series(
            values.values,
            index=pd.PeriodIndex(years, freq="Y")
        ).asfreq("Y")

    @staticmethod
    def _to_exog(df: pd.DataFrame, exog_col: str) -> pd.DataFrame:
        years = df["year"].astype(int)
        values = pd.to_numeric(df[exog_col], errors="coerce")
        return pd.DataFrame(
            values.values,
            index=pd.PeriodIndex(years, freq="Y"),
            columns=[exog_col]
        ).asfreq("Y")

    # -------- Core: fit + forecast 1 môn --------
    def _fit_and_forecast_one(
        self,
        df_sub: pd.DataFrame,
        target_year: int,
        scenario: str,
    ) -> Optional[dict]:
        df_sub = df_sub.sort_values("year")

        df_train = df_sub[df_sub["year"] < target_year]
        df_test = df_sub[df_sub["year"] == target_year]

        if len(df_train) == 0 or len(df_test) == 0:
            return None

        y_train = self._to_series(df_train, "mean_score")
        if len(y_train.dropna()) < self.min_points_for_ar1:
            order = (0, 0, 0)
        else:
            order = (1, 0, 0)

        last_value = float(y_train.dropna().iloc[-1])

        try:
            # ARIMAX
            if scenario == "post" and "post_2025" in df_train.columns:
                exog_train = self._to_exog(df_train, "post_2025")
                model = ARIMA(y_train, order=order, exog=exog_train).fit()

                exog_future = pd.DataFrame(
                    [1 if target_year >= 2025 else 0],
                    index=pd.PeriodIndex([target_year], freq="Y"),
                    columns=["post_2025"]
                )
                y_pred = float(model.forecast(steps=1, exog=exog_future).iloc[0])

            # ARIMA raw
            else:
                model = ARIMA(y_train, order=order).fit()
                y_pred = float(model.forecast(steps=1).iloc[0])

        except Exception:
            # fallback: giữ nguyên giá trị cuối cùng
            y_pred = last_value

        y_true = float(df_test["mean_score"].iloc[0])
        err = y_true - y_pred

        return {
            "subject": df_sub["subject"].iloc[0],
            "mean_true": y_true,
            "mean_pred": y_pred,
            "MAE": abs(err),
            "RMSE": abs(err),
            "MAPE": abs(err) / y_true * 100 if y_true != 0 else np.nan,
        }

    # -------- Evaluate cho 1 năm --------
    def evaluate_year(
        self,
        df_all: pd.DataFrame,
        year: int,
        scenario: str = "raw",
        verbose: bool = True,
    ) -> pd.DataFrame:
        df_all = df_all.copy()
        df_all["year"] = df_all["year"].astype(int)

        results: list[dict] = []
        for subject, sub_df in df_all.groupby("subject"):
            res = self._fit_and_forecast_one(sub_df, target_year=year, scenario=scenario)
            if res is None:
                continue
            results.append(res)

        if len(results) == 0:
            return pd.DataFrame(
                columns=["subject", "mean_true", "mean_pred", "MAE", "RMSE", "MAPE"]
            )

        df_result = pd.DataFrame(results)[
            ["subject", "mean_true", "mean_pred", "MAE", "RMSE", "MAPE"]
        ]

        mean_mae = df_result["MAE"].mean()
        n_subject = len(df_result)

        if verbose:
            tag = "ARIMAX" if scenario == "post" else "ARIMA"
            print(f"[{tag} {n_subject} subject] MAE ({year}) = {mean_mae:.4f}")

        return df_result


# =========================
# RF-global model
# =========================
class RFGlobalModel:
    """
    RandomForestRegressor train chung cho tất cả môn
    subject được mã hóa one-hot.

    evaluate_year() trả về dataframe:
      subject | mean_true | mean_pred | MAE | RMSE | MAPE
    """

    def __init__(
        self,
        base_features=None,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        self.base_features = base_features or [
            "rate_ge_5",
            "rate_ge_8",
            "is_first_year",
        ]

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model: Optional[RandomForestRegressor] = None
        self.subject_cols: Optional[list[str]] = None

    # -------- Feature builder --------
    def build_features(self, df: pd.DataFrame, scenario: str = "post") -> pd.DataFrame:
        df = df.copy()

        base_cols = list(self.base_features)
        if scenario == "post" and "post_2025" not in base_cols:
            base_cols.append("post_2025")

        for col in base_cols:
            if col not in df.columns:
                df[col] = 0

        base_X = df[base_cols].reset_index(drop=True)
        subject_dummies = pd.get_dummies(df["subject"].astype(str), prefix="sub")

        if self.subject_cols is None:
            self.subject_cols = list(subject_dummies.columns)
        else:
            for col in self.subject_cols:
                if col not in subject_dummies.columns:
                    subject_dummies[col] = 0
            subject_dummies = subject_dummies[self.subject_cols]

        X = pd.concat([base_X, subject_dummies.reset_index(drop=True)], axis=1)
        return X

    # -------- Fit --------
    def fit(self, df_train: pd.DataFrame, scenario: str = "post") -> None:
        X_train = self.build_features(df_train, scenario)
        y_train = df_train["mean_score"]

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)

    # -------- Predict 1 năm --------
    def predict_year(self, df_test: pd.DataFrame, scenario: str) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("RFGlobalModel chưa được fit.")
        X_test = self.build_features(df_test, scenario)
        df_out = df_test.copy()
        df_out["mean_pred"] = self.model.predict(X_test)
        return df_out

    # -------- Evaluate 1 năm --------
    def evaluate_year(self, X_test: pd.DataFrame, scenario: str = "post") -> pd.DataFrame:
        df_pred = self.predict_year(X_test, scenario=scenario)

        df_pred["mean_true"] = df_pred["mean_score"]
        df_pred["MAE"] = np.abs(df_pred["mean_true"] - df_pred["mean_pred"])
        df_pred["RMSE"] = df_pred["MAE"]
        df_pred["MAPE"] = (
            df_pred["MAE"] / df_pred["mean_true"] * 100
        )

        result = df_pred[
            ["subject", "mean_true", "mean_pred", "MAE", "RMSE", "MAPE"]
        ]

        print(
            f"[RF-global-{scenario} {result['subject'].nunique()} subject] "
            f"MAE (2025) = {result['MAE'].mean():.4f}"
        )

        return result


# =========================
# XGB-global model
# =========================
class XGBGlobalModel:
    """
    XGBoost train chung cho tất cả môn
    subject được mã hóa one-hot.

    evaluate_year() trả về dataframe:
      subject | mean_true | mean_pred | MAE | RMSE | MAPE
    """

    def __init__(
        self,
        features_base=None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        # base features (không gồm subject)
        self.features_base = features_base or [
            "rate_ge_5",
            "rate_ge_8",
            "is_first_year",
            "post_2025",
        ]

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
        )
        self.subject_cols: Optional[list[str]] = None

    # -------- Feature builder --------
    def build_features(self, df: pd.DataFrame, scenario: str = "post", fit_ohe: bool = False) -> pd.DataFrame:
        df = df.copy()

        for col in self.features_base:
            if col not in df.columns:
                df[col] = 0

        X_base = df[self.features_base].reset_index(drop=True)
        subject_dummies = pd.get_dummies(df["subject"].astype(str), prefix="sub")

        if fit_ohe or (self.subject_cols is None):
            self.subject_cols = list(subject_dummies.columns)
        else:
            for col in self.subject_cols:
                if col not in subject_dummies.columns:
                    subject_dummies[col] = 0
            subject_dummies = subject_dummies[self.subject_cols]

        X = pd.concat([X_base, subject_dummies.reset_index(drop=True)], axis=1)
        return X

    # -------- Fit --------
    def fit(self, df_train: pd.DataFrame, scenario: str = "post") -> None:
        X_train = self.build_features(df_train, scenario=scenario, fit_ohe=True)
        y_train = df_train["mean_score"].values.astype(float)
        self.model.fit(X_train, y_train)

    # -------- Predict 1 năm --------
    def predict_year(self, X_test: pd.DataFrame, scenario: str = "post") -> pd.DataFrame:
        X = self.build_features(X_test, scenario=scenario, fit_ohe=False)
        preds = self.model.predict(X)

        df_pred = X_test.copy()
        df_pred["mean_pred"] = preds

        return df_pred[["subject", "mean_score", "mean_pred"]]

    # -------- Evaluate 1 năm --------
    def evaluate_year(self, X_test: pd.DataFrame, scenario: str = "post") -> pd.DataFrame:
        df_pred = self.predict_year(X_test, scenario=scenario)

        df_pred["mean_true"] = df_pred["mean_score"]
        df_pred["MAE"] = np.abs(df_pred["mean_true"] - df_pred["mean_pred"])
        df_pred["RMSE"] = df_pred["MAE"]
        df_pred["MAPE"] = (
            df_pred["MAE"] / df_pred["mean_true"] * 100
        )

        result = df_pred[
            ["subject", "mean_true", "mean_pred", "MAE", "RMSE", "MAPE"]
        ]

        print(
            f"[XGB-global-{scenario} {result['subject'].nunique()} subject] "
            f"MAE (2025) = {result['MAE'].mean():.4f}"
        )

        return result


# =========================
# Forecast 2026 với XGB + khoảng tin cậy
# =========================
def forecast_2026_xgb_with_ci(
    model: XGBGlobalModel,
    df_ml: pd.DataFrame,
    year_start: int = 2023,
    year_end: int = 2026,
    alpha: float = 0.05,
    scenario: str = "post",
) -> pd.DataFrame:
    """
    Forecast điểm thi bằng XGBoost và xuất bảng:
        subject | year | mean_actual | mean_predicted | lower_CI | upper_CI
    """

    df_ml = df_ml.copy()
    df_ml["year"] = df_ml["year"].astype(int)

    results = []

    for year in range(year_start, year_end + 1):
        df_train = df_ml[df_ml["year"] < year].copy()
        df_test = df_ml[df_ml["year"] == year].copy()

        if df_train.empty or df_test.empty:
            continue

        # Fit XGB cho từng năm
        model.fit(df_train, scenario=scenario)
        df_year_pred = model.evaluate_year(df_test, scenario=scenario)
        df_year_pred["year"] = year
        
        # Tính khoảng tin cậy dựa trên sai số
        df_year_pred["residual"] = df_year_pred["mean_true"] - df_year_pred["mean_pred"]
        sigma = df_year_pred["residual"].std(ddof=1)

        z = norm.ppf(1 - alpha / 2)
        df_year_pred["lower_CI"] = df_year_pred["mean_pred"] - z * sigma
        df_year_pred["upper_CI"] = df_year_pred["mean_pred"] + z * sigma

        df_year_out = df_year_pred[
            ["subject", "year", "mean_true", "mean_pred", "lower_CI", "upper_CI"]
        ].rename(
            columns={
                "mean_true": "mean_actual",
                "mean_pred": "mean_predicted",
            }
        )

        results.append(df_year_out)

    if not results:
        return pd.DataFrame(
            columns=["subject", "year", "mean_actual", "mean_predicted", "lower_CI", "upper_CI"]
        )

    df_final = (
        pd.concat(results)
        .sort_values(["subject", "year"])
        .reset_index(drop=True)
    )

    return df_final


# =========================
# Wrapper class cho toàn bộ pipeline dự báo theo môn
# =========================
class ForecastModel:
    """
    Lớp bao toàn bộ các model dự báo điểm trung bình theo môn (ARIMA, RF, XGB).

    Workflow (tương đương notebook Forecast.ipynb phần 3.1):
        fm = ForecastModel(df_ml)
        arima_res = fm.evaluate_arima(scenario="raw")
        rf_res    = fm.evaluate_rf(scenario="raw")
        xgb_res   = fm.evaluate_xgb(scenario="post")
        df_forecast_2026 = fm.forecast_2026_xgb(alpha=0.05, scenario="post")
    """

    __slots__ = (
        "df_ml",
        "year_for_test",
        "arima",
        "rf",
        "xgb",
    )

    def __init__(self, df_ml: pd.DataFrame):
        self.df_ml = df_ml.copy()
        self.df_ml["year"] = self.df_ml["year"].astype(int)
        self.year_for_test = int(self.df_ml["year"].max())

        # Khởi tạo các model đúng như trong Forecast.ipynb
        self.arima = ArimaSubjectModel(min_points_for_ar1=4)
        self.rf = RFGlobalModel()
        self.xgb = XGBGlobalModel()

    # ---- ARIMA ----
    def evaluate_arima(self, scenario: str = "raw", year: Optional[int] = None) -> pd.DataFrame:
        year = int(year or self.year_for_test)
        return self.arima.evaluate_year(self.df_ml, year=year, scenario=scenario)

    # ---- RF-global ----
    def evaluate_rf(self, scenario: str = "raw") -> pd.DataFrame:
        X_train = self.df_ml[self.df_ml["year"] < self.year_for_test].copy()
        X_test = self.df_ml[self.df_ml["year"] == self.year_for_test].copy()

        self.rf.fit(X_train, scenario=scenario)
        return self.rf.evaluate_year(X_test, scenario=scenario)

    # ---- XGB-global ----
    def evaluate_xgb(self, scenario: str = "post") -> pd.DataFrame:
        X_train = self.df_ml[self.df_ml["year"] < self.year_for_test].copy()
        X_test = self.df_ml[self.df_ml["year"] == self.year_for_test].copy()

        self.xgb.fit(X_train, scenario=scenario)
        return self.xgb.evaluate_year(X_test, scenario=scenario)

    # ---- Forecast 2026 với XGB + CI ----
    def forecast_2026_xgb(self, alpha: float = 0.05, scenario: str = "post") -> pd.DataFrame:
        year_start = int(self.df_ml["year"].min())
        year_end = self.year_for_test + 1  # 2026

        return forecast_2026_xgb_with_ci(
            model=self.xgb,
            df_ml=self.df_ml,
            year_start=year_start,
            year_end=year_end,
            alpha=alpha,
            scenario=scenario,
        )
