# File: Model/ForecastBlockModel.py

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA


# =========================
# Linear Regression với Gradient Descent (baseline)
# =========================
class LinearRegressionGD:
    def __init__(
        self,
        lr: float = 0.01,
        n_iter: int = 2000,
        reg_lambda: float = 0.0,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.reg_lambda = float(reg_lambda)
        self.verbose = verbose
        self.random_state = random_state

        self.W: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(scale=0.01, size=n_features)
        self.b = 0.0

        for it in range(self.n_iter):
            y_pred = X @ self.W + self.b
            err = y_pred - y

            grad_W = (X.T @ err) / n_samples + self.reg_lambda * self.W
            grad_b = err.mean()

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            if self.verbose and (it % 200 == 0 or it == self.n_iter - 1):
                mse = (err ** 2).mean()
                loss = mse + 0.5 * self.reg_lambda * np.sum(self.W ** 2)
                # print(f"Iter {it:4d} | loss = {loss:.8f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError("Model chưa được fit.")
        X = np.asarray(X, dtype=float)
        return X @ self.W + self.b


# =========================
# MultiShareRegressor (Ridge + softmax / shift normalization)
# =========================
class MultiShareRegressor:
    """Regression model dự đoán share_in_year.
    - X = base_features + one-hot khoi_group
    - y = share_in_year
    - Predict theo năm: clip >=0 + normalize tổng = 1

    Workflow gốc trong Forecast.ipynb:
        fit(df_model) -> evaluate_year(df_model, year) -> forecast_next_year(df_model)
    """

    def __init__(
        self,
        base_features: Sequence[str],
        alpha: float = 1.0,
        random_state: int = 42,
        normalize_method: str = "softmax",
        temperature: float = 2.0,
    ):
        self.base_features = list(base_features)
        self.alpha = float(alpha)
        self.random_state = random_state
        self.normalize_method = normalize_method  # "softmax" or "shift"
        self.temperature = float(temperature)

        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state)),
            ]
        )
        self.dummy_cols_: Optional[list[str]] = None
        self._fitted: bool = False

    @staticmethod
    def _softmax(v: np.ndarray, temp: float = 1.0) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        z = (v - np.max(v)) / max(temp, 1e-9)
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    @staticmethod
    def _shift_norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        v = v - v.min()
        if v.sum() <= 0:
            v = np.ones_like(v)
        return v / (v.sum() + 1e-12)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["nam_hoc"] = pd.to_numeric(df["nam_hoc"], errors="coerce")
        df = df.dropna(subset=["nam_hoc"])
        df["nam_hoc"] = df["nam_hoc"].astype(int)
        return df

    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["khoi_group"] = df["khoi_group"].astype(str)

        # Đảm bảo đủ cột feature
        for c in self.base_features:
            if c not in df.columns:
                df[c] = 0.0

        # Ép numeric + fill NaN = 0 cho toàn bộ base_features
        base_X = df[self.base_features].copy()
        base_X = base_X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        base_X = base_X.reset_index(drop=True)

        # One-hot khoi_group
        dummies = pd.get_dummies(df["khoi_group"], prefix="khoi")

        if self.dummy_cols_ is None:
            self.dummy_cols_ = list(dummies.columns)
        else:
            for col in self.dummy_cols_:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[self.dummy_cols_]

        dummies = dummies.reset_index(drop=True)

        # Ghép lại thành X cuối
        X = pd.concat([base_X, dummies], axis=1)
        return X


    def fit(self, df_train: pd.DataFrame, weight_col: str = "n_students") -> "MultiShareRegressor":
        # Chuẩn hoá năm + sort
        df_train = self._clean(df_train).copy()

        # Đảm bảo cột feature tồn tại (nếu thiếu thì tạo = 0.0)
        for c in self.base_features:
            if c not in df_train.columns:
                df_train[c] = 0.0

        # Ép kiểu numeric rồi fill NaN = 0 cho feature & y
        for c in self.base_features:
            df_train[c] = pd.to_numeric(df_train[c], errors="coerce")
        df_train[self.base_features] = df_train[self.base_features].fillna(0.0)

        df_train["share_in_year"] = pd.to_numeric(
            df_train["share_in_year"], errors="coerce"
        ).fillna(0.0)

        # Trọng số (số thí sinh)
        if weight_col in df_train.columns:
            df_train[weight_col] = pd.to_numeric(
                df_train[weight_col], errors="coerce"
            ).fillna(0.0)
            w = df_train[weight_col].values.astype(float)
        else:
            w = None

        # Xây ma trận X sau khi đã làm sạch
        X = self._build_X(df_train).values
        y = df_train["share_in_year"].values.astype(float)

        # Fit Ridge trong pipeline
        if w is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, ridge__sample_weight=w)

        self._fitted = True
        return self



    def predict_share_year(self, df_year: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("MultiShareRegressor chưa được fit.")

        df_year = self._clean(df_year)
        df_year = df_year.copy()
        df_year["khoi_group"] = df_year["khoi_group"].astype(str)

        X = self._build_X(df_year).values
        preds = self.model.predict(X)

        # chọn chuẩn hoá
        if self.normalize_method == "softmax":
            share_pred = self._softmax(preds, temp=self.temperature)
        else:
            share_pred = self._shift_norm(preds)

        out = pd.DataFrame(
            {
                "nam_hoc": df_year["nam_hoc"].values.astype(int),
                "khoi_group": df_year["khoi_group"].values.astype(str),
                "share_pred": share_pred,
            }
        )

        out["share_pred"] = out["share_pred"] / (out["share_pred"].sum() + 1e-12)
        return out

    def evaluate_year(self, df_all: pd.DataFrame, year: int) -> tuple[pd.DataFrame, float]:
        df_all = self._clean(df_all)
        year = int(year)

        df_year = df_all[df_all["nam_hoc"] == year].copy()
        df_true = (
            df_year[["khoi_group", "share_in_year"]]
            .groupby("khoi_group", as_index=False)["share_in_year"]
            .sum()
            .rename(columns={"share_in_year": "share_true"})
        )

        df_pred = self.predict_share_year(df_year)
        df_eval = df_true.merge(df_pred, on="khoi_group", how="outer").fillna(0)
        df_eval["abs_err"] = (df_eval["share_true"] - df_eval["share_pred"]).abs()
        mae = df_eval["abs_err"].mean()
        return df_eval.sort_values("abs_err", ascending=False).reset_index(drop=True), mae

    def build_features_for_next_year(self, df_all: pd.DataFrame) -> pd.DataFrame:
        df_all = self._clean(df_all)
        df_all = df_all.sort_values(["khoi_group", "nam_hoc"]).reset_index(drop=True)

        last_year = int(df_all["nam_hoc"].max())
        df_last = df_all[df_all["nam_hoc"] == last_year].copy()

        df_next = df_last[["khoi_group", "share_in_year"]].copy()
        df_next["nam_hoc"] = last_year + 1
        df_next["year_idx"] = df_last["year_idx"].max() + 1

        # giữ nguyên share trước đó làm lag
        df_next["share_lag1"] = df_next["share_in_year"]
        df_next["delta_share"] = df_next["share_in_year"] - df_next["share_in_year"]

        # đảm bảo đủ base_features
        for c in self.base_features:
            if c not in df_next.columns:
                df_next[c] = 0

        return df_next[["nam_hoc", "khoi_group", "share_in_year"] + self.base_features].copy()

    def forecast_next_year(self, df_all: pd.DataFrame) -> pd.DataFrame:
        df_next = self.build_features_for_next_year(df_all)
        return self.predict_share_year(df_next)


# =========================
# ARIMA forecast cho share từng khối
# =========================
class ArimaShareModel:
    """ARIMA forecast cho share theo từng khoi_group.
    - Sử dụng PeriodIndex(freq="Y") giống notebook
    - Forecast từng group, clip >= 1e-9 và chuẩn hoá tổng share = 1
    """

    def __init__(self, min_points_for_ar1: int = 4):
        self.min_points_for_ar1 = min_points_for_ar1

    @staticmethod
    def _to_series(sub_df: pd.DataFrame) -> pd.Series:
        years = sub_df["nam_hoc"].astype(int).tolist()
        values = pd.to_numeric(sub_df["share_in_year"], errors="coerce").tolist()
        return pd.Series(
            values,
            index=pd.PeriodIndex(years, freq="Y"),
        ).asfreq("Y")

    def _forecast_one(self, sub_df: pd.DataFrame, target_year: int) -> float:
        s = self._to_series(sub_df)
        s = s.dropna()
        if len(s) == 0:
            return 0.0

        order = (1, 0, 0) if len(s) >= self.min_points_for_ar1 else (0, 0, 0)
        try:
            res = ARIMA(s, order=order).fit()
            pred = float(res.forecast(steps=1).iloc[0])
        except Exception:
            pred = float(s.iloc[-1])

        return max(pred, 1e-9)

    def forecast_year(self, df_all: pd.DataFrame, target_year: int) -> pd.DataFrame:
        df_all = df_all.copy()
        df_all["nam_hoc"] = df_all["nam_hoc"].astype(int)

        results: list[dict] = []
        for khoi_group, sub_df in df_all.groupby("khoi_group"):
            y_pred = self._forecast_one(sub_df, target_year=target_year)
            results.append(
                {
                    "khoi_group": khoi_group,
                    f"share_pred_{target_year}": y_pred,
                }
            )

        df_pred = pd.DataFrame(results)
        total = df_pred[f"share_pred_{target_year}"].sum()
        if total > 0:
            df_pred[f"share_pred_{target_year}"] /= total

        return df_pred

    def evaluate_year(self, df_all: pd.DataFrame, year: int) -> tuple[pd.DataFrame, float]:
        year = int(year)
        df_all = df_all.copy()
        df_all["nam_hoc"] = df_all["nam_hoc"].astype(int)

        df_true = (
            df_all[df_all["nam_hoc"] == year][["khoi_group", "share_in_year"]]
            .groupby("khoi_group", as_index=False)["share_in_year"]
            .mean()
            .rename(columns={"share_in_year": "share_true"})
        )

        df_pred = self.forecast_year(df_all, target_year=year)
        pred_col = f"share_pred_{year}"

        df_eval = df_true.merge(df_pred, on="khoi_group", how="outer").fillna(0)
        df_eval["abs_err"] = (df_eval["share_true"] - df_eval[pred_col]).abs()
        mae = df_eval["abs_err"].mean()
        return df_eval.sort_values("abs_err", ascending=False).reset_index(drop=True), mae


# =========================
# Wrapper ForecastBlockModel
# =========================
class ForecastBlockModel:
    """Bao các model dùng cho dự báo tỉ lệ khối (tổ hợp).

    Giữ nguyên logic từ Forecast.ipynb:
    - Chuẩn hoá df_block_features -> df_top (10 khối chính, share_in_year, year_idx, lag, post_2025).
    - So sánh 3 model: LinearRegressionGD, MultiShareRegressor, ArimaShareModel để lấy MAE 2025.
    - Chọn top-2 (MultiShareReg & ARIMA) để ensemble dự báo 2026.
    """

    __slots__ = (
        "TOP_BLOCKS",
        "df_top",
        "df_block_all",
        "year_test",
        "multi_reg",
        "arima",
        "mae_table",
        "top2_ids",
    )

    def __init__(
        self,
        df_block_features: pd.DataFrame,
        year_test: int = 2025,
        top_blocks: Optional[Sequence[str]] = None,
    ):
        self.TOP_BLOCKS = list(
            top_blocks
            or ["A00", "A01", "B00", "C00", "D01", "D07", "D08", "D09", "A02", "A03"]
        )

        # ----- Chuẩn hoá như notebook -----
        df_top = df_block_features[df_block_features["khoi"].isin(self.TOP_BLOCKS)].copy()
        df_top["nam_hoc"] = pd.to_numeric(df_top["nam_hoc"], errors="coerce")
        df_top = df_top.dropna(subset=["nam_hoc"]).copy()
        df_top["nam_hoc"] = df_top["nam_hoc"].astype(int)

        for col in ["mean", "median", "mode", "std", "min", "max", "n_students", "total_students_year"]:
            if col in df_top.columns:
                df_top[col] = pd.to_numeric(df_top[col], errors="coerce")

        # share trong 10 khối
        df_top["share_in_year"] = df_top["n_students"] / df_top["total_students_year"]
        df_top["share_in_year"] = df_top["share_in_year"] / (
            df_top.groupby("nam_hoc")["share_in_year"].transform("sum") + 1e-12
        )

        # year_idx & lag
        df_top = df_top.sort_values(["khoi", "nam_hoc"]).reset_index(drop=True)
        df_top["year_idx"] = df_top["nam_hoc"] - df_top["nam_hoc"].min()
        df_top["share_lag1"] = (
            df_top.groupby("khoi")["share_in_year"]
                  .shift(1)
        )
        df_top["delta_share"] = df_top["share_in_year"] - df_top["share_lag1"]

        # post_2025
        df_top["post_2025"] = (df_top["nam_hoc"] >= 2025).astype(int)

        self.df_top = df_top
        self.year_test = int(year_test)

        # Chuẩn hoá df_block_all cho MultiShareRegressor / ARIMA
        self.df_block_all = self.df_top.copy().rename(columns={"khoi": "khoi_group"})

        self.multi_reg = None
        self.arima = None
        self.mae_table = None
        self.top2_ids = None

    # -------- So sánh model trên năm test (2025) --------
    def compare_models_2025(self) -> pd.DataFrame:
        YEAR_TEST = self.year_test

        base_features_block = [
            "year_idx",
            "share_lag1",
            "delta_share",
            "post_2025",
        ]

        rows = []

        # 1) LinearRegressionGD baseline (fit theo từng khối)
        abs_err_list = []
        for khoi, sub in self.df_top.groupby("khoi"):
            sub = sub.sort_values("nam_hoc")
            if sub["nam_hoc"].nunique() < 2:
                continue

            train = sub[sub["nam_hoc"] != YEAR_TEST]
            test = sub[sub["nam_hoc"] == YEAR_TEST]
            if train.empty or test.empty:
                continue

            X_train = train[["year_idx"]].values
            y_train = train["share_in_year"].values
            X_test = test[["year_idx"]].values
            y_test = test["share_in_year"].values

            linreg = LinearRegressionGD(lr=0.01, n_iter=2000, reg_lambda=0.0, verbose=False)
            linreg.fit(X_train, y_train)

            y_pred = linreg.predict(X_test)
            abs_err = np.abs(y_test - y_pred).mean()
            abs_err_list.append(abs_err)

        mae_linear = float(np.mean(abs_err_list)) if abs_err_list else np.nan
        rows.append(
            {
                "model_id": "LinearGD",
                "model_display": "Linear-GD (baseline)",
                "MAE_2025": mae_linear,
            }
        )

        # 2) Multi-REG
        self.multi_reg = MultiShareRegressor(
            base_features=base_features_block,
            alpha=1.0,
            random_state=42,
            normalize_method="softmax",
            temperature=2.0,
        )
        self.multi_reg.fit(self.df_block_all, weight_col="n_students")
        eval_multi_2025, mae_multi_2025 = self.multi_reg.evaluate_year(self.df_block_all, YEAR_TEST)
        rows.append(
            {
                "model_id": "MultiShareReg",
                "model_display": "Multi-REG (Ridge + softmax)",
                "MAE_2025": mae_multi_2025,
            }
        )

        # 3) ARIMA
        self.arima = ArimaShareModel(min_points_for_ar1=3)
        eval_arima_2025, mae_arima_2025 = self.arima.evaluate_year(self.df_block_all, YEAR_TEST)
        rows.append(
            {
                "model_id": "ARIMA",
                "model_display": "ARIMA share",
                "MAE_2025": mae_arima_2025,
            }
        )

        mae_table = pd.DataFrame(rows).sort_values("MAE_2025").reset_index(drop=True)
        self.mae_table = mae_table

        # giữ đúng logic: chỉ chọn trong Multi-REG & ARIMA
        forecast_candidates = ["MultiShareReg", "ARIMA"]
        self.top2_ids = [
            mid for mid in mae_table["model_id"] if mid in forecast_candidates
        ][:2]

        print("\n🔎 So sánh MAE năm 2025 giữa các model:")
        print(mae_table[["model_display", "MAE_2025"]])
        print("👉 Top-2 model dùng cho forecast 2026:", self.top2_ids)

        return mae_table

    # -------- Ensemble forecast 2026 --------
    def forecast_2026(self) -> pd.DataFrame:
        if self.mae_table is None or self.top2_ids is None:
            self.compare_models_2025()

        YEAR_TEST = self.year_test
        YEAR_FORECAST = YEAR_TEST + 1

        df_block_all = self.df_block_all

        # Multi-REG 2026
        df_pred_2026_multi = None
        if "MultiShareReg" in self.top2_ids and self.multi_reg is not None:
            df_pred_2026_multi = self.multi_reg.forecast_next_year(df_block_all)
            df_pred_2026_multi = df_pred_2026_multi.rename(
                columns={"khoi_group": "khoi", "share_pred": "share_2026_multi"}
            )
            df_pred_2026_multi["nam_hoc"] = YEAR_FORECAST
            df_pred_2026_multi["share_2026_multi"] /= df_pred_2026_multi["share_2026_multi"].sum()

        # ARIMA 2026
        df_pred_2026_arima = None
        if "ARIMA" in self.top2_ids and self.arima is not None:
            df_pred_2026_arima = self.arima.forecast_year(df_block_all, target_year=YEAR_FORECAST)
            df_pred_2026_arima = df_pred_2026_arima.rename(
                columns={f"share_pred_{YEAR_FORECAST}": "share_2026_arima"}
            )
            df_pred_2026_arima["khoi"] = df_pred_2026_arima["khoi_group"]
            df_pred_2026_arima["nam_hoc"] = YEAR_FORECAST

        # Merge kết quả 2025 & 2026
        df_share_2025 = (
            self.df_top[self.df_top["nam_hoc"] == YEAR_TEST][["khoi", "share_in_year"]]
            .rename(columns={"share_in_year": "share_2025"})
        )

        df_merge = df_share_2025.copy()
        if df_pred_2026_multi is not None:
            df_merge = df_merge.merge(
                df_pred_2026_multi[["khoi", "share_2026_multi"]],
                on="khoi",
                how="left",
            )
        if df_pred_2026_arima is not None:
            df_merge = df_merge.merge(
                df_pred_2026_arima[["khoi", "share_2026_arima"]],
                on="khoi",
                how="left",
            )

        # Ensemble: trung bình các model có trong top2
        num_models = 0
        share_sum = np.zeros(len(df_merge), dtype=float)
        if "MultiShareReg" in self.top2_ids and "share_2026_multi" in df_merge.columns:
            share_sum += df_merge["share_2026_multi"].fillna(0).values
            num_models += 1
        if "ARIMA" in self.top2_ids and "share_2026_arima" in df_merge.columns:
            share_sum += df_merge["share_2026_arima"].fillna(0).values
            num_models += 1

        if num_models > 0:
            share_final = share_sum / num_models
        else:
            # fallback: giữ nguyên tỉ lệ 2025
            share_final = df_merge["share_2025"].values

        share_final = np.clip(share_final, 1e-9, None)
        share_final = share_final / (share_final.sum() + 1e-12)

        df_merge["share_2026_final"] = share_final

        df_pred_2026_final = df_merge[
            [
                "khoi",
                "share_2025",
                *(["share_2026_multi"] if "share_2026_multi" in df_merge.columns else []),
                *(["share_2026_arima"] if "share_2026_arima" in df_merge.columns else []),
                "share_2026_final",
            ]
        ].copy()
        df_pred_2026_final["nam_hoc"] = YEAR_FORECAST

        # Gom 2023–2026 vào một bảng
        df_block_share_2023_2026 = pd.concat(
            [
                self.df_top[["nam_hoc", "khoi", "share_in_year"]].rename(
                    columns={"share_in_year": "share"}
                ),
                df_pred_2026_final[["nam_hoc", "khoi", "share_2026_final"]].rename(
                    columns={"share_2026_final": "share"}
                ),
            ],
            ignore_index=True,
        )

        df_block_share_2023_2026 = df_block_share_2023_2026.sort_values(
            ["khoi", "nam_hoc"]
        ).reset_index(drop=True)

        print("\nBảng share 2026 (final):")
        print(df_block_share_2023_2026[df_block_share_2023_2026["nam_hoc"] == YEAR_FORECAST])

        return df_block_share_2023_2026
