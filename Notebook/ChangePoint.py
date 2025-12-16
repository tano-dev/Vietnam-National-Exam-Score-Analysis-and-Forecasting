import pandas as pd
import numpy as np




# Lưu ý: Class này sẽ hoạt động khi được gọi từ Notebook đã setup đường dẫn
# Nên ta vẫn import từ 'Module' bình thường.
try:
    from Module.Load_Data import CleanDataLoader
except ImportError:
    # Fallback cho trường hợp IDE báo lỗi đỏ (không ảnh hưởng khi chạy thật từ Notebook)
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Module.Load_Data import CleanDataLoader

class ChangePointPreparer:
    """
    Chuẩn bị dữ liệu cho bài toán Change Point Detection.
    """
    def __init__(self, clean_loader: CleanDataLoader):
        self.loader = clean_loader

    def _transform_to_timeseries(self, df: pd.DataFrame, series_id: str, metric: str = "mean") -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Kiểm tra cột metric
        if metric not in df.columns:
            # Fallback: lấy cột số đầu tiên không phải nam_hoc
            numeric_cols = df.select_dtypes(include=np.number).columns.difference(['nam_hoc'])
            if len(numeric_cols) > 0:
                metric = numeric_cols[0]
            else:
                return pd.DataFrame()
            
        df_out = df[["nam_hoc", metric]].copy()
        df_out.columns = ["year", "value"]
        df_out["series_id"] = series_id
        return df_out[["year", "series_id", "value"]].sort_values("year").reset_index(drop=True)

    def get_subject_series(self, subjects: list[str], metric: str = "mean") -> pd.DataFrame:
        results = []
        for subj in subjects:
            try:
                df = self.loader.get_subject_data(subject=subj, kind="analysis")
                ts = self._transform_to_timeseries(df, series_id=subj, metric=metric)
                results.append(ts)
            except Exception:
                continue
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["year", "series_id", "value"])

    def get_block_series(self, blocks: list[str], metric: str = "mean") -> pd.DataFrame:
        results = []
        for blk in blocks:
            try:
                df = self.loader.get_block_data(block=blk, kind="analysis")
                ts = self._transform_to_timeseries(df, series_id=blk, metric=metric)
                results.append(ts)
            except Exception:
                continue
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["year", "series_id", "value"])

    def get_province_series(self, provinces: list[str], metric: str = "mean") -> pd.DataFrame:
        results = []
        for prov in provinces:
            try:
                df = self.loader.get_province_data(province=prov, kind="analysis")
                ts = self._transform_to_timeseries(df, series_id=prov, metric=metric)
                results.append(ts)
            except Exception:
                continue
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["year", "series_id", "value"])
