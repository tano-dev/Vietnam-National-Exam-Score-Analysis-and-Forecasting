import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import warnings

# Thử import thư viện Bayesian nếu có
try:
    from bayesian_changepoint_detection.priors import const_prior
    from bayesian_changepoint_detection.offline_likelihoods import StudentT
    from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
    HAS_BAYESIAN = True
except ImportError:
    HAS_BAYESIAN = False

class ChangePointDetector:
    """
    Thực hiện phát hiện điểm thay đổi (Change Point Detection) trên dữ liệu chuỗi thời gian.
    Hỗ trợ: Ruptures (PELT, BinSeg, Window), CUSUM, và Bayesian (Offline).
    """
    def __init__(self, data: pd.DataFrame, target_year: int = 2025, tolerance: int = 1):
        """
        :param data: DataFrame output từ ChangePointPreparer (cols: year, series_id, value)
        :param target_year: Năm mục tiêu để đối chiếu (ví dụ: 2025)
        :param tolerance: Sai số cho phép (năm) khi đối chiếu với target_year
        """
        self.data = data
        self.target_year = target_year
        self.tolerance = tolerance
        
        if self.data.empty:
            warnings.warn("Dữ liệu đầu vào rỗng.")

    def _get_signal(self, series_id: str):
        """Lấy tín hiệu array và danh sách năm tương ứng cho một series_id."""
        df_sub = self.data[self.data['series_id'] == series_id].sort_values('year')
        if df_sub.empty:
            return None, None
        return df_sub['value'].values, df_sub['year'].values

    def _map_indices_to_years(self, indices, years_arr):
        """Chuyển đổi index của mảng thành năm thực tế."""
        detected_years = []
        n_samples = len(years_arr)
        for idx in indices:
            # Ruptures thường trả về index là điểm bắt đầu segment mới
            # hoặc điểm kết thúc segment cũ. Ta lấy index-1 để map vào năm xảy ra gãy
            if 0 < idx <= n_samples:
                detected_years.append(years_arr[idx-1]) 
        return detected_years

    def check_target_hit(self, detected_years):
        """Kiểm tra xem các điểm tìm được có nằm trong vùng target_year ± tolerance không."""
        for y in detected_years:
            if abs(y - self.target_year) <= self.tolerance:
                return True, y
        return False, None

    # ---------------------------------------------------------
    # 1. Ruptures Methods
    # ---------------------------------------------------------
    def detect_ruptures(self, method="pelt", model="l2", pen=10, width=3, n_bkps=5):
        """
        Áp dụng các thuật toán từ thư viện Ruptures.
        :param method: 'pelt', 'binseg', 'window'
        :param model: 'l2', 'rbf', 'l1', 'normal'
        :param pen: Penalty cho PELT/Window (độ nhạy)
        :param width: Độ rộng cửa sổ cho Window-based
        :param n_bkps: Số điểm gãy tối đa cho Binary Segmentation
        """
        results = []
        unique_series = self.data['series_id'].unique()

        for series_id in unique_series:
            signal, years = self._get_signal(series_id)
            if signal is None or len(signal) < width: 
                continue

            try:
                algo = None
                # Chọn thuật toán
                if method == "pelt":
                    algo = rpt.Pelt(model=model).fit(signal)
                    bkps = algo.predict(pen=pen)
                elif method == "binseg":
                    algo = rpt.Binseg(model=model).fit(signal)
                    bkps = algo.predict(n_bkps=n_bkps)
                elif method == "window":
                    algo = rpt.Window(width=width, model=model).fit(signal)
                    bkps = algo.predict(pen=pen)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Ruptures luôn trả về index cuối cùng của chuỗi, cần loại bỏ
                bkps = [x for x in bkps if x < len(signal)]
                
                detected_years = self._map_indices_to_years(bkps, years)
                is_hit, hit_val = self.check_target_hit(detected_years)

                results.append({
                    "series_id": series_id,
                    "algorithm": f"Ruptures_{method}_{model}",
                    "detected_years": detected_years,
                    "hit_target": is_hit,
                    "hit_year": hit_val
                })
            except Exception as e:
                print(f"Error Ruptures {series_id}: {e}")
                continue

        return pd.DataFrame(results)

    # ---------------------------------------------------------
    # 2. CUSUM (Cumulative Sum)
    # ---------------------------------------------------------
    def detect_cusum(self, threshold_std=2.0):
        """
        Sử dụng CUSUM để phát hiện điểm thay đổi trung bình lớn nhất.
        Logic: Tính tổng tích lũy của (giá trị - trung bình). Điểm cực trị của CUSUM
        thường là điểm thay đổi.
        """
        results = []
        unique_series = self.data['series_id'].unique()

        for series_id in unique_series:
            signal, years = self._get_signal(series_id)
            if signal is None or len(signal) < 3: continue

            # Tính CUSUM
            mean_val = np.mean(signal)
            # Normalized signal
            s = signal - mean_val
            cusum = np.cumsum(s)
            
            # Tìm điểm có độ lệch tích lũy lớn nhất (Abs)
            max_idx = np.argmax(np.abs(cusum))
            
            # Kiểm tra độ tin cậy đơn giản (Z-score tại điểm đó)
            # Nếu sự thay đổi không đáng kể so với độ lệch chuẩn, bỏ qua
            std_val = np.std(signal)
            if std_val == 0: 
                detected_year = []
            else:
                # Đây là heuristic đơn giản
                detected_year = [years[max_idx]]
            
            is_hit, hit_val = self.check_target_hit(detected_year)

            results.append({
                "series_id": series_id,
                "algorithm": "CUSUM_Mean",
                "detected_years": detected_year,
                "hit_target": is_hit,
                "hit_year": hit_val,
                "cusum_values": cusum  # Lưu để plot nếu cần
            })
            
        return pd.DataFrame(results)

    # ---------------------------------------------------------
    # 3. Bayesian Change Point (Offline)
    # ---------------------------------------------------------
    def detect_bayesian(self, truncate=-10):
        """
        Sử dụng Bayesian Offline Change Point Detection.
        Yêu cầu: thư viện `bayesian_changepoint_detection`.
        Logic: Tính xác suất hậu nghiệm (posterior probability) của độ dài run-length.
        """
        if not HAS_BAYESIAN:
            print("Warning: Chưa cài đặt thư viện 'bayesian_changepoint_detection'. Bỏ qua.")
            return pd.DataFrame()

        results = []
        unique_series = self.data['series_id'].unique()

        for series_id in unique_series:
            signal, years = self._get_signal(series_id)
            if signal is None or len(signal) < 3: continue

            try:
                # Cấu hình Priors (giả định phân phối Student-T cho dữ liệu chưa biết variance)
                # Hazard rate lambda = 1/100 (giả định cứ 100 điểm thì có 1 điểm gãy)
                Q, P, Pcp = offline_changepoint_detection(
                    signal, 
                    partial(const_prior, l=(len(signal) + 1)), 
                    StudentT(), 
                    truncate=truncate
                )
                
                # Pcp là xác suất có changepoint tại mỗi thời điểm
                # Lấy các điểm có xác suất > ngưỡng (ví dụ 0.5 hoặc đỉnh local max)
                # Ở đây ta lấy đỉnh cao nhất để đơn giản hoá
                
                # Cách lấy điểm gãy từ Pcp: tìm peaks
                # Pcp shape: (time_steps, 1) -> exp(Pcp) để ra xác suất
                probs = np.exp(Pcp).sum(0) # Marginal probability
                
                # Lấy các điểm có xác suất cao đột biến (ngưỡng 0.3)
                # Bỏ qua vài điểm đầu/cuối do biên
                threshold = 0.3
                detected_indices = np.where(probs[1:-1] > threshold)[0] + 1
                
                detected_years = self._map_indices_to_years(detected_indices, years)
                is_hit, hit_val = self.check_target_hit(detected_years)

                results.append({
                    "series_id": series_id,
                    "algorithm": "Bayesian_Offline",
                    "detected_years": detected_years,
                    "hit_target": is_hit,
                    "hit_year": hit_val,
                    "probs": probs # Lưu để plot
                })

            except Exception as e:
                # Fallback nếu thư viện lỗi hoặc config sai
                print(f"Bayesian Error {series_id}: {e}")
                continue

        return pd.DataFrame(results)

    # ---------------------------------------------------------
    # Tổng hợp & Visualization
    # ---------------------------------------------------------
    def analyze_all(self):
        """Chạy tất cả các thuật toán và tổng hợp kết quả."""
        df_pelt = self.detect_ruptures(method="pelt", model="l2", pen=10)
        df_binseg = self.detect_ruptures(method="binseg", model="l2", n_bkps=3)
        df_win = self.detect_ruptures(method="window", model="l2", width=3, pen=10)
        df_cusum = self.detect_cusum()
        
        # Merge kết quả
        dfs = [df_pelt, df_binseg, df_win, df_cusum]
        
        if HAS_BAYESIAN:
            # Cần functools cho Bayesian
            global partial
            from functools import partial
            df_bayes = self.detect_bayesian()
            if not df_bayes.empty:
                dfs.append(df_bayes)

        final_df = pd.concat(dfs, ignore_index=True)
        return final_df.sort_values(["series_id", "algorithm"])

    def plot_series_with_cp(self, series_id, algorithm=None, detected_years=None):
        """Vẽ biểu đồ chuỗi thời gian và các điểm gãy đã phát hiện."""
        signal, years = self._get_signal(series_id)
        if signal is None: return

        plt.figure(figsize=(10, 4))
        plt.plot(years, signal, label='Value', marker='o')
        
        if detected_years:
            for cp in detected_years:
                plt.axvline(x=cp, color='red', linestyle='--', label=f'CP: {cp}')
                # Highlight vùng 2025 nếu cần
                if abs(cp - self.target_year) <= self.tolerance:
                    plt.text(cp, max(signal), 'Target Hit!', color='red', rotation=90)

        plt.title(f"Change Points: {series_id} - Algo: {algorithm}")
        plt.xlabel("Year")
        plt.ylabel("Value")
        
        # Xử lý label trùng lặp
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(True, alpha=0.3)
        plt.show()