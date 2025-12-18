# File: Module/ChangePointDetector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt

class ChangePointDetector:
    """
    Class phát hiện điểm gãy (Change Point Detection) trên chuỗi thời gian.
    Hỗ trợ:
    1. Ruptures (PELT)
    2. CUSUM (Mean Shift)
    3. Bayesian (Likelihood Ratio)
    4. Trực quan hóa nâng cao
    """
    
    def __init__(self, data: pd.DataFrame, target_year: int = 2025, tolerance: int = 1):
        self.data = data
        self.target_year = target_year
        self.tolerance = tolerance
        self.results = pd.DataFrame()

    def _get_series(self, series_id):
        """Lấy chuỗi giá trị (value) của một series_id cụ thể."""
        df_sub = self.data[self.data['series_id'] == series_id].sort_values('year')
        if df_sub.empty:
            return None, None
        return df_sub['value'].values, df_sub['year'].values

    # -------------------------------------------------------------------------
    # 1. THUẬT TOÁN RUPTURES (PELT)
    # -------------------------------------------------------------------------
    def detect_ruptures(self, method="pelt", model="l2", pen=10) -> pd.DataFrame:
        results = []
        series_ids = self.data['series_id'].unique()

        for sid in series_ids:
            signal, years = self._get_series(sid)
            if signal is None or len(signal) < 3: 
                continue

            try:
                if method == "pelt":
                    algo = rpt.Pelt(model=model, min_size=1).fit(signal)
                    bkps = algo.predict(pen=pen)
                elif method == "binseg":
                    algo = rpt.Binseg(model=model, min_size=1).fit(signal)
                    bkps = algo.predict(n_bkps=1)
                else:
                    bkps = []

                detected_years = []
                for b in bkps:
                    if b < len(years):
                        detected_years.append(int(years[b])) 
                    elif b == len(years): 
                         detected_years.append(int(years[-1]))

                hit = any(abs(y - self.target_year) <= self.tolerance for y in detected_years)
                hit_year = next((y for y in detected_years if abs(y - self.target_year) <= self.tolerance), None)

                results.append({
                    "series_id": sid,
                    "algorithm": f"Ruptures_{method}",
                    "detected_years": detected_years,
                    "hit_target": hit,
                    "hit_year": hit_year
                })

            except Exception as e:
                print(f"Error Ruptures {sid}: {e}")
                continue

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 2. THUẬT TOÁN CUSUM
    # -------------------------------------------------------------------------
    def detect_cusum(self) -> pd.DataFrame:
        results = []
        series_ids = self.data['series_id'].unique()

        for sid in series_ids:
            signal, years = self._get_series(sid)
            if signal is None or len(signal) < 3:
                continue
            
            # Chuẩn hóa
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            if std_val == 0: std_val = 1e-9
            
            standardized = (signal - mean_val) / std_val
            cusum = np.cumsum(standardized)
            
            # Tìm điểm thay đổi lớn nhất
            diffs = np.diff(signal)
            max_change_idx = np.argmax(np.abs(diffs)) + 1 
            
            detected_year = int(years[max_change_idx])
            hit = abs(detected_year - self.target_year) <= self.tolerance

            results.append({
                "series_id": sid,
                "algorithm": "CUSUM_Mean",
                "detected_years": [detected_year],
                "hit_target": hit,
                "hit_year": detected_year if hit else None,
                "cusum_values": list(cusum)
            })
            
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 3. THUẬT TOÁN BAYESIAN (Mới thêm)
    # -------------------------------------------------------------------------
    def detect_bayesian(self, probability_threshold=0.5) -> pd.DataFrame:
        """
        Phát hiện điểm gãy dựa trên Likelihood Ratio Test (giả lập Bayesian đơn giản).
        """
        results = []
        series_ids = self.data['series_id'].unique()

        for sid in series_ids:
            signal, years = self._get_series(sid)
            if signal is None or len(signal) < 3:
                continue
            
            n = len(signal)
            # Log Likelihood H0 (Không có gãy)
            mean_0 = np.mean(signal)
            std_0 = np.std(signal)
            if std_0 == 0: std_0 = 1e-9
            ll0 = -0.5 * n * np.log(2 * np.pi * std_0**2) - np.sum((signal - mean_0)**2) / (2 * std_0**2)

            best_t = None
            max_gain = -np.inf
            
            # Quét qua các điểm cắt
            for t in range(1, n):
                seg1 = signal[:t]
                seg2 = signal[t:]
                
                if len(seg1) == 0 or len(seg2) == 0: continue

                m1, s1 = np.mean(seg1), np.std(seg1)
                m2, s2 = np.mean(seg2), np.std(seg2)
                if s1 == 0: s1 = 1e-9
                if s2 == 0: s2 = 1e-9
                
                # Log Likelihood H1 (Có gãy tại t)
                ll1 = -0.5 * t * np.log(2 * np.pi * s1**2) - np.sum((seg1 - m1)**2) / (2 * s1**2)
                ll2 = -0.5 * (n-t) * np.log(2 * np.pi * s2**2) - np.sum((seg2 - m2)**2) / (2 * s2**2)
                
                gain = (ll1 + ll2) - ll0
                if gain > max_gain:
                    max_gain = gain
                    best_t = t

            # Tính xác suất (heuristic từ log likelihood ratio)
            prob = 1 / (1 + np.exp(-max_gain))
            
            hit = False
            detected_years = []
            
            if prob > probability_threshold and best_t is not None:
                dy = int(years[best_t])
                detected_years.append(dy)
                hit = abs(dy - self.target_year) <= self.tolerance

            results.append({
                "series_id": sid,
                "algorithm": "Bayesian_BOCPD",
                "detected_years": detected_years,
                "hit_target": hit,
                "hit_year": detected_years[0] if hit and detected_years else None,
                "probability": prob
            })

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 4. TỔNG HỢP (Updated)
    # -------------------------------------------------------------------------
    def analyze_all(self):
        """Chạy tất cả thuật toán và tổng hợp kết quả."""
        df_pelt = self.detect_ruptures(method="pelt", pen=1)
        df_cusum = self.detect_cusum()
        df_bayes = self.detect_bayesian(probability_threshold=0.5)
        
        # Gộp tất cả kết quả lại (Quan trọng: Phải gộp cả df_bayes)
        self.results = pd.concat([df_pelt, df_cusum, df_bayes], ignore_index=True)
        return self.results

    def plot_series_with_cp(self, series_id, algorithm="Original", detected_years=None):
        """Vẽ biểu đồ cơ bản."""
        signal, years = self._get_series(series_id)
        if signal is None: return

        plt.figure(figsize=(10, 5))
        plt.plot(years, signal, marker='o', label=series_id)
        if detected_years:
            for cp in detected_years:
                plt.axvline(x=cp, color='red', linestyle='--', label=f'CP: {cp}')
        plt.title(f"{series_id} - {algorithm}")
        plt.legend()
        plt.show()

    # -------------------------------------------------------------------------
    # 5. VISUALIZATION NÂNG CAO 
    # -------------------------------------------------------------------------
    def plot_enhanced(self, series_id, detected_years):
        """Vẽ biểu đồ phân tích điểm gãy chuyên sâu."""
        signal, years = self._get_series(series_id)
        if signal is None: return
            
        if not detected_years or len(detected_years) == 0:
            self.plot_series_with_cp(series_id, "Original Data", [])
            return

        cp = detected_years[0]
        mask_before = years < cp
        mask_after = years >= cp
        
        if not (any(mask_before) and any(mask_after)):
            self.plot_series_with_cp(series_id, "Basic", detected_years)
            return

        mean_before = np.mean(signal[mask_before])
        mean_after = np.mean(signal[mask_after])
        pct_change = ((mean_after - mean_before) / mean_before) * 100

        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        
        # 1. Dữ liệu gốc
        plt.plot(years, signal, marker='o', color='#2c3e50', linewidth=3, label='Điểm trung bình', zorder=3)
        
        # 2. Điểm gãy
        plt.axvline(x=cp, color='#e74c3c', linestyle='--', linewidth=2, label=f'Điểm gãy ({cp})', zorder=2)
        
        # 3. Mean Levels
        plt.hlines(y=mean_before, xmin=min(years), xmax=cp, colors='#27ae60', linestyles=':', linewidth=2, label=f'Trước: {mean_before:.2f}')
        plt.hlines(y=mean_after, xmin=cp, xmax=max(years), colors='#c0392b', linestyles=':', linewidth=2, label=f'Sau: {mean_after:.2f}')
        
        # 4. Mũi tên & Annotation
        mid_y = (mean_before + mean_after) / 2
        plt.annotate('', xy=(cp, mean_after), xytext=(cp, mean_before),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3), zorder=4)
        
        bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#e74c3c", alpha=0.9)
        plt.text(cp + 0.05, mid_y, f"{pct_change:+.1f}%", ha='left', va='center', 
                 color='#c0392b', fontsize=13, fontweight='bold', bbox=bbox_props, zorder=5)

        # 5. Vùng tô màu
        plt.axvspan(cp, max(years)+0.5, color='#e74c3c', alpha=0.08, label='Giai đoạn mới')

        plt.title(f"Phân tích Điểm gãy: {series_id}", fontsize=15, fontweight='bold')
        plt.xticks(years, [str(y) for y in years])
        plt.legend()
        plt.tight_layout()
        plt.show()