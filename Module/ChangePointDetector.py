import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt

class ChangePointDetector:
    """
    Class phát hiện điểm gãy (Change Point Detection) trên chuỗi thời gian.
    Hỗ trợ:
    1. Ruptures (PELT, Binary Segmentation...)
    2. CUSUM (Cumulative Sum)
    3. Trực quan hóa kết quả
    """
    
    def __init__(self, data: pd.DataFrame, target_year: int = 2025, tolerance: int = 1):
        """
        Args:
            data: DataFrame có các cột ['year', 'series_id', 'value']
            target_year: Năm mục tiêu cần kiểm định (ví dụ 2025)
            tolerance: Sai số chấp nhận được (ví dụ +/- 1 năm)
        """
        self.data = data
        self.target_year = target_year
        self.tolerance = tolerance
        self.results = []

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
        """
        Chạy thuật toán Ruptures để tìm điểm gãy.
        
        Args:
            method: 'pelt' (tốt nhất), 'binseg', 'window'
            model: 'l2' (cho mean shift), 'rbf', 'normal'
            pen: Penalty (phạt), càng cao càng ít điểm gãy (quan trọng với chuỗi ngắn)
        """
        results = []
        series_ids = self.data['series_id'].unique()

        for sid in series_ids:
            signal, years = self._get_series(sid)
            if signal is None or len(signal) < 3: # Cần tối thiểu 3 điểm
                continue

            try:
                # Chọn thuật toán
                if method == "pelt":
                    # min_size=1 để bắt được gãy ở chuỗi cực ngắn (3 năm)
                    algo = rpt.Pelt(model=model, min_size=1).fit(signal)
                    # pen nhỏ (1-2) để nhạy hơn với chuỗi ngắn
                    bkps = algo.predict(pen=pen)
                elif method == "binseg":
                    algo = rpt.Binseg(model=model, min_size=1).fit(signal)
                    bkps = algo.predict(n_bkps=1) # Chỉ tìm 1 điểm gãy
                else:
                    bkps = []

                # Chuyển index thành năm (bkps trả về index của điểm gãy)
                # Lưu ý: ruptures trả về index kết thúc đoạn, nên cần -1 để lấy điểm gãy thực tế
                detected_years = []
                for b in bkps:
                    if b < len(years):
                        detected_years.append(int(years[b])) 
                    elif b == len(years): # Trường hợp gãy ở cuối
                         detected_years.append(int(years[-1]))

                # Kiểm tra xem có trúng mục tiêu (2025) không
                hit = any(abs(y - self.target_year) <= self.tolerance for y in detected_years)
                hit_year = next((y for y in detected_years if abs(y - self.target_year) <= self.tolerance), None)

                results.append({
                    "series_id": sid,
                    "algorithm": f"Ruptures_{method}_{model}",
                    "detected_years": detected_years,
                    "hit_target": hit,
                    "hit_year": hit_year
                })

            except Exception as e:
                print(f"Error Ruptures {sid}: {e}")
                continue

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 2. THUẬT TOÁN CUSUM (Cumulative Sum Control Chart)
    # -------------------------------------------------------------------------
    def detect_cusum(self, threshold_std=1.0) -> pd.DataFrame:
        """
        Chạy thuật toán CUSUM để phát hiện sự trượt trung bình.
        Threshold tính bằng số lần độ lệch chuẩn (std).
        """
        results = []
        series_ids = self.data['series_id'].unique()

        for sid in series_ids:
            signal, years = self._get_series(sid)
            if signal is None or len(signal) < 3:
                continue
            
            # Tính CUSUM
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            if std_val == 0: std_val = 1e-9 # Tránh chia 0
            
            # Chuẩn hóa
            standardized = (signal - mean_val) / std_val
            cusum = np.cumsum(standardized)
            
            # Tìm điểm vượt ngưỡng (Change Point)
            # Logic: Khi CUSUM đảo chiều mạnh hoặc vượt ngưỡng
            # Đơn giản hóa: Tìm năm mà giá trị thay đổi đột ngột nhất
            diffs = np.diff(signal)
            # Điểm gãy là nơi có bước nhảy lớn nhất về giá trị tuyệt đối
            max_change_idx = np.argmax(np.abs(diffs)) + 1 # +1 vì diff giảm 1 phần tử
            
            detected_year = int(years[max_change_idx])
            
            # Kiểm tra trúng đích
            hit = abs(detected_year - self.target_year) <= self.tolerance

            results.append({
                "series_id": sid,
                "algorithm": "CUSUM_Mean",
                "detected_years": [detected_year],
                "hit_target": hit,
                "hit_year": detected_year if hit else None,
                "cusum_values": list(cusum) # Lưu để vẽ nếu cần
            })
            
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # 3. TỔNG HỢP & PHÂN TÍCH
    # -------------------------------------------------------------------------
    def analyze_all(self):
        """Chạy cả PELT và CUSUM, tổng hợp kết quả."""
        df_pelt = self.detect_ruptures(method="pelt", pen=1) # Pen nhỏ cho chuỗi ngắn
        df_cusum = self.detect_cusum()
        
        # Gộp kết quả
        final_df = pd.concat([df_pelt, df_cusum], ignore_index=True)
        self.results = final_df
        return final_df

    # -------------------------------------------------------------------------
    # 4. TRỰC QUAN HÓA (VISUALIZATION)
    # -------------------------------------------------------------------------
    def plot_enhanced(self, series_id, detected_years):
        """
        Vẽ biểu đồ phân tích điểm gãy chuyên sâu:
        - Hiển thị Mean Level trước/sau điểm gãy.
        - Chú thích % thay đổi.
        """
        signal, years = self._get_series(series_id)
        if signal is None:
            print(f"Không tìm thấy dữ liệu cho {series_id}")
            return
            
        # Nếu không có điểm gãy, vẽ biểu đồ thường
        if not detected_years or len(detected_years) == 0:
            print(f"Không có điểm gãy để vẽ nâng cao cho {series_id}. Dùng biểu đồ cơ bản.")
            self.plot_series_with_cp(series_id, "Original Data", [])
            return

        cp = detected_years[0] # Lấy điểm gãy đầu tiên làm trọng tâm
        
        # Tách dữ liệu: Trước và Sau điểm gãy
        # Logic: Điểm gãy là năm bắt đầu của giai đoạn mới
        mask_before = years < cp
        mask_after = years >= cp
        
        # Tính toán thống kê
        if not (any(mask_before) and any(mask_after)):
            print("Điểm gãy nằm ở biên, không tính được mean 2 giai đoạn.")
            self.plot_series_with_cp(series_id, "Basic", detected_years)
            return

        mean_before = np.mean(signal[mask_before])
        mean_after = np.mean(signal[mask_after])
        pct_change = ((mean_after - mean_before) / mean_before) * 100

        # ---  VẼ ---
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")
        
        # 1. Vẽ đường dữ liệu gốc (Màu xanh đậm)
        plt.plot(years, signal, marker='o', color='#2c3e50', linewidth=3, label='Điểm trung bình', zorder=3)
        
        # 2. Vẽ điểm gãy (Nét đứt màu đỏ)
        plt.axvline(x=cp, color='#e74c3c', linestyle='--', linewidth=2, label=f'Điểm gãy ({cp})', zorder=2)
        
        # 3. Vẽ các mức trung bình (Step lines - đường bậc thang)
        # Đường xanh (Trước gãy)
        plt.hlines(y=mean_before, xmin=min(years), xmax=cp, colors='#27ae60', linestyles=':', linewidth=2, label=f'Mean Trước: {mean_before:.2f}')
        # Đường đỏ (Sau gãy)
        plt.hlines(y=mean_after, xmin=cp, xmax=max(years), colors='#c0392b', linestyles=':', linewidth=2, label=f'Mean Sau: {mean_after:.2f}')
        
        # 4. Annotation: Mũi tên chỉ hướng thay đổi
        mid_x = cp
        # Vị trí giữa của mũi tên
        mid_y = (mean_before + mean_after) / 2
        
        plt.annotate('', xy=(cp, mean_after), xytext=(cp, mean_before),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3, mutation_scale=15),
                     zorder=4)
        
        # 5. Annotation: Hộp Text hiển thị % thay đổi
        bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#e74c3c", alpha=0.9, lw=1.5)
        change_text = f"{pct_change:+.1f}%"
        
        # Đặt text lệch sang phải một chút để không che đường kẻ
        plt.text(cp + 0.05, mid_y, change_text, ha='left', va='center', 
                 color='#c0392b', fontsize=13, fontweight='bold', bbox=bbox_props, zorder=5)

        # 6. Tô màu nền vùng "Kỷ nguyên mới"
        plt.axvspan(cp, max(years)+0.5, color='#e74c3c', alpha=0.08, label='Giai đoạn mới')

        # Trang trí
        plt.title(f"Phân tích Điểm gãy: Môn {series_id} ({min(years)}-{max(years)})", fontsize=15, fontweight='bold', pad=15)
        plt.ylabel("Điểm trung bình", fontsize=12)
        plt.xlabel("Năm học", fontsize=12)
        
        # Chỉ hiện các năm số nguyên trên trục X
        plt.xticks(years, [str(y) for y in years]) 
        
        plt.legend(loc='upper left', frameon=True, shadow=True)
        plt.tight_layout()
        plt.show()