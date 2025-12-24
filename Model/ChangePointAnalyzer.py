import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ChangePointAnalyzer:
    """
    Class phân tích sâu tác động của điểm gãy (Change Point Impact Analysis).
    So sánh thống kê giữa giai đoạn Trước và Sau điểm gãy (đặc biệt là mốc 2025).
    """

    def __init__(self, data: pd.DataFrame, target_year: int = 2025):
        """
        Args:
            data: DataFrame gốc chứa ['year', 'series_id', 'value']
            target_year: Năm mốc để chia cắt dữ liệu (mặc định 2025)
        """
        self.data = data
        self.target_year = target_year
        self.summary_table = pd.DataFrame()

    def _get_series(self, series_id):
        df_sub = self.data[self.data['series_id'] == series_id].sort_values('year')
        return df_sub

    def _calculate_cohens_d(self, group1, group2):
        """
        Tính Cohen's d (Effect Size).
        Logic:
        - Nếu cả 2 nhóm đều có nhiều phần tử: Dùng pooled standard deviation.
        - Nếu nhóm 2 (Post) chỉ có 1 phần tử (VD: chỉ năm 2025): Dùng công thức z-score chuẩn hóa theo nhóm 1.
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        if n1 < 2: return 0.0 # Không đủ dữ liệu để so sánh

        # Trường hợp 1: Nhóm 'Sau' chỉ có 1 năm (VD: 2025)
        if n2 == 1:
            std1 = np.sqrt(var1)
            if std1 == 0: return 0.0
            # d = (Value - Mean_Pre) / SD_Pre
            return (mean2 - mean1) / std1

        # Trường hợp 2: Cả 2 nhóm đều có nhiều năm (Standard Cohen's d)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
        
        if pooled_std == 0: return 0.0
        return (mean2 - mean1) / pooled_std

    def _interpret_effect_size(self, d_value):
        """Phân loại mức độ ảnh hưởng dựa trên Cohen's d."""
        abs_d = abs(d_value)
        if abs_d < 0.2: return "Không đáng kể"
        elif abs_d < 0.5: return "Yếu (Weak)"
        elif abs_d < 0.8: return "Vừa (Medium)"
        else: return "Mạnh (Strong)"

    def analyze_impact(self, detected_results: pd.DataFrame) -> pd.DataFrame:
        """
        Thực hiện T-test và tính Effect Size cho các chuỗi có điểm gãy.
        
        Args:
            detected_results: DataFrame kết quả từ ChangePointDetector (cần cột 'series_id' và 'hit_year')
        
        Returns:
            DataFrame tổng hợp (Summary Table).
        """
        analysis_list = []

        # Chỉ lấy những dòng đã phát hiện trúng target_year (hoặc gần đó)
        target_hits = detected_results[detected_results['hit_target'] == True]
        
        if target_hits.empty:
            print("⚠️ Không có chuỗi nào có điểm gãy tại năm mục tiêu để phân tích.")
            return pd.DataFrame()

        for _, row in target_hits.iterrows():
            sid = row['series_id']
            cp_year = int(row['hit_year']) # Năm gãy thực tế (ví dụ 2025)

            df_series = self._get_series(sid)
            if df_series.empty: continue

            # Tách dữ liệu: Trước và Sau
            # Lưu ý: "Trước" thường lấy khoảng gần (ví dụ 3-4 năm trước) để so sánh cục bộ
            # Ở đây ta lấy toàn bộ dữ liệu có sẵn trước đó hoặc giới hạn từ 2020 trở đi nếu muốn
            pre_data = df_series[df_series['year'] < cp_year]['value'].values
            post_data = df_series[df_series['year'] >= cp_year]['value'].values

            if len(pre_data) == 0 or len(post_data) == 0:
                continue

            # 1. Tính Mean, Variance
            mean_pre = np.mean(pre_data)
            var_pre = np.var(pre_data, ddof=1) if len(pre_data) > 1 else 0
            
            mean_post = np.mean(post_data)
            var_post = np.var(post_data, ddof=1) if len(post_data) > 1 else 0

            # 2. T-test (Kiểm định giả thuyết)
            # Nếu Post chỉ có 1 mẫu (chỉ năm 2025), dùng 1-sample t-test so với mean của Pre
            if len(post_data) == 1:
                t_stat, p_val = stats.ttest_1samp(pre_data, popmean=mean_post)
            else:
                # Independent t-test (giả định phương sai không bằng nhau - Welch's t-test)
                t_stat, p_val = stats.ttest_ind(pre_data, post_data, equal_var=False)

            # 3. Cohen's d (Effect Size)
            d_val = self._calculate_cohens_d(pre_data, post_data)
            magnitude = self._interpret_effect_size(d_val)

            analysis_list.append({
                "series_id": sid,
                "break_point": cp_year,
                "mean_pre": round(mean_pre, 2),
                "mean_post": round(mean_post, 2),
                "delta_pct": round(((mean_post - mean_pre)/mean_pre)*100, 2),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05, # Ý nghĩa thống kê ở mức 5%
                "cohen_d": round(d_val, 2),
                "magnitude": magnitude
            })

        self.summary_table = pd.DataFrame(analysis_list)
        return self.summary_table

    def plot_impact_visual(self, series_id: str):
        """
        Vẽ biểu đồ so sánh Trước - Sau với band màu và thông tin thống kê.
        """
        # Lấy thông tin thống kê từ bảng summary
        if self.summary_table.empty:
            print("Vui lòng chạy analyze_impact() trước.")
            return

        stat_info = self.summary_table[self.summary_table['series_id'] == series_id]
        if stat_info.empty:
            print(f"Không có dữ liệu phân tích tác động cho {series_id} (có thể không có điểm gãy 2025).")
            return
        
        # Extract data
        stats_row = stat_info.iloc[0]
        cp_year = int(stats_row['break_point'])
        d_val = stats_row['cohen_d']
        mag = stats_row['magnitude']
        pval = stats_row['p_value']
        
        # Get Time Series Data
        df = self._get_series(series_id)
        years = df['year'].values
        values = df['value'].values

        # Setup Plot
        plt.figure(figsize=(12, 6))
        sns.set_theme(style="ticks")

        # 1. Vẽ Band màu (Vùng Trước và Sau)
        # Vùng Trước (Xanh nhạt)
        plt.axvspan(years.min(), cp_year, color='green', alpha=0.1, label='Giai đoạn Trước (Stable?)')
        # Vùng Sau (Cam nhạt)
        plt.axvspan(cp_year, years.max() + 0.5, color='orange', alpha=0.15, label='Giai đoạn Sau (New Normal)')

        # 2. Vẽ đường dữ liệu
        plt.plot(years, values, marker='o', color='#34495e', linewidth=2.5, zorder=3)
        
        # 3. Vẽ điểm gãy
        plt.axvline(x=cp_year, color='red', linestyle='--', linewidth=2, label=f'Điểm gãy {cp_year}')

        # 4. Vẽ Mean Lines (Nét đứt ngang)
        plt.hlines(stats_row['mean_pre'], xmin=years.min(), xmax=cp_year, colors='green', linestyles='--')
        plt.hlines(stats_row['mean_post'], xmin=cp_year, xmax=years.max(), colors='orange', linestyles='--')

        # 5. Annotation (Thông tin thống kê)
        info_text = (
            f"Pre-Mean: {stats_row['mean_pre']}\n"
            f"Post-Mean: {stats_row['mean_post']} ({stats_row['delta_pct']}%)\n"
            f"Cohen's d: {d_val} ({mag})\n"
            f"P-value: {pval}"
        )
        
        # Box góc trên
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        plt.text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=props)

        # Trang trí
        plt.title(f"Phân tích Tác động Điểm gãy: {series_id}", fontsize=14, fontweight='bold')
        plt.xlabel("Năm học")
        plt.ylabel("Giá trị trung bình")
        plt.xticks(years, [str(y) for y in years])
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()